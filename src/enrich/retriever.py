"""Retrieval over the catalogue itself.

The most reliable source for a missing attribute is usually another product in
the same catalogue: the same series, the same frame size, the same manufacturer.
TF-IDF over a text projection of each record is enough for that and needs no
model download, which matters when the whole catalogue has to be indexed in
seconds rather than minutes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from config import RAG_TOP_K
from src.models import ProductRecord

logger = logging.getLogger(__name__)


@dataclass
class Neighbour:
    """A similar catalogue product and why it was considered similar."""

    record: ProductRecord
    similarity: float
    shared_attributes: list[str]


def record_to_text(record: ProductRecord) -> str:
    """Flatten a record into the text used for similarity search."""
    parts = [record.sku, record.category or ""]
    for key, av in record.all_fields().items():
        if av.is_empty:
            continue
        value = ", ".join(str(v) for v in av.value) if isinstance(av.value, list) else av.value
        parts.append(f"{key} {value} {av.unit or ''}")
    return " ".join(str(p) for p in parts)


class CatalogRetriever:
    """TF-IDF index over the catalogue, rebuilt whenever the corpus changes."""

    def __init__(self, records: Iterable[ProductRecord] | None = None) -> None:
        self._records: list[ProductRecord] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        if records:
            self.build(records)

    def build(self, records: Iterable[ProductRecord]) -> int:
        """Index the given records. Returns the corpus size."""
        self._records = [r for r in records if r.filled_keys()]
        if len(self._records) < 2:
            self._vectorizer, self._matrix = None, None
            return len(self._records)

        corpus = [record_to_text(r) for r in self._records]
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w[\w\-\./]*\b",
        )
        self._matrix = self._vectorizer.fit_transform(corpus)
        logger.info("Indexed %d catalogue records", len(self._records))
        return len(self._records)

    @property
    def size(self) -> int:
        return len(self._records)

    def similar(
        self,
        record: ProductRecord,
        *,
        top_k: int = RAG_TOP_K,
        same_category_only: bool = True,
        min_similarity: float = 0.12,
    ) -> list[Neighbour]:
        """Find catalogue products most like this one."""
        if self._vectorizer is None or self._matrix is None:
            return []

        query = self._vectorizer.transform([record_to_text(record)])
        scores = linear_kernel(query, self._matrix).ravel()

        order = np.argsort(scores)[::-1]
        neighbours: list[Neighbour] = []
        own_keys = set(record.filled_keys())

        for idx in order:
            candidate = self._records[idx]
            if candidate.record_id == record.record_id or candidate.sku == record.sku:
                continue
            score = float(scores[idx])
            if score < min_similarity:
                break
            if same_category_only and candidate.category != record.category:
                continue
            shared = sorted(own_keys & set(candidate.filled_keys()))
            neighbours.append(Neighbour(candidate, round(score, 4), shared))
            if len(neighbours) >= top_k:
                break
        return neighbours

    def neighbour_block(
        self, record: ProductRecord, missing_keys: list[str], *, top_k: int = RAG_TOP_K
    ) -> str:
        """Render similar products as prompt context, focused on the missing fields."""
        neighbours = self.similar(record, top_k=top_k)
        if not neighbours:
            return ""

        lines: list[str] = []
        for n in neighbours:
            head = f"- {n.record.sku} (similarity {n.similarity:.2f})"
            details: list[str] = []
            for key in missing_keys:
                av = n.record.get(key)
                if av and not av.is_empty and av.penalised_confidence() >= 0.6:
                    unit = f" {av.unit}" if av.unit else ""
                    details.append(f"{key}={av.value}{unit}")
            # A neighbour with nothing to offer on the missing fields is noise.
            if not details:
                continue
            for key in ("product_name", "series", "brand"):
                value = n.record.value_of(key)
                if value:
                    head += f" [{key}: {value}]"
            lines.append(head + "\n    " + "; ".join(details))

        return "\n".join(lines)

    def series_consensus(
        self, record: ProductRecord, key: str, *, min_agreement: int = 3
    ) -> tuple[Any, float, list[str]] | None:
        """Value shared by most same-series products, used for high-trust gap fill.

        Returns (value, agreement_ratio, supporting_skus) or None.
        """
        series = record.value_of("series")
        brand = record.value_of("brand")
        if not series and not brand:
            return None

        peers = [
            r
            for r in self._records
            if r.record_id != record.record_id
            and r.category == record.category
            and (
                (series and str(r.value_of("series", "")).lower() == str(series).lower())
                or (not series and str(r.value_of("brand", "")).lower() == str(brand).lower())
            )
        ]
        if len(peers) < min_agreement:
            return None

        tally: dict[str, list[str]] = {}
        for peer in peers:
            av = peer.get(key)
            if av is None or av.is_empty or av.penalised_confidence() < 0.6:
                continue
            token = str(av.value)
            tally.setdefault(token, []).append(peer.sku)

        if not tally:
            return None
        best = max(tally, key=lambda t: len(tally[t]))
        supporters = tally[best]
        ratio = len(supporters) / len(peers)
        if len(supporters) < min_agreement or ratio < 0.6:
            return None

        # Recover the original typed value rather than the stringified tally key.
        for peer in peers:
            av = peer.get(key)
            if av and str(av.value) == best:
                return av.value, round(ratio, 2), supporters[:5]
        return best, round(ratio, 2), supporters[:5]
