"""Catalogue knowledge graph.

Nodes are products, brands, series, categories and attribute-value pairs. The
graph earns its place in two ways: it turns "which products share this exact
spec" into a one-hop query, and it exposes structural outliers - a product whose
attribute value nothing else in its series shares - which is a strong signal of
a data error that no per-record rule can see.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import networkx as nx

from src.models import ProductRecord

logger = logging.getLogger(__name__)

# Attributes worth turning into shared graph nodes. High-cardinality numerics
# (weight, dimensions) would create a node per product and teach us nothing.
GRAPH_ATTRIBUTES = (
    "brand", "series", "category", "phase", "efficiency_class", "frame_size",
    "mounting_type", "insulation_class", "bearing_type", "seal_type",
    "clearance_class", "valve_type", "connection_type", "actuation",
    "body_material", "sensor_type", "output_type", "thread_size",
    "conductor_material", "insulation_material", "breaker_type", "poles",
    "tripping_curve", "pump_type", "ingress_protection",
)


# Attributes that belong to a brand's own numbering, not to the category. A
# category-wide peer group is the wrong evidence for these: the abrasives group
# is dominated by 3M, so a brandless Diablo belt would inherit 3M's 775L series.
BRAND_SCOPED_ATTRIBUTES = frozenset(
    {"series", "model", "product_line", "collection", "upc", "mpn", "part_number"}
)


def _node(kind: str, value: Any) -> str:
    return f"{kind}::{str(value).strip().lower()}"


@dataclass
class Outlier:
    """A product whose attribute value is unshared within its peer group."""

    sku: str
    attribute: str
    value: Any
    peer_group: str
    peer_size: int
    majority_value: Any
    majority_share: float


class ProductGraph:
    """Undirected graph linking products to their shared characteristics."""

    def __init__(self, records: Iterable[ProductRecord] | None = None) -> None:
        self.graph = nx.Graph()
        self._records: dict[str, ProductRecord] = {}
        if records:
            self.build(records)

    # ----------------------------------------------------------------- build
    def build(self, records: Iterable[ProductRecord]) -> nx.Graph:
        """Rebuild the graph from scratch."""
        self.graph.clear()
        self._records.clear()

        for record in records:
            self._records[record.sku] = record
            product_node = _node("product", record.sku)
            self.graph.add_node(
                product_node,
                kind="product",
                sku=record.sku,
                category=record.category,
                label=str(record.value_of("product_name", record.sku))[:60],
            )

            if record.category:
                cat_node = _node("category", record.category)
                self.graph.add_node(cat_node, kind="category", label=record.category)
                self.graph.add_edge(product_node, cat_node, relation="in_category")

            for key in GRAPH_ATTRIBUTES:
                av = record.get(key)
                if av is None or av.is_empty:
                    continue
                values = av.value if isinstance(av.value, list) else [av.value]
                for value in values:
                    attr_node = _node(key, value)
                    self.graph.add_node(
                        attr_node, kind="attribute", attribute=key, value=value, label=f"{key}={value}"
                    )
                    self.graph.add_edge(
                        product_node,
                        attr_node,
                        relation="has_attribute",
                        confidence=round(av.penalised_confidence(), 3),
                    )

        logger.info(
            "Graph built: %d nodes, %d edges", self.graph.number_of_nodes(), self.graph.number_of_edges()
        )
        return self.graph

    @property
    def stats(self) -> dict[str, int]:
        kinds = Counter(data.get("kind", "?") for _, data in self.graph.nodes(data=True))
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "products": kinds.get("product", 0),
            "attribute_nodes": kinds.get("attribute", 0),
            "categories": kinds.get("category", 0),
        }

    # ---------------------------------------------------------------- query
    def products_sharing(self, attribute: str, value: Any) -> list[str]:
        """SKUs that carry a given attribute value."""
        attr_node = _node(attribute, value)
        if attr_node not in self.graph:
            return []
        return [
            self.graph.nodes[n]["sku"]
            for n in self.graph.neighbors(attr_node)
            if self.graph.nodes[n].get("kind") == "product"
        ]

    def related_products(self, sku: str, *, limit: int = 8) -> list[tuple[str, int]]:
        """Products sharing the most attribute nodes with this one."""
        product_node = _node("product", sku)
        if product_node not in self.graph:
            return []

        shared: Counter[str] = Counter()
        for attr_node in self.graph.neighbors(product_node):
            if self.graph.nodes[attr_node].get("kind") not in ("attribute", "category"):
                continue
            for peer in self.graph.neighbors(attr_node):
                peer_data = self.graph.nodes[peer]
                if peer_data.get("kind") == "product" and peer_data["sku"] != sku:
                    shared[peer_data["sku"]] += 1
        return shared.most_common(limit)

    def peer_group(self, record: ProductRecord) -> tuple[str, list[ProductRecord]]:
        """The tightest available peer set: same series, else same brand+category."""
        series = record.value_of("series")
        brand = record.value_of("brand")

        if series:
            peers = [
                r
                for r in self._records.values()
                if r.sku != record.sku
                and str(r.value_of("series", "")).lower() == str(series).lower()
            ]
            if len(peers) >= 2:
                return f"series={series}", peers

        if brand and record.category:
            peers = [
                r
                for r in self._records.values()
                if r.sku != record.sku
                and r.category == record.category
                and str(r.value_of("brand", "")).lower() == str(brand).lower()
            ]
            if len(peers) >= 2:
                return f"brand={brand}/{record.category}", peers

        peers = [
            r for r in self._records.values()
            if r.sku != record.sku and r.category == record.category
        ]
        return f"category={record.category}", peers

    # -------------------------------------------------------------- outliers
    def find_outliers(self, *, min_peer_size: int = 4, majority_threshold: float = 0.8) -> list[Outlier]:
        """Attribute values that stand alone inside an otherwise uniform peer group."""
        outliers: list[Outlier] = []

        for record in self._records.values():
            group_label, peers = self.peer_group(record)
            if len(peers) < min_peer_size:
                continue

            for key in GRAPH_ATTRIBUTES:
                av = record.get(key)
                if av is None or av.is_empty or isinstance(av.value, list):
                    continue

                tally: Counter[str] = Counter()
                for peer in peers:
                    peer_av = peer.get(key)
                    if peer_av and not peer_av.is_empty:
                        tally[str(peer_av.value)] += 1
                if sum(tally.values()) < min_peer_size:
                    continue

                majority_value, majority_count = tally.most_common(1)[0]
                share = majority_count / sum(tally.values())
                own = str(av.value)
                if share >= majority_threshold and own != majority_value:
                    outliers.append(
                        Outlier(
                            sku=record.sku,
                            attribute=key,
                            value=av.value,
                            peer_group=group_label,
                            peer_size=len(peers),
                            majority_value=majority_value,
                            majority_share=round(share, 2),
                        )
                    )
        return outliers

    # ------------------------------------------------------------ gap filling
    def infer_from_peers(
        self, record: ProductRecord, key: str, *, min_support: int = 3, min_share: float = 0.75
    ) -> tuple[Any, float, list[str]] | None:
        """Propose a value for a missing attribute from an overwhelming peer majority.

        Returns (value, share, supporting_skus) or None when there is no consensus.
        """
        group_label, peers = self.peer_group(record)
        if len(peers) < min_support:
            return None

        # A category-only group says nothing about brand-specific identifiers.
        if key in BRAND_SCOPED_ATTRIBUTES and group_label.startswith("category="):
            logger.debug(
                "peer inference for %s.%s refused: %s is not brand-scoped",
                record.sku, key, group_label,
            )
            return None

        tally: dict[str, list[str]] = defaultdict(list)
        typed: dict[str, Any] = {}
        for peer in peers:
            av = peer.get(key)
            if av is None or av.is_empty or av.penalised_confidence() < 0.6:
                continue
            token = str(av.value)
            tally[token].append(peer.sku)
            typed.setdefault(token, av.value)

        if not tally:
            return None
        total = sum(len(v) for v in tally.values())
        best_token = max(tally, key=lambda t: len(tally[t]))
        support = tally[best_token]
        share = len(support) / total

        if len(support) < min_support or share < min_share:
            return None
        return typed[best_token], round(share, 2), support[:5]

    # ------------------------------------------------------------- rendering
    def to_cytoscape(self, *, focus_sku: str | None = None, radius: int = 1) -> dict[str, Any]:
        """Export a subgraph in a shape a front-end graph widget can draw."""
        if focus_sku:
            centre = _node("product", focus_sku)
            if centre in self.graph:
                nodes = set(nx.ego_graph(self.graph, centre, radius=radius).nodes)
            else:
                nodes = set()
        else:
            nodes = set(self.graph.nodes)

        sub = self.graph.subgraph(nodes)
        return {
            "nodes": [
                {"data": {"id": n, "label": d.get("label", n), "kind": d.get("kind", "?")}}
                for n, d in sub.nodes(data=True)
            ],
            "edges": [
                {"data": {"source": u, "target": v, "relation": d.get("relation", "")}}
                for u, v, d in sub.edges(data=True)
            ],
        }
