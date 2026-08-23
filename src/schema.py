"""Loads the canonical attribute dictionary and answers schema questions about it.

Everything downstream (extraction prompts, validation rules, scoring weights) is
driven from `schemas/attributes.json`, so swapping that file re-targets the whole
pipeline at a different output contract.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from rapidfuzz import fuzz, process

from config import ALIAS_MATCH_THRESHOLD, AMBIGUITY_MARGIN, ATTRIBUTE_SCHEMA_PATH

GENERIC_CATEGORY = "general_product"


@dataclass(frozen=True)
class AttributeSpec:
    """Normalised description of one canonical attribute."""

    key: str
    label: str
    datatype: str                     # string | number | enum | boolean | list
    unit: str | None = None
    unit_family: str | None = None
    enum: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    aliases: tuple[str, ...] = ()
    required: bool = False
    weight: int = 1
    regex: str | None = None
    scope: str = "common"             # common | category | content

    @property
    def display_unit(self) -> str:
        return self.unit or ""

    def prompt_line(self) -> str:
        """One-line description of this attribute for an LLM extraction prompt."""
        parts = [f'"{self.key}" ({self.label})', f"type={self.datatype}"]
        if self.unit:
            parts.append(f"unit={self.unit} (convert any other unit to this)")
        if self.enum:
            parts.append("allowed=" + " | ".join(self.enum))
        if self.minimum is not None or self.maximum is not None:
            parts.append(f"range=[{self.minimum}, {self.maximum}]")
        if self.required:
            parts.append("REQUIRED")
        return "  - " + "; ".join(parts)


def _spec_from_raw(key: str, raw: dict[str, Any], scope: str) -> AttributeSpec:
    return AttributeSpec(
        key=key,
        label=raw.get("label", key.replace("_", " ").title()),
        datatype=raw.get("datatype", "string"),
        unit=raw.get("unit"),
        unit_family=raw.get("unit_family"),
        enum=tuple(raw.get("enum", ())),
        minimum=raw.get("min"),
        maximum=raw.get("max"),
        aliases=tuple(a.lower() for a in raw.get("aliases", ())),
        required=bool(raw.get("required", False)),
        weight=int(raw.get("weight", 1)),
        regex=raw.get("regex"),
        scope=scope,
    )


class AttributeSchema:
    """In-memory view of the attribute dictionary with lookup helpers."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self._raw = raw
        self.version: str = raw.get("version", "0")

        self.common: dict[str, AttributeSpec] = {
            k: _spec_from_raw(k, v, "common") for k, v in raw.get("common", {}).items()
        }
        self.content: dict[str, AttributeSpec] = {
            k: _spec_from_raw(k, v, "content")
            for k, v in raw.get("content_fields", {}).items()
        }
        self.categories: dict[str, dict[str, AttributeSpec]] = {}
        self.category_labels: dict[str, str] = {}
        self.category_keywords: dict[str, tuple[str, ...]] = {}
        # Delivery templates want a three-level path, not a flat category, so
        # every leaf carries the Dept and Class it hangs off.
        self.category_dept: dict[str, str] = {}
        self.category_class: dict[str, str] = {}
        self.taxonomy_separator: str = raw.get("taxonomy", {}).get("separator", ">")

        for cat_key, cat_raw in raw.get("categories", {}).items():
            self.categories[cat_key] = {
                k: _spec_from_raw(k, v, "category")
                for k, v in cat_raw.get("attributes", {}).items()
            }
            self.category_labels[cat_key] = cat_raw.get("label", cat_key)
            self.category_keywords[cat_key] = tuple(
                kw.lower() for kw in cat_raw.get("keywords", ())
            )
            self.category_dept[cat_key] = cat_raw.get("dept", "")
            self.category_class[cat_key] = cat_raw.get("class", "")

        # alias -> canonical key, built once per (category) on demand
        self._alias_cache: dict[str, dict[str, str]] = {}

    # ------------------------------------------------------------------ lookup
    def category_keys(self) -> list[str]:
        return list(self.categories)

    def label_for(self, category: str) -> str:
        return self.category_labels.get(category, category)

    # ------------------------------------------------------------- taxonomy
    def dept_for(self, category: str | None) -> str:
        return self.category_dept.get(category or "", "")

    def class_for(self, category: str | None) -> str:
        return self.category_class.get(category or "", "")

    def fine_for(self, category: str | None) -> str:
        return self.category_labels.get(category or "", "")

    def classpath_for(self, category: str | None) -> str:
        """Dept > Class > Fine, as the delivery template expects it."""
        parts = [
            self.dept_for(category),
            self.class_for(category),
            self.fine_for(category),
        ]
        return self.taxonomy_separator.join(p for p in parts if p)

    def specs_for(self, category: str | None) -> dict[str, AttributeSpec]:
        """All attribute specs applicable to a category: common + category-specific."""
        specs: dict[str, AttributeSpec] = dict(self.common)
        if category and category in self.categories:
            specs.update(self.categories[category])
        elif GENERIC_CATEGORY in self.categories:
            specs.update(self.categories[GENERIC_CATEGORY])
        return specs

    def spec(self, key: str, category: str | None = None) -> AttributeSpec | None:
        return self.specs_for(category).get(key) or self.content.get(key)

    def required_keys(self, category: str | None) -> list[str]:
        return [k for k, s in self.specs_for(category).items() if s.required]

    # ------------------------------------------------------------- alias match
    def _alias_map(self, category: str | None) -> dict[str, str]:
        cache_key = category or "__none__"
        if cache_key not in self._alias_cache:
            mapping: dict[str, str] = {}
            for key, spec in self.specs_for(category).items():
                mapping[key.lower()] = key
                mapping[spec.label.lower()] = key
                for alias in spec.aliases:
                    mapping.setdefault(alias, key)
            self._alias_cache[cache_key] = mapping
        return self._alias_cache[cache_key]

    def resolve_attribute(
        self, raw_name: str, category: str | None = None
    ) -> tuple[str | None, float]:
        """Map a messy source attribute name to a canonical key.

        Returns (canonical_key, match_score 0-100). Exact/alias hits score 100.
        """
        if not raw_name:
            return None, 0.0
        needle = re.sub(r"[^a-z0-9 ]+", " ", raw_name.lower()).strip()
        needle = re.sub(r"\s+", " ", needle)
        alias_map = self._alias_map(category)

        if needle in alias_map:
            return alias_map[needle], 100.0

        # Take the best few, then refuse ambiguous ties. A near-tie between two
        # aliases that map to *different* canonical keys means the header is
        # genuinely unclear, and guessing there produces silent mis-mapping -
        # the most expensive kind of error in a product catalogue.
        candidates = process.extract(
            needle,
            list(alias_map.keys()),
            scorer=fuzz.WRatio,
            limit=3,
            score_cutoff=ALIAS_MATCH_THRESHOLD,
        )
        if not candidates:
            return None, 0.0

        best_alias, best_score = candidates[0][0], float(candidates[0][1])
        best_key = alias_map[best_alias]

        for alias, score, _ in candidates[1:]:
            if alias_map[alias] != best_key and best_score - float(score) < AMBIGUITY_MARGIN:
                return None, 0.0

        return best_key, best_score

    # ------------------------------------------------- keyword category guess
    def guess_category(self, *text_parts: str) -> tuple[str, float]:
        """Cheap keyword vote for category. Returns (category_key, confidence)."""
        blob = " ".join(p for p in text_parts if p).lower()
        if not blob:
            return GENERIC_CATEGORY, 0.0

        scores: dict[str, int] = {}
        for cat, keywords in self.category_keywords.items():
            for kw in keywords:
                if kw and kw in blob:
                    # longer keyword match is stronger evidence
                    scores[cat] = scores.get(cat, 0) + len(kw.split())
        if not scores:
            return GENERIC_CATEGORY, 0.0

        best = max(scores, key=lambda c: scores[c])
        total = sum(scores.values())
        return best, round(scores[best] / total, 2)

    # -------------------------------------------------------------- prompting
    def extraction_prompt_block(self, category: str | None) -> str:
        """Render the attribute contract that goes into an extraction prompt."""
        specs = self.specs_for(category)
        lines = [spec.prompt_line() for spec in specs.values()]
        return "\n".join(lines)


@lru_cache(maxsize=1)
def load_schema(path: str | None = None) -> AttributeSchema:
    """Load and cache the attribute dictionary."""
    target = path or ATTRIBUTE_SCHEMA_PATH
    with open(target, "r", encoding="utf-8") as fh:
        return AttributeSchema(json.load(fh))
