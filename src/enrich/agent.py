"""Enrichment: fill the gaps that extraction could not, then write the copy.

Gap filling is deliberately tiered. Peer consensus in the catalogue is checked
before the model is asked to infer anything, because a value that forty sibling
SKUs already agree on is better evidence than a language model's reasoning. Only
what survives that pass is sent to the model, and whatever the model returns is
labelled INFERRED so it can never be mistaken for an extracted fact.
"""

from __future__ import annotations

import logging
from typing import Any

from config import CONTENT_TARGETS, MIN_PUBLISH_CONFIDENCE
from src.enrich.kg import ProductGraph
from src.enrich.retriever import CatalogRetriever
from src.llm.prompts import ENRICHER_SYSTEM, content_prompt, infer_prompt
from src.llm.provider import LLMClient, LLMError
from src.models import (
    AttributeValue,
    Method,
    ProductRecord,
    Severity,
    SourceType,
    ValidationIssue,
)
from src.normalize import canonical
from src.schema import AttributeSchema

logger = logging.getLogger(__name__)

# Attributes that must never be invented; getting these wrong is a commercial risk.
NEVER_INFER = frozenset(
    {"sku", "brand", "certifications", "unspsc_code", "warranty_months", "country_of_origin"}
)

BASIS_CONFIDENCE_CAP = {
    "derived_from_attributes": 0.72,
    "engineering_standard": 0.68,
    "catalog_pattern": 0.60,
}


def _verified_block(record: ProductRecord, schema: AttributeSchema) -> str:
    """Render the attributes an enricher is allowed to treat as ground truth."""
    lines: list[str] = []
    for key, av in record.attributes.items():
        if av.is_empty or av.penalised_confidence() < 0.6:
            continue
        spec = schema.spec(key, record.category)
        label = spec.label if spec else key
        unit = f" {av.unit}" if av.unit else ""
        value = ", ".join(str(v) for v in av.value) if isinstance(av.value, list) else av.value
        lines.append(f"  {label} ({key}) = {value}{unit}  [confidence {av.penalised_confidence():.2f}]")
    return "\n".join(lines) or "  (none verified)"


def _missing_block(record: ProductRecord, schema: AttributeSchema) -> tuple[str, list[str]]:
    """List the attributes still worth attempting, weighted ones first."""
    specs = schema.specs_for(record.category)
    missing: list[str] = []
    for key, spec in specs.items():
        if key in NEVER_INFER:
            continue
        av = record.get(key)
        if av is None or av.is_empty:
            missing.append(key)

    missing.sort(key=lambda k: specs[k].weight, reverse=True)
    block = "\n".join(specs[k].prompt_line() for k in missing)
    return block, missing


# ------------------------------------------------------------- tier 1: peers
def fill_from_peers(
    record: ProductRecord,
    missing: list[str],
    schema: AttributeSchema,
    graph: ProductGraph | None,
    retriever: CatalogRetriever | None,
) -> list[str]:
    """Fill gaps where the catalogue itself already answers the question."""
    filled: list[str] = []
    specs = schema.specs_for(record.category)

    for key in list(missing):
        spec = specs.get(key)
        if spec is None:
            continue

        proposal = None
        if graph is not None:
            proposal = graph.infer_from_peers(record, key)
        if proposal is None and retriever is not None:
            proposal = retriever.series_consensus(record, key)
        if proposal is None:
            continue

        value, share, supporters = proposal
        av = AttributeValue.make(
            key=key,
            value=value,
            source_type=SourceType.CATALOG_SIBLING,
            method=Method.KG_INFER,
            source_ref="catalog_peer_consensus",
            evidence=f"{int(share * 100)}% of peers ({', '.join(supporters)}) use {value}",
            confidence=min(0.70, 0.45 + share * 0.3),
        )
        av.issues.append(
            ValidationIssue(
                code="INFERRED_FROM_PEERS",
                severity=Severity.INFO,
                message=f"Filled from catalogue peer consensus ({int(share * 100)}% agreement across {len(supporters)}+ SKUs)",
                field_key=key,
                raised_by="enricher",
            )
        )
        canonical.apply_to(av, spec)
        if not av.is_empty:
            record.attributes[key] = av
            filled.append(key)
            missing.remove(key)

    return filled


# --------------------------------------------------------------- tier 2: llm
def fill_from_model(
    record: ProductRecord,
    missing: list[str],
    schema: AttributeSchema,
    llm: LLMClient,
    retriever: CatalogRetriever | None,
) -> list[str]:
    """Ask the model to infer what remains, labelling everything it returns."""
    if not missing or not llm.is_configured:
        return []

    specs = schema.specs_for(record.category)
    missing_block = "\n".join(specs[k].prompt_line() for k in missing if k in specs)
    if not missing_block:
        return []

    similar_block = (
        retriever.neighbour_block(record, missing) if retriever is not None else ""
    )

    try:
        payload, response = llm.complete_json(
            infer_prompt(
                category_label=schema.label_for(record.category or ""),
                verified_block=_verified_block(record, schema),
                missing_block=missing_block,
                similar_block=similar_block,
            ),
            system=ENRICHER_SYSTEM,
            default={},
            temperature=0.2,
        )
    except LLMError as exc:
        logger.warning("Inference call failed for %s: %s", record.sku, exc)
        return []

    filled: list[str] = []
    for key, item in ((payload or {}).get("inferred") or {}).items():
        spec = specs.get(key)
        if spec is None or key in NEVER_INFER or not isinstance(item, dict):
            continue
        value = item.get("value")
        if value in (None, "", []):
            continue

        basis = str(item.get("basis", "catalog_pattern"))
        cap = BASIS_CONFIDENCE_CAP.get(basis, 0.55)
        stated = float(item.get("confidence", 0.5) or 0.5)

        av = AttributeValue.make(
            key=key,
            value=value,
            source_type=SourceType.INFERRED,
            method=Method.KG_INFER if basis == "catalog_pattern" else Method.LLM_GENERATE,
            source_ref=f"inference:{basis}",
            evidence=str(item.get("reasoning", ""))[:400],
            confidence=min(cap, stated),
            model_id=response.model_id,
        )
        av.issues.append(
            ValidationIssue(
                code="INFERRED_VALUE",
                severity=Severity.INFO,
                message=f"Inferred, not extracted ({basis}): {item.get('reasoning', '')}",
                field_key=key,
                raised_by="enricher",
            )
        )
        canonical.apply_to(av, spec)
        if av.is_empty or av.penalised_confidence() < MIN_PUBLISH_CONFIDENCE:
            continue
        record.attributes[key] = av
        filled.append(key)

    return filled


# ------------------------------------------------------------ commerce copy
def generate_content(
    record: ProductRecord, schema: AttributeSchema, llm: LLMClient
) -> list[str]:
    """Write descriptions, bullets and SEO fields from verified attributes only."""
    if not llm.is_configured:
        return []

    verified = _verified_block(record, schema)
    if verified.strip() == "(none verified)":
        record.record_issues.append(
            ValidationIssue(
                code="CONTENT_SKIPPED",
                severity=Severity.WARNING,
                message="No verified attributes; content generation was skipped rather than invented",
                raised_by="enricher",
            )
        )
        return []

    try:
        payload, response = llm.complete_json(
            content_prompt(
                category_label=schema.label_for(record.category or ""),
                brand=str(record.value_of("brand", "")),
                sku=record.sku,
                attribute_block=verified,
                targets=CONTENT_TARGETS,
            ),
            system=ENRICHER_SYSTEM,
            default={},
            temperature=0.4,
            max_tokens=2500,
        )
    except LLMError as exc:
        logger.warning("Content generation failed for %s: %s", record.sku, exc)
        return []

    written: list[str] = []
    for key, spec in schema.content.items():
        value = (payload or {}).get(key)
        if value in (None, "", []):
            continue

        av = AttributeValue.make(
            key=key,
            value=value,
            source_type=SourceType.INFERRED,
            method=Method.LLM_GENERATE,
            source_ref="content_generator",
            evidence="Generated from the verified attribute set",
            confidence=0.75,
            model_id=response.model_id,
        )
        canonical.apply_to(av, spec)
        for issue in _check_content_length(key, av.value):
            av.issues.append(issue)
        record.content[key] = av
        written.append(key)

    return written


def _check_content_length(key: str, value: Any) -> list[ValidationIssue]:
    """Verify generated copy against the configured length targets."""
    target = CONTENT_TARGETS.get(key)
    if not target:
        return []
    low, high = target

    if isinstance(value, list):
        count = len(value)
        if count < low:
            return [
                ValidationIssue(
                    code="CONTENT_TOO_SHORT",
                    severity=Severity.INFO,
                    message=f"{key} has {count} items, below the target of {low}",
                    field_key=key,
                    raised_by="enricher",
                )
            ]
        if count > high:
            return [
                ValidationIssue(
                    code="CONTENT_TOO_LONG",
                    severity=Severity.INFO,
                    message=f"{key} has {count} items, above the target of {high}",
                    field_key=key,
                    raised_by="enricher",
                )
            ]
        return []

    length = len(str(value))
    if length < low * 0.8:
        return [
            ValidationIssue(
                code="CONTENT_TOO_SHORT",
                severity=Severity.INFO,
                message=f"{key} is {length} characters, below the {low}-{high} target",
                field_key=key,
                raised_by="enricher",
            )
        ]
    if length > high * 1.2:
        return [
            ValidationIssue(
                code="CONTENT_TOO_LONG",
                severity=Severity.INFO,
                message=f"{key} is {length} characters, above the {low}-{high} target",
                field_key=key,
                raised_by="enricher",
            )
        ]
    return []


# ------------------------------------------------------------------ pipeline
def enrich(
    record: ProductRecord,
    schema: AttributeSchema,
    llm: LLMClient,
    *,
    graph: ProductGraph | None = None,
    retriever: CatalogRetriever | None = None,
    write_content: bool = True,
) -> ProductRecord:
    """Run the full enrichment stage over one record."""
    _, missing = _missing_block(record, schema)
    before = len(record.filled_keys())

    from_peers = fill_from_peers(record, missing, schema, graph, retriever)
    from_model = fill_from_model(record, missing, schema, llm, retriever)
    content_keys = generate_content(record, schema, llm) if write_content else []

    record.log_stage(
        "enrich",
        fields_before=before,
        filled_from_peers=from_peers,
        filled_from_model=from_model,
        content_written=content_keys,
        still_missing=[k for k in missing if k not in from_model],
    )
    return record
