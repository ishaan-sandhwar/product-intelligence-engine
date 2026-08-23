"""Adversarial review pass over an assembled record.

The judge sees each value next to the evidence it was read from, which lets it
catch the class of error the rule engine cannot express: evidence that was
misread, a figure assigned to the wrong attribute, or a specification that is
impossible for this particular product even though it passes every range check.

It runs last, and it can only lower confidence or raise issues. It is never
allowed to write a value, because a model that both produces and approves data
is not a check on anything.
"""

from __future__ import annotations

import logging
from typing import Any

from src.llm.prompts import JUDGE_SYSTEM, judge_prompt
from src.llm.provider import LLMClient, LLMError
from src.models import ProductRecord, Severity, ValidationIssue
from src.schema import AttributeSchema

logger = logging.getLogger(__name__)

VALID_CODES = {
    "VALUE_CONTRADICTS_EVIDENCE",
    "IMPLAUSIBLE_MAGNITUDE",
    "INTERNAL_CONTRADICTION",
    "UNIT_SUSPECT",
    "WRONG_ATTRIBUTE",
    "MISSING_CRITICAL",
}

SEVERITY_MAP = {
    "error": Severity.ERROR,
    "warning": Severity.WARNING,
    "info": Severity.INFO,
}

# The judge may lower a confidence freely but may only raise it a little; a model
# talking itself into trusting its own extraction is not evidence.
MAX_UPWARD_ADJUSTMENT = 0.05


def _evidence_block(record: ProductRecord, schema: AttributeSchema, limit: int = 45) -> str:
    """Render every published field with its provenance for the auditor."""
    lines: list[str] = []
    for key, av in list(record.attributes.items())[:limit]:
        if av.is_empty:
            continue
        spec = schema.spec(key, record.category)
        label = spec.label if spec else key
        unit = f" {av.unit}" if av.unit else ""
        value = ", ".join(str(v) for v in av.value) if isinstance(av.value, list) else av.value
        evidence = (av.provenance.evidence or "no evidence quoted").replace("\n", " ")[:220]
        lines.append(
            f'{key} ({label}) = {value}{unit}  '
            f'[{av.provenance.source_type.value}/{av.provenance.method.value}] "{evidence}"'
        )
    return "\n".join(lines)


def review(
    record: ProductRecord, schema: AttributeSchema, llm: LLMClient
) -> tuple[list[ValidationIssue], str]:
    """Run the audit. Returns (issues, verdict)."""
    if not llm.is_configured:
        return [], "skipped"

    block = _evidence_block(record, schema)
    if not block.strip():
        return [], "empty"

    try:
        payload, response = llm.complete_json(
            judge_prompt(schema.label_for(record.category or ""), block),
            system=JUDGE_SYSTEM,
            default={},
            max_tokens=2000,
        )
    except LLMError as exc:
        logger.warning("Judge call failed for %s: %s", record.sku, exc)
        return [], "unavailable"

    payload = payload or {}
    issues: list[ValidationIssue] = []

    for raw in payload.get("issues") or []:
        if not isinstance(raw, dict):
            continue
        code = str(raw.get("code", "")).upper()
        if code not in VALID_CODES:
            continue
        field_key = raw.get("field_key")
        # Do not accept an issue pointing at a field the record does not have.
        if field_key and record.get(field_key) is None and code != "MISSING_CRITICAL":
            field_key = None

        issues.append(
            ValidationIssue(
                code=code,
                severity=SEVERITY_MAP.get(str(raw.get("severity", "warning")).lower(), Severity.WARNING),
                message=str(raw.get("message", ""))[:500],
                field_key=field_key,
                suggestion=raw.get("suggestion"),
                raised_by="llm_judge",
            )
        )

    _apply_confidence_adjustments(record, payload.get("confidence_adjustments") or {})

    verdict = str(payload.get("verdict", "clean"))
    record.log_stage(
        "judge",
        verdict=verdict,
        issues_raised=len(issues),
        model=response.model_id,
        cached=response.cached,
    )
    return issues, verdict


def _apply_confidence_adjustments(record: ProductRecord, adjustments: dict[str, Any]) -> None:
    """Let the judge revise confidence, downward freely and upward only slightly."""
    for key, raw in adjustments.items():
        av = record.get(key)
        if av is None:
            continue
        try:
            proposed = float(raw)
        except (TypeError, ValueError):
            continue
        proposed = max(0.0, min(1.0, proposed))

        if proposed < av.confidence:
            av.confidence = proposed
        else:
            av.confidence = min(av.confidence + MAX_UPWARD_ADJUSTMENT, proposed)
