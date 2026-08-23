"""Golden-record assembly: many candidates per field, one published value.

Resolution runs cheapest-first. Identical values agree and reinforce each other.
Values that differ only by unit are folded together. Only a genuine disagreement
between two credible sources is escalated to the model, and whatever loses is
retained on the field as an alternative so the decision stays inspectable.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from src.llm.prompts import JUDGE_SYSTEM, conflict_prompt
from src.llm.provider import LLMClient, LLMError
from src.models import AttributeValue, Severity, ValidationIssue
from src.normalize import canonical
from src.schema import AttributeSpec, AttributeSchema

logger = logging.getLogger(__name__)

# Two numbers within this relative tolerance are treated as the same figure.
NUMERIC_AGREEMENT_TOLERANCE = 0.02
# Confidence bonus per additional independent source that agrees.
AGREEMENT_BONUS = 0.06
MAX_AGREEMENT_BONUS = 0.15


def _values_agree(left: Any, right: Any) -> bool:
    """Do two canonicalised values represent the same fact?"""
    if left is None or right is None:
        return False
    if isinstance(left, bool) or isinstance(right, bool):
        return bool(left) == bool(right)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if left == right:
            return True
        scale = max(abs(float(left)), abs(float(right)), 1e-9)
        return abs(float(left) - float(right)) / scale <= NUMERIC_AGREEMENT_TOLERANCE
    if isinstance(left, list) and isinstance(right, list):
        return {str(x).lower() for x in left} == {str(x).lower() for x in right}
    return str(left).strip().lower() == str(right).strip().lower()


def _merge_lists(candidates: list[AttributeValue]) -> list[str]:
    """Union of list-valued candidates, preserving first-seen ordering."""
    seen: dict[str, str] = {}
    for av in candidates:
        items = av.value if isinstance(av.value, list) else [av.value]
        for item in items:
            token = str(item).strip()
            if token and token.lower() not in seen:
                seen[token.lower()] = token
    return list(seen.values())


def _rank(av: AttributeValue) -> tuple[int, float]:
    """Sort key: source precedence first, then confidence."""
    return (av.provenance.precedence, av.confidence)


def _as_alternative(av: AttributeValue) -> dict[str, Any]:
    return {
        "value": av.value,
        "unit": av.unit,
        "confidence": round(av.confidence, 3),
        "source_type": av.provenance.source_type.value,
        "source_ref": av.provenance.source_ref,
        "evidence": av.provenance.evidence,
        "method": av.provenance.method.value,
    }


def resolve_field(
    key: str,
    candidates: list[AttributeValue],
    spec: AttributeSpec,
    llm: LLMClient | None = None,
) -> AttributeValue | None:
    """Collapse candidates for one attribute into a single published value."""
    # Canonicalise every candidate first, so comparisons are apples-to-apples.
    usable: list[AttributeValue] = []
    for av in candidates:
        canonical.apply_to(av, spec)
        if not av.is_empty:
            usable.append(av)

    if not usable:
        return None

    if spec.datatype == "list":
        winner = max(usable, key=_rank)
        merged = _merge_lists(usable)
        winner.value = merged
        if len(usable) > 1:
            winner.confidence = min(
                1.0, winner.confidence + min(MAX_AGREEMENT_BONUS, AGREEMENT_BONUS * (len(usable) - 1))
            )
            winner.alternatives = [_as_alternative(a) for a in usable if a is not winner]
        return winner

    usable.sort(key=_rank, reverse=True)
    winner = usable[0]
    agreeing = [a for a in usable[1:] if _values_agree(a.value, winner.value)]
    disagreeing = [a for a in usable[1:] if not _values_agree(a.value, winner.value)]

    # Corroboration from independent sources raises confidence.
    independent = {a.provenance.source_type for a in agreeing} - {winner.provenance.source_type}
    if independent:
        winner.confidence = min(
            1.0, winner.confidence + min(MAX_AGREEMENT_BONUS, AGREEMENT_BONUS * len(independent))
        )

    if not disagreeing:
        winner.alternatives = [_as_alternative(a) for a in agreeing]
        return winner

    # A human edit is final; nothing overrules it.
    if winner.provenance.source_type.value == "human":
        winner.alternatives = [_as_alternative(a) for a in usable[1:]]
        return winner

    contenders = [winner] + disagreeing
    resolved = _resolve_disagreement(key, contenders, spec, llm)
    resolved.alternatives = [_as_alternative(a) for a in contenders if a is not resolved]
    return resolved


def _resolve_disagreement(
    key: str,
    contenders: list[AttributeValue],
    spec: AttributeSpec,
    llm: LLMClient | None,
) -> AttributeValue:
    """Pick between genuinely conflicting candidates."""
    top, runner_up = contenders[0], contenders[1]
    gap = top.provenance.precedence - runner_up.provenance.precedence

    def _flag(chosen: AttributeValue, note: str, severity: Severity) -> AttributeValue:
        others = ", ".join(
            f"{c.value}{(' ' + c.unit) if c.unit else ''} [{c.provenance.short()}]"
            for c in contenders
            if c is not chosen
        )
        chosen.issues.append(
            ValidationIssue(
                code="SOURCE_CONFLICT",
                severity=severity,
                message=f"Sources disagree on {spec.label}. Kept {chosen.value} ({note}). Rejected: {others}",
                field_key=key,
                raised_by="conflict_resolver",
            )
        )
        return chosen

    # A clearly more authoritative source settles it without a model call.
    if gap >= 20:
        top.confidence = min(top.confidence, 0.85)
        return _flag(top, f"higher source precedence: {top.provenance.source_type.value}", Severity.INFO)

    # Order-of-magnitude gaps are almost always a unit error, not a real conflict.
    if spec.datatype == "number" and all(
        isinstance(c.value, (int, float)) for c in (top, runner_up)
    ):
        low, high = sorted([abs(float(top.value)), abs(float(runner_up.value))])
        if low > 0 and math.isclose(high / low, round(high / low), rel_tol=0.02) and round(high / low) in (10, 100, 1000):
            top.confidence = min(top.confidence, 0.5)
            top.issues.append(
                ValidationIssue(
                    code="UNIT_SUSPECT",
                    severity=Severity.WARNING,
                    message=(
                        f"Candidates differ by a factor of {round(high / low)} "
                        f"({top.value} vs {runner_up.value}) - likely a unit error"
                    ),
                    field_key=key,
                    raised_by="conflict_resolver",
                )
            )
            return top

    if llm is None or not llm.is_configured:
        top.confidence = min(top.confidence, 0.55)
        return _flag(top, "unresolved, kept highest-confidence candidate", Severity.WARNING)

    block = "\n".join(
        f"  [{i}] {c.value}{(' ' + c.unit) if c.unit else ''}\n"
        f"      source: {c.provenance.source_type.value} ({c.provenance.source_ref})\n"
        f"      evidence: {(c.provenance.evidence or 'none')[:200]}"
        for i, c in enumerate(contenders)
    )
    try:
        payload, _ = llm.complete_json(
            conflict_prompt(spec.label, block),
            system=JUDGE_SYSTEM,
            default={},
            max_tokens=500,
        )
    except LLMError as exc:
        logger.warning("Conflict resolution call failed for %s: %s", key, exc)
        top.confidence = min(top.confidence, 0.55)
        return _flag(top, "adjudication unavailable", Severity.WARNING)

    payload = payload or {}
    index = payload.get("winner_index")
    chosen = contenders[index] if isinstance(index, int) and 0 <= index < len(contenders) else top
    chosen.confidence = min(chosen.confidence, float(payload.get("confidence", 0.7)))

    if payload.get("same_value_different_units"):
        chosen.issues.append(
            ValidationIssue(
                code="UNIT_RECONCILED",
                severity=Severity.INFO,
                message=f"Candidates were the same figure in different units. {payload.get('reasoning', '')}",
                field_key=key,
                raised_by="conflict_resolver",
            )
        )
        return chosen

    return _flag(chosen, payload.get("reasoning", "adjudicated by model"), Severity.INFO)


def build_golden_record(
    grouped: dict[str, list[AttributeValue]],
    schema: AttributeSchema,
    category: str | None,
    llm: LLMClient | None = None,
) -> tuple[dict[str, AttributeValue], int]:
    """Resolve every field. Returns (published attributes, conflict count)."""
    specs = schema.specs_for(category)
    resolved: dict[str, AttributeValue] = {}
    conflicts = 0

    for key, candidates in grouped.items():
        spec = specs.get(key) or schema.content.get(key)
        if spec is None:
            continue
        winner = resolve_field(key, candidates, spec, llm)
        if winner is None:
            continue
        if any(i.code == "SOURCE_CONFLICT" for i in winner.issues):
            conflicts += 1
        resolved[key] = winner

    return resolved, conflicts
