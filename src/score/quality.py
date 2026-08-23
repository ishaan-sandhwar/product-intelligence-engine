"""Data Quality Score.

Four components, each 0-100, combined with the weights in config:

  completeness  weighted coverage of the attributes this category requires
  accuracy      confidence of what is published, penalised by open issues
  consistency   share of validation rules that passed, and unit discipline
  richness      commerce content present and within its length targets

The score is computed twice per record - once on the raw input, once on the
finished record - so the uplift the pipeline delivered is a measured number
rather than a claim.
"""

from __future__ import annotations

from typing import Any, Iterable

from config import CONTENT_TARGETS, QUALITY_WEIGHTS
from src.models import ProductRecord, QualityScore, Severity
from src.schema import AttributeSchema

# A field below this confidence is counted as present but untrustworthy.
TRUST_FLOOR = 0.5


def _completeness(record: ProductRecord, schema: AttributeSchema) -> tuple[float, dict[str, Any]]:
    """Weighted coverage of the applicable attribute set."""
    specs = schema.specs_for(record.category)
    if not specs:
        return 0.0, {}

    total_weight = 0.0
    earned = 0.0
    missing_required: list[str] = []

    for key, spec in specs.items():
        # Required attributes count double: a catalogue missing them is unusable.
        weight = float(spec.weight) * (2.0 if spec.required else 1.0)
        total_weight += weight

        av = record.get(key)
        if av is None or av.is_empty:
            if spec.required:
                missing_required.append(key)
            continue

        # Partial credit for a value that is present but weakly supported.
        credit = 1.0 if av.penalised_confidence() >= TRUST_FLOOR else 0.5
        earned += weight * credit

    score = 100.0 * earned / total_weight if total_weight else 0.0
    return round(score, 1), {
        "attributes_expected": len(specs),
        "attributes_present": len([k for k in specs if record.get(k) and not record.get(k).is_empty]),
        "missing_required": missing_required,
    }


def _accuracy(record: ProductRecord) -> tuple[float, dict[str, Any]]:
    """Weighted mean confidence of published fields, after issue penalties."""
    fields = [av for av in record.all_fields().values() if not av.is_empty]
    if not fields:
        return 0.0, {"fields_scored": 0}

    confidences = [av.penalised_confidence() for av in fields]
    mean = sum(confidences) / len(confidences)

    errors = sum(1 for av in fields if av.has_error)
    unsupported = sum(1 for av in fields if not av.provenance.evidence)

    # An error anywhere is a whole-record accuracy problem, not a local one.
    penalty = min(0.30, 0.08 * errors) + min(0.15, 0.02 * unsupported)
    score = max(0.0, (mean - penalty)) * 100.0

    return round(score, 1), {
        "fields_scored": len(fields),
        "mean_confidence": round(mean, 3),
        "fields_with_errors": errors,
        "fields_without_evidence": unsupported,
    }


def _consistency(record: ProductRecord, rule_summary: dict[str, Any] | None) -> tuple[float, dict[str, Any]]:
    """How much of the record survives cross-field and unit checks."""
    # Absence is a completeness problem, not a consistency one. Counting a
    # missing required field in both dimensions punished the same gap twice and
    # made a record score *lower* after processing than before it, purely
    # because the baseline pass never ran the rules.
    absence_codes = {"MISSING_REQUIRED", "BELOW_PUBLISH_THRESHOLD"}
    issues = [i for i in record.all_issues() if i.code not in absence_codes]

    errors = sum(1 for i in issues if i.severity == Severity.ERROR)
    warnings = sum(1 for i in issues if i.severity == Severity.WARNING)
    conflicts = sum(1 for i in issues if i.code == "SOURCE_CONFLICT")

    score = 100.0
    score -= 18.0 * errors
    score -= 6.0 * warnings
    score -= 4.0 * conflicts

    # Reward unit discipline: every numeric field should carry its canonical unit.
    numeric_fields = [
        av for av in record.all_fields().values()
        if isinstance(av.value, (int, float)) and not isinstance(av.value, bool)
    ]
    if numeric_fields:
        with_units = sum(1 for av in numeric_fields if av.unit)
        score = score * (0.85 + 0.15 * (with_units / len(numeric_fields)))

    detail: dict[str, Any] = {
        "errors": errors,
        "warnings": warnings,
        "conflicts": conflicts,
        "numeric_fields": len(numeric_fields),
    }
    if rule_summary:
        fired = len(rule_summary.get("rules_fired", []))
        applicable = max(1, rule_summary.get("rules_applicable", 1))
        detail["rules_passed"] = f"{applicable - fired}/{applicable}"

    return round(max(0.0, min(100.0, score)), 1), detail


def _richness(record: ProductRecord) -> tuple[float, dict[str, Any]]:
    """Presence and adequacy of commerce-ready content."""
    if not CONTENT_TARGETS:
        return 0.0, {}

    earned = 0.0
    per_field: dict[str, str] = {}

    for key, target in CONTENT_TARGETS.items():
        av = record.get(key)
        if av is None or av.is_empty:
            per_field[key] = "missing"
            continue

        low, high = target
        size = len(av.value) if isinstance(av.value, list) else len(str(av.value))
        if low <= size <= high:
            earned += 1.0
            per_field[key] = "ok"
        elif low * 0.7 <= size <= high * 1.3:
            earned += 0.6
            per_field[key] = f"off-target ({size})"
        else:
            earned += 0.25
            per_field[key] = f"far off-target ({size})"

    score = 100.0 * earned / len(CONTENT_TARGETS)
    return round(score, 1), {"content_fields": per_field}


def score_record(
    record: ProductRecord,
    schema: AttributeSchema,
    *,
    rule_summary: dict[str, Any] | None = None,
) -> QualityScore:
    """Compute the composite quality score for one record."""
    completeness, c_detail = _completeness(record, schema)
    accuracy, a_detail = _accuracy(record)
    consistency, x_detail = _consistency(record, rule_summary)
    richness, r_detail = _richness(record)

    weights = QUALITY_WEIGHTS
    overall = (
        completeness * weights.completeness
        + accuracy * weights.accuracy
        + consistency * weights.consistency
        + richness * weights.richness
    )

    return QualityScore(
        completeness=completeness,
        accuracy=accuracy,
        consistency=consistency,
        richness=richness,
        overall=round(overall, 1),
        detail={
            "weights": weights.as_dict(),
            "completeness": c_detail,
            "accuracy": a_detail,
            "consistency": x_detail,
            "richness": r_detail,
        },
    )


def score_catalog(records: Iterable[ProductRecord]) -> dict[str, Any]:
    """Aggregate catalogue-level statistics for the dashboard."""
    records = list(records)
    if not records:
        return {"count": 0}

    def _mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 1) if values else 0.0

    before = [r.quality_before.overall for r in records if r.quality_before]
    after = [r.quality_after.overall for r in records if r.quality_after]

    grades: dict[str, int] = {}
    for record in records:
        if record.quality_after:
            grade = record.quality_after.grade()
            grades[grade] = grades.get(grade, 0) + 1

    total_fields = sum(len(r.filled_keys()) for r in records)
    flagged = sum(len(r.needs_review()) for r in records)
    all_issues = [i for r in records for i in r.all_issues()]

    issue_counts: dict[str, int] = {}
    for issue in all_issues:
        issue_counts[issue.code] = issue_counts.get(issue.code, 0) + 1

    category_counts: dict[str, int] = {}
    for record in records:
        key = record.category or "unclassified"
        category_counts[key] = category_counts.get(key, 0) + 1

    return {
        "count": len(records),
        "avg_before": _mean(before),
        "avg_after": _mean(after),
        "uplift": round(_mean(after) - _mean(before), 1),
        "avg_completeness": _mean([r.quality_after.completeness for r in records if r.quality_after]),
        "avg_accuracy": _mean([r.quality_after.accuracy for r in records if r.quality_after]),
        "avg_consistency": _mean([r.quality_after.consistency for r in records if r.quality_after]),
        "avg_richness": _mean([r.quality_after.richness for r in records if r.quality_after]),
        "grades": dict(sorted(grades.items())),
        "total_fields": total_fields,
        "fields_flagged": flagged,
        "auto_approved_pct": round(
            100.0 * (1 - flagged / total_fields) if total_fields else 0.0, 1
        ),
        "top_issues": dict(sorted(issue_counts.items(), key=lambda kv: kv[1], reverse=True)[:12]),
        "categories": dict(sorted(category_counts.items(), key=lambda kv: kv[1], reverse=True)),
    }
