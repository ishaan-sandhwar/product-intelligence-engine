"""Coerces raw extracted values into the datatype the schema declares.

This is the deterministic half of data quality. Before any model is asked to
judge a value, the value is parsed, unit-converted, enum-snapped and range-checked
by code, because those failures are cheap to catch and expensive to miss.
"""

from __future__ import annotations

import re
from typing import Any

from rapidfuzz import fuzz, process

from src.models import AttributeValue, Method, Severity, ValidationIssue
from src.normalize.units import awg_to_mm2, canonical_unit_for, normalize_quantity
from src.schema import AttributeSpec

TRUE_TOKENS = {
    "yes", "y", "true", "t", "1", "available", "present", "included",
    "armoured", "armored", "with", "supported", "fitted",
}
FALSE_TOKENS = {
    "no", "n", "false", "f", "0", "none", "not available", "absent",
    "unarmoured", "unarmored", "without", "not included", "na", "n/a",
}

LIST_SPLIT = re.compile(r"\s*[;,/|]\s*|\s+and\s+|\s*\n\s*|\s*•\s*")

ENUM_SNAP_THRESHOLD = 82   # fuzzy score needed to accept an enum coercion

# Enum aliases that fuzzy matching alone gets wrong.
ENUM_HINTS: dict[str, dict[str, str]] = {
    "phase": {
        "1": "Single Phase", "1ph": "Single Phase", "1-ph": "Single Phase",
        "single": "Single Phase", "1 phase": "Single Phase",
        "3": "Three Phase", "3ph": "Three Phase", "3-ph": "Three Phase",
        "three": "Three Phase", "3 phase": "Three Phase", "tri-phase": "Three Phase",
        "dc": "DC", "direct current": "DC",
    },
    "poles": {
        "1": "1P", "2": "2P", "3": "3P", "4": "4P",
        "single pole": "1P", "double pole": "2P", "triple pole": "3P",
        "tp": "3P", "dp": "2P", "sp": "1P", "tpn": "3P+N", "spn": "1P+N", "fp": "4P",
    },
    "seal_type": {
        "2rs": "2RS (Rubber Sealed)", "rs2": "2RS (Rubber Sealed)",
        "zz": "ZZ (Metal Shielded)", "2z": "ZZ (Metal Shielded)",
        "rs": "RS (Single Seal)", "z": "Z (Single Shield)",
        "sealed": "2RS (Rubber Sealed)", "shielded": "ZZ (Metal Shielded)",
        "open": "Open",
    },
}


class CanonicalisationResult:
    """Outcome of coercing one raw value, with any issues raised on the way."""

    __slots__ = ("value", "unit", "issues", "meta", "changed")

    def __init__(
        self,
        value: Any,
        unit: str | None,
        issues: list[ValidationIssue],
        meta: dict[str, Any],
        changed: bool,
    ) -> None:
        self.value = value
        self.unit = unit
        self.issues = issues
        self.meta = meta
        self.changed = changed


def _issue(code: str, message: str, key: str, severity: Severity, suggestion: Any = None):
    return ValidationIssue(
        code=code, severity=severity, message=message, field_key=key, suggestion=suggestion
    )


def _coerce_boolean(raw: Any, key: str) -> tuple[Any, list[ValidationIssue]]:
    if isinstance(raw, bool):
        return raw, []
    token = str(raw).strip().lower()
    if token in TRUE_TOKENS:
        return True, []
    if token in FALSE_TOKENS:
        return False, []
    # Sometimes the presence of the word itself is the signal, e.g. "SWA armoured".
    if any(t in token for t in ("armour", "armor", "retardant", "resistant")):
        return True, []
    return None, [
        _issue("UNPARSEABLE_BOOLEAN", f"Cannot read '{raw}' as yes/no", key, Severity.WARNING)
    ]


def _coerce_list(raw: Any, key: str) -> tuple[list[str], list[ValidationIssue]]:
    if isinstance(raw, (list, tuple, set)):
        items = [str(x).strip() for x in raw if str(x).strip()]
    else:
        items = [p.strip() for p in LIST_SPLIT.split(str(raw)) if p.strip()]
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        low = item.lower()
        if low not in seen:
            seen.add(low)
            unique.append(item)
    return unique, []


def _coerce_enum(
    raw: Any, spec: AttributeSpec
) -> tuple[Any, list[ValidationIssue]]:
    token = str(raw).strip()
    if token in spec.enum:
        return token, []

    low = token.lower()
    hints = ENUM_HINTS.get(spec.key, {})
    if low in hints:
        return hints[low], []
    for hint_key, hint_value in hints.items():
        if hint_key in low:
            return hint_value, []

    # Exact case-insensitive hit.
    for option in spec.enum:
        if option.lower() == low:
            return option, []
    # Substring hit, e.g. "IP67 rated" -> "IP67".
    for option in spec.enum:
        if option.lower() in low or low in option.lower():
            return option, []

    match = process.extractOne(
        token, list(spec.enum), scorer=fuzz.WRatio, score_cutoff=ENUM_SNAP_THRESHOLD
    )
    if match:
        return match[0], [
            _issue(
                "ENUM_SNAPPED",
                f"Snapped '{token}' to '{match[0]}' (similarity {match[1]:.0f}%)",
                spec.key,
                Severity.INFO,
            )
        ]

    return token, [
        _issue(
            "ENUM_NOT_ALLOWED",
            f"'{token}' is not one of: {', '.join(spec.enum)}",
            spec.key,
            Severity.WARNING,
        )
    ]


def _coerce_number(
    raw: Any, spec: AttributeSpec
) -> tuple[Any, str | None, list[ValidationIssue], dict[str, Any]]:
    issues: list[ValidationIssue] = []

    # Cable gauges arrive as "12 AWG" and must become mm2 before anything else.
    if spec.unit_family == "area" and isinstance(raw, str):
        from_awg = awg_to_mm2(raw)
        if from_awg is not None:
            return (
                from_awg,
                "mm2",
                [
                    _issue(
                        "AWG_CONVERTED",
                        f"Converted {raw} to {from_awg} mm2",
                        spec.key,
                        Severity.INFO,
                    )
                ],
                {"converted": True, "from": str(raw), "to": f"{from_awg} mm2"},
            )

    value, unit, meta = normalize_quantity(raw, spec.unit_family, spec.unit)

    if value is None:
        issues.append(
            _issue(
                "UNPARSEABLE_NUMBER",
                f"No numeric value found in '{raw}'",
                spec.key,
                Severity.WARNING,
            )
        )
        return None, unit, issues, meta

    if "error" in meta:
        issues.append(
            _issue("UNIT_UNKNOWN", str(meta["error"]), spec.key, Severity.WARNING)
        )
    if meta.get("converted"):
        issues.append(
            _issue(
                "UNIT_CONVERTED",
                f"Converted {meta['from']} to {meta['to']}",
                spec.key,
                Severity.INFO,
            )
        )
    if meta.get("range"):
        issues.append(
            _issue(
                "RANGE_COLLAPSED",
                f"Source gave a range {meta['range']}; stored the midpoint",
                spec.key,
                Severity.INFO,
            )
        )

    # Range check against the schema envelope.
    if spec.minimum is not None and value < spec.minimum:
        issues.append(
            _issue(
                "OUT_OF_RANGE",
                f"{value} {unit or ''} is below the minimum {spec.minimum}",
                spec.key,
                Severity.ERROR,
            )
        )
    if spec.maximum is not None and value > spec.maximum:
        issues.append(
            _issue(
                "OUT_OF_RANGE",
                f"{value} {unit or ''} exceeds the maximum {spec.maximum}",
                spec.key,
                Severity.ERROR,
            )
        )

    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return value, unit, issues, meta


def _coerce_string(raw: Any, spec: AttributeSpec) -> tuple[Any, list[ValidationIssue]]:
    text = " ".join(str(raw).split()).strip(" .;,-")
    issues: list[ValidationIssue] = []
    if spec.regex and text and not re.fullmatch(spec.regex, text, re.IGNORECASE):
        issues.append(
            _issue(
                "PATTERN_MISMATCH",
                f"'{text}' does not match the expected pattern for {spec.label}",
                spec.key,
                Severity.WARNING,
            )
        )
    return text, issues


def canonicalise(raw: Any, spec: AttributeSpec) -> CanonicalisationResult:
    """Coerce one raw value into the schema datatype, reporting what happened."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return CanonicalisationResult(None, spec.unit, [], {}, False)

    original = raw
    meta: dict[str, Any] = {}
    unit = spec.unit or canonical_unit_for(spec.unit_family)

    if spec.datatype == "number":
        value, unit, issues, meta = _coerce_number(raw, spec)
    elif spec.datatype == "enum":
        value, issues = _coerce_enum(raw, spec)
        unit = None
    elif spec.datatype == "boolean":
        value, issues = _coerce_boolean(raw, spec.key)
        unit = None
    elif spec.datatype == "list":
        value, issues = _coerce_list(raw, spec.key)
        unit = None
    else:
        value, issues = _coerce_string(raw, spec)

    changed = str(value) != str(original)
    return CanonicalisationResult(value, unit, issues, meta, changed)


def apply_to(av: AttributeValue, spec: AttributeSpec) -> AttributeValue:
    """Canonicalise an AttributeValue in place, preserving the raw value."""
    result = canonicalise(av.value, spec)
    if av.raw_value is None:
        av.raw_value = av.value

    av.value = result.value
    av.unit = result.unit
    av.issues.extend(result.issues)

    if result.changed and result.value is not None:
        # A conversion is a derived step; record it without destroying the origin.
        if any(i.code in ("UNIT_CONVERTED", "AWG_CONVERTED") for i in result.issues):
            av.provenance.method = Method.UNIT_CONVERT

    # An unparseable value should not keep a high confidence.
    if result.value is None and av.raw_value:
        av.confidence = min(av.confidence, 0.25)
    return av
