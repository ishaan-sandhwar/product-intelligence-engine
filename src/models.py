"""Core data structures. Every attribute value carries its own provenance.

The design rule: no bare values anywhere in the pipeline. A value without a
source, a method and a confidence cannot be published, which is what makes the
final catalogue auditable field-by-field.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from config import METHOD_BASE_CONFIDENCE, SOURCE_PRECEDENCE


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------- enums
class SourceType(str, Enum):
    """Where a value physically came from."""

    INPUT = "input"                     # the supplied catalogue row
    PDF = "pdf"                         # spec sheet / datasheet
    IMAGE = "image"                     # product photo, nameplate, label
    WEB = "web"                         # manufacturer or distributor page
    CATALOG_SIBLING = "catalog_sibling" # inferred from similar SKUs in-catalogue
    INFERRED = "inferred"               # model-generated, no direct source
    HUMAN = "human"                     # reviewer edit


class Method(str, Enum):
    """How the value was obtained. Drives the base confidence."""

    INPUT_FIELD = "input_field"
    REGEX = "regex"
    TABLE_EXTRACT = "table_extract"
    LLM_EXTRACT = "llm_extract"
    VLM = "vlm"
    WEB_EXTRACT = "web_extract"
    UNIT_CONVERT = "unit_convert"
    KG_INFER = "kg_infer"
    LLM_GENERATE = "llm_generate"
    HUMAN_EDIT = "human_edit"


class Severity(str, Enum):
    ERROR = "error"       # blocks publication
    WARNING = "warning"   # publishes, but flagged
    INFO = "info"


class ReviewState(str, Enum):
    AUTO_APPROVED = "auto_approved"
    NEEDS_REVIEW = "needs_review"
    HUMAN_APPROVED = "human_approved"
    HUMAN_REJECTED = "human_rejected"


# ----------------------------------------------------------------- provenance
class Provenance(BaseModel):
    """The audit trail for a single value."""

    source_type: SourceType
    source_ref: str = ""                 # "datasheet.pdf#page=3", "img_front.jpg", a URL
    evidence: str | None = None          # verbatim snippet the value was read from
    locator: str | None = None           # page/bbox/table-cell, when available
    method: Method = Method.LLM_EXTRACT
    extracted_at: str = Field(default_factory=_now)
    model_id: str | None = None          # which LLM produced it, if any

    @property
    def precedence(self) -> int:
        return SOURCE_PRECEDENCE.get(self.source_type.value, 0)

    def short(self) -> str:
        ref = self.source_ref or self.source_type.value
        return f"{self.source_type.value}:{ref}"


class ValidationIssue(BaseModel):
    """A single rule violation attached to a field or to the record."""

    code: str                            # "OUT_OF_RANGE", "UNIT_MISMATCH", ...
    severity: Severity = Severity.WARNING
    message: str
    field_key: str | None = None
    suggestion: Any | None = None        # auto-fix proposal, if the rule has one
    raised_by: str = "rule_engine"       # rule_engine | llm_judge | conflict_resolver


class AttributeValue(BaseModel):
    """One canonical attribute: its value plus everything needed to defend it."""

    key: str
    value: Any = None
    unit: str | None = None
    raw_value: Any = None                # pre-normalisation, for the audit trail
    confidence: float = 0.5
    provenance: Provenance
    issues: list[ValidationIssue] = Field(default_factory=list)
    review_state: ReviewState = ReviewState.AUTO_APPROVED
    alternatives: list[dict[str, Any]] = Field(default_factory=list)
    # ^ rejected candidates from conflict resolution: {value, unit, provenance, confidence}

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    @property
    def is_empty(self) -> bool:
        return self.value is None or self.value == "" or self.value == []

    @property
    def has_error(self) -> bool:
        return any(i.severity == Severity.ERROR for i in self.issues)

    def penalised_confidence(self) -> float:
        """Confidence after subtracting a penalty for each open issue."""
        penalty = sum(
            {Severity.ERROR: 0.35, Severity.WARNING: 0.12, Severity.INFO: 0.02}[i.severity]
            for i in self.issues
        )
        return max(0.0, round(self.confidence - penalty, 3))

    @classmethod
    def make(
        cls,
        key: str,
        value: Any,
        *,
        source_type: SourceType,
        method: Method,
        source_ref: str = "",
        evidence: str | None = None,
        unit: str | None = None,
        confidence: float | None = None,
        model_id: str | None = None,
        locator: str | None = None,
    ) -> AttributeValue:
        """Build a value with method-derived default confidence."""
        base = METHOD_BASE_CONFIDENCE.get(method.value, 0.5)
        return cls(
            key=key,
            value=value,
            unit=unit,
            raw_value=value,
            confidence=base if confidence is None else confidence,
            provenance=Provenance(
                source_type=source_type,
                source_ref=source_ref,
                evidence=evidence,
                method=method,
                model_id=model_id,
                locator=locator,
            ),
        )


# --------------------------------------------------------------- source docs
class SourceDocument(BaseModel):
    """A raw input artefact attached to a product before extraction."""

    doc_id: str = Field(default_factory=lambda: _new_id("doc"))
    kind: Literal["pdf", "image", "web", "text", "table"]
    name: str
    path: str | None = None
    url: str | None = None
    text: str | None = None              # extracted plain text
    tables: list[list[list[str]]] = Field(default_factory=list)
    page_count: int = 0
    checksum: str | None = None

    def fingerprint(self) -> str:
        payload = (self.text or "") + (self.url or "") + self.name
        return hashlib.sha256(payload.encode("utf-8", "ignore")).hexdigest()[:16]


# ------------------------------------------------------------ quality scoring
class QualityScore(BaseModel):
    """Composite data-quality score with its four components, all 0-100."""

    completeness: float = 0.0
    accuracy: float = 0.0
    consistency: float = 0.0
    richness: float = 0.0
    overall: float = 0.0
    detail: dict[str, Any] = Field(default_factory=dict)

    def grade(self) -> str:
        if self.overall >= 90:
            return "A"
        if self.overall >= 75:
            return "B"
        if self.overall >= 60:
            return "C"
        if self.overall >= 40:
            return "D"
        return "F"


# ------------------------------------------------------------- product record
class ProductRecord(BaseModel):
    """The golden record: canonical attributes, content, evidence and scores."""

    record_id: str = Field(default_factory=lambda: _new_id("prd"))
    sku: str
    category: str | None = None
    category_confidence: float = 0.0

    attributes: dict[str, AttributeValue] = Field(default_factory=dict)
    content: dict[str, AttributeValue] = Field(default_factory=dict)

    sources: list[SourceDocument] = Field(default_factory=list)
    record_issues: list[ValidationIssue] = Field(default_factory=list)

    quality_before: QualityScore | None = None
    quality_after: QualityScore | None = None

    review_state: ReviewState = ReviewState.AUTO_APPROVED
    stage_log: list[dict[str, Any]] = Field(default_factory=list)

    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)

    # -------------------------------------------------------------- accessors
    def all_fields(self) -> dict[str, AttributeValue]:
        """Attributes and content merged into one view."""
        return {**self.attributes, **self.content}

    def get(self, key: str) -> AttributeValue | None:
        return self.attributes.get(key) or self.content.get(key)

    def value_of(self, key: str, default: Any = None) -> Any:
        av = self.get(key)
        return default if av is None or av.is_empty else av.value

    def set_attribute(self, av: AttributeValue) -> None:
        target = self.content if av.key in self.content else self.attributes
        target[av.key] = av
        self.updated_at = _now()

    def filled_keys(self) -> list[str]:
        return [k for k, v in self.all_fields().items() if not v.is_empty]

    def needs_review(self) -> list[AttributeValue]:
        return [
            v for v in self.all_fields().values()
            if v.review_state == ReviewState.NEEDS_REVIEW or v.has_error
        ]

    def all_issues(self) -> list[ValidationIssue]:
        out = list(self.record_issues)
        for av in self.all_fields().values():
            out.extend(av.issues)
        return out

    def log_stage(self, stage: str, **details: Any) -> None:
        self.stage_log.append({"stage": stage, "at": _now(), **details})

    # ------------------------------------------------------------- export
    def to_flat_dict(self) -> dict[str, Any]:
        """Commerce-ready flat row: value columns only, for CSV/PIM export."""
        row: dict[str, Any] = {"sku": self.sku, "category": self.category}
        for key, av in self.all_fields().items():
            row[key] = av.value
            if av.unit:
                row[f"{key}__unit"] = av.unit
        row["quality_score"] = self.quality_after.overall if self.quality_after else None
        row["review_state"] = self.review_state.value
        return row

    def to_audit_dict(self) -> dict[str, Any]:
        """Full traceable export: every field with source, evidence, confidence."""
        def _field(av: AttributeValue) -> dict[str, Any]:
            return {
                "value": av.value,
                "unit": av.unit,
                "confidence": av.penalised_confidence(),
                "source_type": av.provenance.source_type.value,
                "source_ref": av.provenance.source_ref,
                "evidence": av.provenance.evidence,
                "method": av.provenance.method.value,
                "model": av.provenance.model_id,
                "review_state": av.review_state.value,
                "issues": [i.model_dump() for i in av.issues],
                "alternatives": av.alternatives,
            }

        return {
            "record_id": self.record_id,
            "sku": self.sku,
            "category": self.category,
            "category_confidence": self.category_confidence,
            "attributes": {k: _field(v) for k, v in self.attributes.items()},
            "content": {k: _field(v) for k, v in self.content.items()},
            "sources": [s.model_dump(exclude={"text", "tables"}) for s in self.sources],
            "quality_before": self.quality_before.model_dump() if self.quality_before else None,
            "quality_after": self.quality_after.model_dump() if self.quality_after else None,
            "record_issues": [i.model_dump() for i in self.record_issues],
            "review_state": self.review_state.value,
            "stage_log": self.stage_log,
            "updated_at": self.updated_at,
        }


class BatchResult(BaseModel):
    """Summary of one pipeline run over a catalogue slice."""

    run_id: str = Field(default_factory=lambda: _new_id("run"))
    started_at: str = Field(default_factory=_now)
    finished_at: str | None = None
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    errors: list[dict[str, str]] = Field(default_factory=list)
    avg_quality_before: float = 0.0
    avg_quality_after: float = 0.0
    fields_filled: int = 0
    fields_flagged: int = 0
    llm_calls: int = 0
    cache_hits: int = 0

    @property
    def uplift(self) -> float:
        return round(self.avg_quality_after - self.avg_quality_before, 1)
