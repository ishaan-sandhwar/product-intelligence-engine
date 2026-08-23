"""Pipeline orchestration.

Stage order and why it is that order:

  1 classify   the category decides which attributes even exist, so nothing
               downstream can run before it is settled
  2 extract    every source independently, producing competing candidates
  3 resolve    candidates collapse into one golden value per field, with the
               losers retained as alternatives
  4 rules      deterministic cross-field checks, run before any model review
               because they are free and more reliable
  5 enrich     gaps filled from catalogue peers first, the model second
  6 judge      adversarial audit over the assembled record
  7 route      confidence and open issues decide auto-approve vs review queue
  8 score      quality measured against the same yardstick as the input

Batch mode runs the two catalogue-wide passes in the right order: everything is
extracted and resolved first, then the retriever and graph are built over that
result, and only then does enrichment run - otherwise the first products
processed would have no peers to learn from.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from config import (
    LLM_CONCURRENCY,
    MIN_PUBLISH_CONFIDENCE,
    REVIEW_CONFIDENCE_THRESHOLD,
)
from src.enrich import agent as enrich_agent
from src.enrich.kg import ProductGraph
from src.enrich.retriever import CatalogRetriever
from src.ingest import extract as extract_stage
from src.llm.provider import LLMClient
from src.models import (
    BatchResult,
    ProductRecord,
    ReviewState,
    Severity,
    ValidationIssue,
)
from src.normalize import identity
from src.schema import AttributeSchema, load_schema
from src.score import quality
from src.validate import conflicts, judge, rules

logger = logging.getLogger(__name__)

ProgressFn = Callable[[int, int, str], None]


@dataclass
class PipelineOptions:
    """Per-run switches, so the demo can trade cost against depth."""

    use_llm: bool = True
    use_vision: bool = True
    use_judge: bool = True
    write_content: bool = True
    use_peers: bool = True
    concurrency: int = LLM_CONCURRENCY
    review_threshold: float = REVIEW_CONFIDENCE_THRESHOLD
    provider: str | None = None


@dataclass
class PipelineContext:
    """Shared catalogue-level state available to every record."""

    schema: AttributeSchema
    llm: LLMClient
    options: PipelineOptions
    retriever: CatalogRetriever | None = None
    graph: ProductGraph | None = None
    stats: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------ single record
def score_baseline(record: ProductRecord, schema: AttributeSchema) -> ProductRecord:
    """Measure the record as it arrived, before anything touched it."""
    record.quality_before = quality.score_record(record, schema)
    return record


def process_record(record: ProductRecord, ctx: PipelineContext) -> ProductRecord:
    """Run stages 1-4 over one record: classify, extract, resolve, rule-check."""
    started = time.perf_counter()

    if record.quality_before is None:
        score_baseline(record, ctx.schema)

    llm = ctx.llm if ctx.options.use_llm else _NullLLM()

    # 1. classify
    extract_stage.classify(record, ctx.schema, llm if ctx.options.use_llm else None)
    extract_stage.seed_category_attribute(record, ctx.schema)

    # 2. extract
    if not ctx.options.use_vision:
        record.sources = [s for s in record.sources if s.kind != "image"]
    grouped = extract_stage.extract_all(record, ctx.schema, llm)

    # 3. resolve into a golden record
    resolved, conflict_count = conflicts.build_golden_record(
        grouped, ctx.schema, record.category, llm if ctx.options.use_llm else None
    )
    record.attributes = resolved
    # Resolution can hand the category back in the supplier's spelling, since an
    # input cell outranks an inferred value. Re-canonicalise after the swap.
    extract_stage.seed_category_attribute(record, ctx.schema)
    record.log_stage("resolve", fields=len(resolved), conflicts=conflict_count)

    # 4. deterministic rules
    rule_issues, rule_summary = rules.run_rules(record, ctx.schema)
    rules.attach_issues(record, rule_issues)
    record.log_stage("rules", **rule_summary)
    record.stage_log[-1]["_rule_summary"] = rule_summary

    record.log_stage("timing_extract_phase", seconds=round(time.perf_counter() - started, 2))
    return record


def finalise_record(record: ProductRecord, ctx: PipelineContext) -> ProductRecord:
    """Run stages 5-8: enrich, judge, route to review, score."""
    llm = ctx.llm if ctx.options.use_llm else _NullLLM()

    # 5. enrich
    enrich_agent.enrich(
        record,
        ctx.schema,
        llm,
        graph=ctx.graph if ctx.options.use_peers else None,
        retriever=ctx.retriever if ctx.options.use_peers else None,
        write_content=ctx.options.write_content,
    )

    # Re-run rules: enrichment added fields that cross-field checks must see.
    rule_issues, rule_summary = rules.run_rules(record, ctx.schema)
    _replace_rule_issues(record, rule_issues)

    # 6. judge
    if ctx.options.use_judge and ctx.options.use_llm:
        judge_issues, verdict = judge.review(record, ctx.schema, llm)
        rules.attach_issues(record, judge_issues)
    else:
        verdict = "skipped"

    # 7. route
    route_for_review(record, threshold=ctx.options.review_threshold)

    # 8. score
    record.quality_after = quality.score_record(record, ctx.schema, rule_summary=rule_summary)
    record.log_stage(
        "score",
        before=record.quality_before.overall if record.quality_before else None,
        after=record.quality_after.overall,
        grade=record.quality_after.grade(),
        verdict=verdict,
    )
    return record


def _replace_rule_issues(record: ProductRecord, fresh: list[ValidationIssue]) -> None:
    """Swap in a fresh rule pass without duplicating the previous one."""
    for av in record.all_fields().values():
        av.issues = [i for i in av.issues if i.raised_by != "rule_engine"]
    record.record_issues = [i for i in record.record_issues if i.raised_by != "rule_engine"]
    rules.attach_issues(record, fresh)


def route_for_review(record: ProductRecord, *, threshold: float) -> ProductRecord:
    """Decide what a human must look at. This is the human-in-the-loop gate."""
    flagged = 0

    for av in record.all_fields().values():
        if av.review_state in (ReviewState.HUMAN_APPROVED, ReviewState.HUMAN_REJECTED):
            continue  # a human already ruled on this field

        confidence = av.penalised_confidence()
        if av.has_error or confidence < threshold:
            av.review_state = ReviewState.NEEDS_REVIEW
            flagged += 1
        else:
            av.review_state = ReviewState.AUTO_APPROVED

        # Too weak to publish at all: keep it, but never ship it silently.
        if confidence < MIN_PUBLISH_CONFIDENCE and not av.is_empty:
            av.issues.append(
                ValidationIssue(
                    code="BELOW_PUBLISH_THRESHOLD",
                    severity=Severity.WARNING,
                    message=f"Confidence {confidence:.2f} is below the publish floor of {MIN_PUBLISH_CONFIDENCE}",
                    field_key=av.key,
                    raised_by="router",
                )
            )

    record_errors = sum(1 for i in record.record_issues if i.severity == Severity.ERROR)
    if record.review_state not in (ReviewState.HUMAN_APPROVED, ReviewState.HUMAN_REJECTED):
        record.review_state = (
            ReviewState.NEEDS_REVIEW if (flagged or record_errors) else ReviewState.AUTO_APPROVED
        )

    record.log_stage("route", fields_flagged=flagged, record_errors=record_errors)
    return record


# -------------------------------------------------------------------- batch
class _NullLLM(LLMClient):
    """Stand-in used when a run is configured to skip model calls entirely."""

    def __init__(self) -> None:  # noqa: D107 - deliberately does not call super
        from src.llm.provider import LLMStats

        self.stats = LLMStats()
        self.order = []

    @property
    def is_configured(self) -> bool:
        return False


def run_batch(
    records: list[ProductRecord],
    *,
    options: PipelineOptions | None = None,
    schema: AttributeSchema | None = None,
    progress: ProgressFn | None = None,
) -> tuple[list[ProductRecord], BatchResult]:
    """Process a whole catalogue slice. Returns (records, run summary)."""
    options = options or PipelineOptions()
    schema = schema or load_schema()
    llm = LLMClient(preferred=options.provider)

    ctx = PipelineContext(schema=schema, llm=llm, options=options)
    result = BatchResult(total=len(records))
    total_steps = len(records) * 2

    def _tick(step: int, label: str) -> None:
        if progress:
            progress(step, total_steps, label)

    # ---- phase A: extract and resolve every record --------------------------
    for record in records:
        score_baseline(record, schema)

    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, options.concurrency)) as pool:
        futures = {pool.submit(process_record, r, ctx): r for r in records}
        for future in as_completed(futures):
            record = futures[future]
            completed += 1
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001 - one bad record must not kill the run
                logger.exception("Extraction failed for %s", record.sku)
                result.failed += 1
                result.errors.append({"sku": record.sku, "stage": "extract", "error": str(exc)})
            _tick(completed, f"extracted {record.sku}")

    # ---- resolve identity before anything groups records by brand -----------
    # Peer grouping keys on brand, so this has to happen before the graph is
    # built: a record with no brand falls into a category-wide group.
    identity_counts = identity.resolve_batch(records, schema)
    logger.info(
        "Identity: %d brands, %d manufacturers filled",
        identity_counts.get("brand", 0), identity_counts.get("manufacturer", 0),
    )

    # ---- build catalogue context from what we now know ----------------------
    if options.use_peers:
        ctx.retriever = CatalogRetriever(records)
        ctx.graph = ProductGraph(records)
        logger.info("Catalogue context: %s", ctx.graph.stats)

    # ---- phase B: enrich, judge, route, score -------------------------------
    with ThreadPoolExecutor(max_workers=max(1, options.concurrency)) as pool:
        futures = {pool.submit(finalise_record, r, ctx): r for r in records}
        for future in as_completed(futures):
            record = futures[future]
            completed += 1
            try:
                future.result()
                result.succeeded += 1
            except Exception as exc:  # noqa: BLE001
                logger.exception("Enrichment failed for %s", record.sku)
                result.failed += 1
                result.errors.append({"sku": record.sku, "stage": "enrich", "error": str(exc)})
            _tick(completed, f"enriched {record.sku}")

    # ---- summary ------------------------------------------------------------
    scored_before = [r.quality_before.overall for r in records if r.quality_before]
    scored_after = [r.quality_after.overall for r in records if r.quality_after]

    result.avg_quality_before = round(sum(scored_before) / len(scored_before), 1) if scored_before else 0.0
    result.avg_quality_after = round(sum(scored_after) / len(scored_after), 1) if scored_after else 0.0
    result.fields_filled = sum(len(r.filled_keys()) for r in records)
    result.fields_flagged = sum(len(r.needs_review()) for r in records)
    result.llm_calls = llm.stats.calls
    result.cache_hits = llm.stats.cache_hits
    result.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")

    logger.info(
        "Batch %s: %d/%d ok, quality %.1f -> %.1f (+%.1f), %d LLM calls, %d cache hits",
        result.run_id, result.succeeded, result.total,
        result.avg_quality_before, result.avg_quality_after, result.uplift,
        result.llm_calls, result.cache_hits,
    )
    return records, result


def run_single(
    record: ProductRecord,
    *,
    options: PipelineOptions | None = None,
    schema: AttributeSchema | None = None,
    catalog: Iterable[ProductRecord] | None = None,
) -> ProductRecord:
    """Process one record, optionally against an existing catalogue for context."""
    options = options or PipelineOptions()
    schema = schema or load_schema()
    ctx = PipelineContext(schema=schema, llm=LLMClient(preferred=options.provider), options=options)

    if catalog and options.use_peers:
        peers = list(catalog)
        ctx.retriever = CatalogRetriever(peers)
        ctx.graph = ProductGraph(peers)

    score_baseline(record, schema)
    process_record(record, ctx)
    finalise_record(record, ctx)
    return record


def apply_human_edit(
    record: ProductRecord,
    key: str,
    new_value: Any,
    *,
    reviewer: str = "reviewer",
    note: str = "",
    schema: AttributeSchema | None = None,
) -> ProductRecord:
    """Record a reviewer decision. A human edit becomes the highest-precedence source."""
    from src.models import AttributeValue, Method, SourceType
    from src.normalize import canonical

    schema = schema or load_schema()
    spec = schema.spec(key, record.category)
    existing = record.get(key)

    av = AttributeValue.make(
        key=key,
        value=new_value,
        source_type=SourceType.HUMAN,
        method=Method.HUMAN_EDIT,
        source_ref=f"review:{reviewer}",
        evidence=note or "Corrected during human review",
        confidence=1.0,
    )
    if existing is not None:
        # Keep what the machine proposed; that comparison is the training signal.
        av.alternatives = [
            {
                "value": existing.value,
                "unit": existing.unit,
                "confidence": round(existing.penalised_confidence(), 3),
                "source_type": existing.provenance.source_type.value,
                "source_ref": existing.provenance.source_ref,
                "evidence": existing.provenance.evidence,
                "method": existing.provenance.method.value,
                "superseded_by_human": True,
            }
        ] + existing.alternatives

    if spec is not None:
        canonical.apply_to(av, spec)
    av.review_state = ReviewState.HUMAN_APPROVED
    av.issues = [i for i in av.issues if i.severity != Severity.ERROR]

    record.set_attribute(av)

    # Re-run the deterministic rules: filling a required field must clear the
    # error that said it was missing, and a corrected number can just as easily
    # break a cross-field check that used to pass.
    fresh_issues, rule_summary = rules.run_rules(record, schema)
    _replace_rule_issues(record, fresh_issues)

    record.log_stage("human_edit", field=key, value=new_value, reviewer=reviewer, note=note)
    record.quality_after = quality.score_record(record, schema, rule_summary=rule_summary)

    record_errors = any(i.severity == Severity.ERROR for i in record.record_issues)
    if not record.needs_review() and not record_errors:
        record.review_state = ReviewState.HUMAN_APPROVED
    return record


def approve_field(record: ProductRecord, key: str, *, reviewer: str = "reviewer") -> ProductRecord:
    """Accept a machine-proposed value as-is."""
    av = record.get(key)
    if av is None:
        return record
    av.review_state = ReviewState.HUMAN_APPROVED
    av.confidence = max(av.confidence, 0.95)
    av.issues = [i for i in av.issues if i.severity == Severity.INFO]
    record.log_stage("human_approve", field=key, reviewer=reviewer)

    if not record.needs_review():
        record.review_state = ReviewState.HUMAN_APPROVED
    return record


def reject_field(record: ProductRecord, key: str, *, reviewer: str = "reviewer", reason: str = "") -> ProductRecord:
    """Drop a machine-proposed value the reviewer does not accept."""
    av = record.get(key)
    if av is None:
        return record
    av.review_state = ReviewState.HUMAN_REJECTED
    av.alternatives = [
        {
            "value": av.value,
            "unit": av.unit,
            "confidence": round(av.penalised_confidence(), 3),
            "source_type": av.provenance.source_type.value,
            "rejected_by_human": True,
            "reason": reason,
        }
    ] + av.alternatives
    av.value = None
    av.confidence = 0.0
    record.log_stage("human_reject", field=key, reviewer=reviewer, reason=reason)
    return record
