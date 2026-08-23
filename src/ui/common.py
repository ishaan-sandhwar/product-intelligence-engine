"""Shared Streamlit helpers: page setup, cached loaders, and value formatting.

Every page starts with `bootstrap()`, which fixes the page config, guarantees the
database exists and puts the catalogue in one place. Records live in
`st.session_state` during a run and in SQLite between runs, so a reviewer can
close the browser and come back to the same queue.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

import streamlit as st

from config import REVIEW_CONFIDENCE_THRESHOLD
from src import store
from src.models import AttributeValue, ProductRecord, ReviewState, Severity
from src.schema import AttributeSchema, load_schema

RECORDS_KEY = "records"
RUN_KEY = "last_run"
REVIEWER_KEY = "reviewer"

GRADE_COLORS = {
    "A": "#16a34a",
    "B": "#65a30d",
    "C": "#d97706",
    "D": "#dc2626",
    "F": "#991b1b",
}

SEVERITY_ICONS = {
    Severity.ERROR: "🛑",
    Severity.WARNING: "⚠️",
    Severity.INFO: "ℹ️",
}

STATE_ICONS = {
    ReviewState.AUTO_APPROVED: "✅",
    ReviewState.NEEDS_REVIEW: "🔎",
    ReviewState.HUMAN_APPROVED: "👤✅",
    ReviewState.HUMAN_REJECTED: "👤🚫",
}


# --------------------------------------------------------------------- setup
def bootstrap(title: str, icon: str = "🏭") -> None:
    """Page config + one-time database init. Call first on every page."""
    st.set_page_config(page_title=f"{title} · Product Intelligence Engine",
                       page_icon=icon, layout="wide")
    if not st.session_state.get("_db_ready"):
        store.init_db()
        st.session_state["_db_ready"] = True
    st.session_state.setdefault(REVIEWER_KEY, "reviewer")


@st.cache_resource(show_spinner=False)
def get_schema() -> AttributeSchema:
    """The attribute dictionary, loaded once per server process."""
    return load_schema()


def sidebar_status() -> None:
    """Provider/catalogue status block, identical on every page."""
    from src.llm.provider import LLMClient  # imported late: keeps page load fast

    with st.sidebar:
        st.markdown("### Engine status")
        providers = LLMClient.available_providers()
        if providers:
            st.success(f"LLM ready · {', '.join(providers)}")
        else:
            st.warning("No API key found — pipeline runs in deterministic mode "
                       "(regex, units, rules, peers). Add a key to `.env` for "
                       "extraction and content generation.")

        st.caption(f"Catalogue in database: **{store.count_records()}** SKUs")
        st.text_input("Reviewer name", key=REVIEWER_KEY)


# ---------------------------------------------------------------- catalogue
def set_records(records: list[ProductRecord]) -> None:
    st.session_state[RECORDS_KEY] = records


def session_records() -> list[ProductRecord]:
    """Records from the current session, falling back to the stored catalogue."""
    records = st.session_state.get(RECORDS_KEY)
    if records:
        return records
    records = store.load_all_full()
    if records:
        st.session_state[RECORDS_KEY] = records
    return records


def replace_record(updated: ProductRecord) -> None:
    """Swap one record in the session list and persist it."""
    records = st.session_state.get(RECORDS_KEY) or []
    for i, existing in enumerate(records):
        if existing.sku == updated.sku:
            records[i] = updated
            break
    else:
        records.append(updated)
    st.session_state[RECORDS_KEY] = records
    store.save_record(updated)


def require_records() -> list[ProductRecord]:
    """Guard used by pages that cannot work on an empty catalogue."""
    records = session_records()
    if not records:
        st.info("No products yet. Head to **Ingest & Run** to load a catalogue "
                "and process it.")
        st.stop()
    return records


# ---------------------------------------------------------------- formatting
def fmt_value(value: Any, unit: str | None = None) -> str:
    """Render any attribute value for a table cell."""
    if value is None or value == "":
        return "—"
    if isinstance(value, bool):
        text = "Yes" if value else "No"
    elif isinstance(value, list):
        text = ", ".join(str(v) for v in value)
    elif isinstance(value, dict):
        text = json.dumps(value, ensure_ascii=False)
    elif isinstance(value, float):
        text = f"{value:g}"
    else:
        text = str(value)
    return f"{text} {unit}".strip() if unit else text


def confidence_bar(confidence: float) -> str:
    """Five-block confidence meter — readable at a glance in a dense table."""
    filled = max(0, min(5, round(confidence * 5)))
    return "█" * filled + "░" * (5 - filled) + f" {confidence:.0%}"


def grade_badge(grade: str | None) -> str:
    if not grade:
        return "—"
    color = GRADE_COLORS.get(grade, "#6b7280")
    return f":primary-background[{grade}]" if color else grade


def state_label(state: ReviewState | str) -> str:
    state = ReviewState(state) if isinstance(state, str) else state
    return f"{STATE_ICONS.get(state, '')} {state.value.replace('_', ' ')}".strip()


def provenance_line(av: AttributeValue) -> str:
    """One-line 'where did this come from' string for an attribute value."""
    prov = av.provenance
    bits = [f"**{prov.source_type.value}**", f"via `{prov.method.value}`"]
    if prov.source_ref:
        bits.append(f"from `{prov.source_ref}`")
    if prov.model_id:
        bits.append(f"model `{prov.model_id}`")
    return " · ".join(bits)


def field_rows(record: ProductRecord, schema: AttributeSchema,
               *, only_filled: bool = False) -> list[dict[str, Any]]:
    """Flatten a record into table rows: value, unit, confidence, source, state."""
    specs = schema.specs_for(record.category)
    rows: list[dict[str, Any]] = []
    for key, av in record.all_fields().items():
        if only_filled and av.is_empty:
            continue
        spec = specs.get(key) or schema.content.get(key)
        rows.append({
            "Field": spec.label if spec else key.replace("_", " ").title(),
            "Key": key,
            "Value": fmt_value(av.value, av.unit),
            "Confidence": round(av.penalised_confidence(), 2),
            "Source": av.provenance.source_type.value,
            "Method": av.provenance.method.value,
            "State": av.review_state.value,
            "Issues": len(av.issues),
        })
    return rows


def flagged_fields(records: Iterable[ProductRecord]) -> list[tuple[ProductRecord, AttributeValue]]:
    """Every (record, field) pair waiting on a human, worst confidence first."""
    pairs = [(r, av) for r in records for av in r.needs_review()]
    pairs.sort(key=lambda pair: (
        not pair[1].has_error,                 # errors first
        pair[1].penalised_confidence(),        # then least confident
    ))
    return pairs


def missing_required(records: Iterable[ProductRecord]) -> list[tuple[ProductRecord, str]]:
    """Required fields nobody could fill — record-level gaps, not bad values.

    These never reach the field queue (there is no value to judge), so without
    this they would sit in the catalogue unfixable by a human.
    """
    out: list[tuple[ProductRecord, str]] = []
    for record in records:
        for issue in record.record_issues:
            if issue.code == "MISSING_REQUIRED" and issue.field_key:
                out.append((record, issue.field_key))
    return out


def review_threshold() -> float:
    return float(st.session_state.get("review_threshold", REVIEW_CONFIDENCE_THRESHOLD))
