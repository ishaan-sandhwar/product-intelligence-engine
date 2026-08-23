"""Human-in-the-loop review: approve, correct, reject, or fill what is missing.

Every decision is written to `review_decisions`, which is what the Quality page
uses to show whether the confidence thresholds are actually calibrated — the loop
is closed, not decorative.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src import store
from src.models import AttributeValue, ProductRecord
from src.pipeline import apply_human_edit, approve_field, reject_field
from src.schema import AttributeSpec
from src.ui.common import (
    REVIEWER_KEY,
    bootstrap,
    confidence_bar,
    flagged_fields,
    fmt_value,
    get_schema,
    missing_required,
    provenance_line,
    replace_record,
    require_records,
    sidebar_status,
)

bootstrap("Review Queue", "🔎")
sidebar_status()

st.title("🔎 Review Queue")
st.caption("Only what the engine could not settle on its own reaches this page — "
           "low confidence, an open validation error, a conflict between sources, "
           "or a required field nothing could fill.")

schema = get_schema()
records = require_records()
reviewer = st.session_state.get(REVIEWER_KEY, "reviewer")

queue = flagged_fields(records)
gaps = missing_required(records)

if not queue and not gaps:
    st.success("Queue is empty — every field is either auto-approved or already reviewed.")
    st.stop()


# ------------------------------------------------------------ shared pieces
def edit_widget(spec: AttributeSpec | None, current: Any, widget_key: str,
                label: str = "Corrected value") -> Any:
    """Render the input control that matches the attribute's declared datatype."""
    datatype = spec.datatype if spec else "string"

    if datatype == "number":
        value = float(current) if isinstance(current, (int, float)) else 0.0
        suffix = f" ({spec.unit})" if spec and spec.unit else ""
        return st.number_input(label + suffix, value=value, key=widget_key, format="%g")
    if datatype == "boolean":
        return st.checkbox(label, value=bool(current), key=widget_key)
    if datatype == "enum" and spec and spec.enum:
        options = ["", *spec.enum]
        index = options.index(current) if current in options else 0
        return st.selectbox(label, options, index=index, key=widget_key)
    if datatype == "list":
        joined = ", ".join(str(v) for v in current) if isinstance(current, list) else ""
        raw = st.text_input(label, value=joined, key=widget_key, help="Comma-separated")
        return [part.strip() for part in raw.split(",") if part.strip()]
    return st.text_input(label, value="" if current is None else str(current), key=widget_key)


def log_decision(record: ProductRecord, field_key: str, decision: str, *,
                 machine: AttributeValue | None = None, human_value: Any = None,
                 note: str = "") -> None:
    """One row in review_decisions per human action."""
    store.record_decision(
        sku=record.sku,
        field_key=field_key,
        decision=decision,
        machine_value=machine.value if machine else None,
        human_value=human_value,
        machine_conf=machine.penalised_confidence() if machine else None,
        source_type=machine.provenance.source_type.value if machine else None,
        method=machine.provenance.method.value if machine else None,
        category=record.category,
        reviewer=reviewer,
        note=note,
    )


tab_values, tab_gaps = st.tabs(
    [f"Flagged values ({len(queue)})", f"Missing required ({len(gaps)})"]
)

# ---------------------------------------------------------- flagged values
with tab_values:
    if not queue:
        st.success("No flagged values.")
    else:
        row = st.columns([2, 2, 2, 2])
        categories = sorted({r.category or "unclassified" for r, _ in queue})
        category = row[0].selectbox("Category", ["all", *categories])
        field_keys = sorted({av.key for _, av in queue})
        field_key = row[1].selectbox("Field", ["all", *field_keys])
        only_errors = row[2].toggle("Errors only", value=False)
        page_size = row[3].selectbox("Show", [10, 25, 50], index=0)

        def keep(record: ProductRecord, av: AttributeValue) -> bool:
            if category != "all" and (record.category or "unclassified") != category:
                return False
            if field_key != "all" and av.key != field_key:
                return False
            return not (only_errors and not av.has_error)

        visible = [(r, av) for r, av in queue if keep(r, av)]

        head = st.columns(4)
        head[0].metric("In queue", len(queue))
        head[1].metric("Matching filters", len(visible))
        head[2].metric("With errors", sum(1 for _, av in queue if av.has_error))
        head[3].metric("Products affected", len({r.sku for r, _ in queue}))

        with st.expander("Bulk approve"):
            st.caption("Bulk approval still records one decision per field, so the "
                       "agreement statistics stay honest.")
            floor = st.slider("Approve everything at or above this confidence",
                              0.0, 1.0, 0.7, 0.05)
            candidates = [(r, av) for r, av in visible
                          if av.penalised_confidence() >= floor and not av.has_error]
            st.caption(f"{len(candidates)} field(s) qualify — fields carrying an "
                       "error are never bulk-approved.")
            if candidates and st.button(f"Approve {len(candidates)} field(s)",
                                        type="primary"):
                for record, av in candidates:
                    log_decision(record, av.key, "approved", machine=av,
                                 note="bulk approve")
                    replace_record(approve_field(record, av.key, reviewer=reviewer))
                st.success(f"Approved {len(candidates)} field(s).")
                st.rerun()

        for index, (record, av) in enumerate(visible[:page_size]):
            spec = schema.spec(av.key, record.category)
            label = spec.label if spec else av.key
            icon = "🛑" if av.has_error else "🔎"

            with st.container(border=True):
                st.markdown(f"{icon} **{label}** · `{record.sku}` · "
                            f"{fmt_value(record.value_of('product_name'))}")
                left, right = st.columns([3, 2])

                with left:
                    st.markdown(f"### {fmt_value(av.value, av.unit)}")
                    st.caption(f"{confidence_bar(av.penalised_confidence())} · "
                               f"{provenance_line(av)}")
                    if av.provenance.evidence:
                        st.info(av.provenance.evidence)
                    for issue in av.issues:
                        text = f"`{issue.code}` {issue.message}"
                        if issue.suggestion is not None:
                            text += f" — suggested: {fmt_value(issue.suggestion)}"
                        (st.error if issue.severity.value == "error" else st.warning)(text)
                    if av.alternatives:
                        with st.expander(f"{len(av.alternatives)} rejected alternative(s)"):
                            st.dataframe(pd.DataFrame(av.alternatives),
                                         use_container_width=True, hide_index=True)

                with right:
                    widget_key = f"edit_{record.sku}_{av.key}_{index}"
                    new_value = edit_widget(spec, av.value, widget_key)
                    note = st.text_input("Note (optional)", key=f"note_{widget_key}")

                    actions = st.columns(3)
                    if actions[0].button("Approve", key=f"ok_{widget_key}",
                                         use_container_width=True):
                        log_decision(record, av.key, "approved", machine=av, note=note)
                        replace_record(approve_field(record, av.key, reviewer=reviewer))
                        st.rerun()

                    if actions[1].button("Save edit", key=f"save_{widget_key}",
                                         type="primary", use_container_width=True):
                        log_decision(record, av.key, "edited", machine=av,
                                     human_value=new_value, note=note)
                        replace_record(apply_human_edit(record, av.key, new_value,
                                                        reviewer=reviewer, note=note,
                                                        schema=schema))
                        st.rerun()

                    if actions[2].button("Reject", key=f"no_{widget_key}",
                                         use_container_width=True):
                        log_decision(record, av.key, "rejected", machine=av, note=note)
                        replace_record(reject_field(record, av.key, reviewer=reviewer,
                                                    reason=note))
                        st.rerun()

        if len(visible) > page_size:
            st.caption(f"Showing {page_size} of {len(visible)} — work the queue down "
                       "or narrow the filters to see the rest.")

# --------------------------------------------------------- missing required
with tab_gaps:
    if not gaps:
        st.success("Every required field is filled.")
    else:
        st.caption("Required by the schema, and no source contained it. A value "
                   "typed here becomes the highest-precedence source on the record "
                   "and immediately clears the error.")

        gap_skus = sorted({record.sku for record, _ in gaps})
        chosen_sku = st.selectbox("Product", gap_skus)
        record = next(r for r, _ in gaps if r.sku == chosen_sku)
        keys = [key for r, key in gaps if r.sku == chosen_sku]

        st.markdown(f"**{fmt_value(record.value_of('product_name'))}** · "
                    f"`{record.category or 'unclassified'}` · "
                    f"{len(keys)} required field(s) missing")

        for key in keys:
            spec = schema.spec(key, record.category)
            widget_key = f"gap_{record.sku}_{key}"
            cols = st.columns([3, 2, 1])
            with cols[0]:
                value = edit_widget(spec, None, widget_key,
                                    label=spec.label if spec else key)
            with cols[1]:
                note = st.text_input("Source / note", key=f"note_{widget_key}",
                                     placeholder="where this came from")
            with cols[2]:
                st.write("")
                st.write("")
                if st.button("Save", key=f"save_{widget_key}", use_container_width=True):
                    if value in ("", None, []):
                        st.warning("Enter a value first.")
                    else:
                        log_decision(record, key, "edited", human_value=value,
                                     note=note or "filled a missing required field")
                        replace_record(apply_human_edit(record, key, value,
                                                        reviewer=reviewer, note=note,
                                                        schema=schema))
                        st.rerun()

        st.divider()
        gap_counts = pd.Series([key for _, key in gaps]).value_counts()
        st.subheader("Where the gaps are, catalogue-wide")
        st.bar_chart(gap_counts)
        st.caption("A field missing across most of the catalogue is usually a "
                   "sourcing problem, not a review problem — attach a datasheet "
                   "or a supplier URL and re-run instead of typing it product "
                   "by product.")
