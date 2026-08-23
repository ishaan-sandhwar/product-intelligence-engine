"""Browse the processed catalogue and inspect any product field by field."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.models import ProductRecord
from src.ui.common import (
    bootstrap,
    confidence_bar,
    fmt_value,
    get_schema,
    provenance_line,
    require_records,
    sidebar_status,
    state_label,
)

bootstrap("Catalog", "📦")
sidebar_status()

st.title("📦 Catalog")
st.caption("Every value carries its source, its evidence snippet and its confidence. "
           "Nothing in here is unexplainable.")

schema = get_schema()
records = require_records()
by_sku: dict[str, ProductRecord] = {r.sku: r for r in records}


# ------------------------------------------------------------------ filters
row = st.columns([3, 2, 2, 2])
search = row[0].text_input("Search", placeholder="SKU, name or brand")
categories = sorted({r.category or "unclassified" for r in records})
category = row[1].selectbox("Category", ["all", *categories])
states = sorted({r.review_state.value for r in records})
state = row[2].selectbox("Review state", ["all", *states])
grades = sorted({r.quality_after.grade() for r in records if r.quality_after})
grade = row[3].selectbox("Grade", ["all", *grades])


def keep(record: ProductRecord) -> bool:
    """Apply the four filter widgets to one record."""
    if category != "all" and (record.category or "unclassified") != category:
        return False
    if state != "all" and record.review_state.value != state:
        return False
    if grade != "all" and (record.quality_after.grade() if record.quality_after else None) != grade:
        return False
    if search:
        needle = search.lower()
        haystack = " ".join(
            str(x) for x in (record.sku, record.value_of("product_name", ""),
                             record.value_of("brand", ""))
        ).lower()
        if needle not in haystack:
            return False
    return True


filtered = [r for r in records if keep(r)]
st.caption(f"**{len(filtered)}** of {len(records)} products")

if not filtered:
    st.warning("No products match these filters.")
    st.stop()

summary = pd.DataFrame(
    [
        {
            "SKU": r.sku,
            "Product": fmt_value(r.value_of("product_name")),
            "Brand": fmt_value(r.value_of("brand")),
            "Category": schema.label_for(r.category) if r.category else "—",
            "Grade": r.quality_after.grade() if r.quality_after else "—",
            "Score": round(r.quality_after.overall, 1) if r.quality_after else None,
            "Input score": round(r.quality_before.overall, 1) if r.quality_before else None,
            "Filled": len(r.filled_keys()),
            "Flagged": len(r.needs_review()),
            "State": r.review_state.value,
        }
        for r in filtered
    ]
)
st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100,
                                                 format="%.0f"),
        "Input score": st.column_config.NumberColumn("Input score", format="%.0f"),
    },
)

st.divider()

# ------------------------------------------------------------------- detail
selected_sku = st.selectbox("Inspect product", [r.sku for r in filtered])
record = by_sku[selected_sku]

head = st.columns(5)
head[0].metric("Quality", f"{record.quality_after.overall:.0f}" if record.quality_after else "—",
               delta=(f"{record.quality_after.overall - record.quality_before.overall:+.1f}"
                      if record.quality_after and record.quality_before else None))
head[1].metric("Grade", record.quality_after.grade() if record.quality_after else "—")
head[2].metric("Fields filled", len(record.filled_keys()))
head[3].metric("Needs review", len(record.needs_review()))
head[4].metric("Sources", len(record.sources))

st.markdown(
    f"**{fmt_value(record.value_of('product_name'))}** · "
    f"category `{record.category or 'unclassified'}` "
    f"(confidence {record.category_confidence:.0%}) · {state_label(record.review_state)}"
)

if record.quality_after:
    dims = record.quality_after
    st.progress(min(1.0, dims.overall / 100),
                text=f"completeness {dims.completeness:.0f} · accuracy {dims.accuracy:.0f} · "
                     f"consistency {dims.consistency:.0f} · richness {dims.richness:.0f}")

tabs = st.tabs(["Attributes", "Content", "Evidence", "Issues", "Sources", "Pipeline log", "JSON"])

# --- attributes ------------------------------------------------------------
with tabs[0]:
    specs = schema.specs_for(record.category)
    rows = []
    for key, spec in specs.items():
        av = record.attributes.get(key)
        rows.append(
            {
                "Field": spec.label,
                "Value": fmt_value(av.value, av.unit) if av else "—",
                "Confidence": confidence_bar(av.penalised_confidence()) if av else "",
                "Source": av.provenance.source_type.value if av else "",
                "Method": av.provenance.method.value if av else "",
                "State": av.review_state.value if av else "missing",
                "Required": "yes" if spec.required else "",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=420)

# --- generated content -----------------------------------------------------
with tabs[1]:
    if not record.content:
        st.caption("No marketing content generated for this product yet.")
    for key, av in record.content.items():
        spec = schema.content.get(key)
        st.markdown(f"##### {spec.label if spec else key}")
        if av.is_empty:
            st.caption("—")
        elif isinstance(av.value, list):
            for item in av.value:
                st.markdown(f"- {item}")
        else:
            st.write(av.value)
        st.caption(f"{confidence_bar(av.penalised_confidence())} · {provenance_line(av)}")
        st.divider()

# --- evidence --------------------------------------------------------------
with tabs[2]:
    st.caption("Where each value physically came from — the traceability trail.")
    filled = {k: v for k, v in record.all_fields().items() if not v.is_empty}
    if not filled:
        st.caption("Nothing extracted yet.")
    for key, av in filled.items():
        spec = schema.spec(key, record.category)
        label = spec.label if spec else key
        with st.expander(f"{label} — {fmt_value(av.value, av.unit)} "
                         f"({av.penalised_confidence():.0%})"):
            st.markdown(provenance_line(av))
            if av.provenance.locator:
                st.caption(f"locator: {av.provenance.locator}")
            if av.provenance.evidence:
                st.markdown("**Evidence**")
                st.info(av.provenance.evidence)
            if av.raw_value not in (None, av.value):
                st.caption(f"raw value before normalisation: {av.raw_value}")
            if av.alternatives:
                st.markdown("**Rejected alternatives**")
                st.dataframe(pd.DataFrame(av.alternatives), use_container_width=True,
                             hide_index=True)
            if av.issues:
                for issue in av.issues:
                    st.warning(f"`{issue.code}` {issue.message}")

# --- issues ----------------------------------------------------------------
with tabs[3]:
    issues = record.all_issues()
    if not issues:
        st.success("No validation issues on this record.")
    else:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Severity": i.severity.value,
                        "Code": i.code,
                        "Field": i.field_key or "—",
                        "Message": i.message,
                        "Raised by": i.raised_by,
                        "Suggestion": fmt_value(i.suggestion),
                    }
                    for i in issues
                ]
            ),
            use_container_width=True, hide_index=True,
        )

# --- sources ---------------------------------------------------------------
with tabs[4]:
    if not record.sources:
        st.caption("Only the catalogue row was available for this product.")
    for doc in record.sources:
        st.markdown(f"**{doc.name}** · `{doc.kind}` · "
                    f"{doc.page_count or 0} page(s) · {len(doc.tables)} table(s)")
        if doc.url:
            st.caption(doc.url)
        if doc.text:
            with st.expander("Extracted text"):
                st.text(doc.text[:4000])

# --- pipeline log ----------------------------------------------------------
with tabs[5]:
    st.caption("Stage-by-stage trace of what the engine did to this record.")
    if record.stage_log:
        st.dataframe(pd.DataFrame(record.stage_log), use_container_width=True,
                     hide_index=True)
    else:
        st.caption("No stages logged.")

# --- json ------------------------------------------------------------------
with tabs[6]:
    st.json(json.loads(json.dumps(record.to_audit_dict(), default=str)), expanded=False)
