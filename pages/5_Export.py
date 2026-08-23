"""Export the golden records: flat commerce CSV, or the full traceable audit JSON."""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import streamlit as st

from config import DELIVERY_ATTRIBUTE_SLOTS, EXPORT_DIR, MIN_PUBLISH_CONFIDENCE
from src.export.delivery import coverage_report, to_delivery_frame
from src.models import ProductRecord, ReviewState
from src.ui.common import bootstrap, get_schema, require_records, sidebar_status

bootstrap("Export", "⬇️")
sidebar_status()

st.title("⬇️ Export")
st.caption("Two shapes for two audiences: a flat row per SKU for the PIM or "
           "storefront, and the full evidence trail for whoever has to defend a value.")

schema = get_schema()
records = require_records()

# ------------------------------------------------------------------ filters
row = st.columns(3)
scope = row[0].selectbox(
    "Which products",
    ["Publish-ready only", "Everything", "Needs review only"],
    help="Publish-ready = nothing waiting on a human and nothing rejected.",
)
min_conf = row[1].slider("Drop fields below confidence", 0.0, 1.0,
                         MIN_PUBLISH_CONFIDENCE, 0.05)
include_content = row[2].toggle("Include generated content", value=True)


def in_scope(record: ProductRecord) -> bool:
    if scope == "Everything":
        return True
    needs_review = (record.review_state == ReviewState.NEEDS_REVIEW
                    or bool(record.needs_review()))
    return needs_review if scope == "Needs review only" else not needs_review


selected = [r for r in records if in_scope(r)]
st.caption(f"**{len(selected)}** of {len(records)} products in scope")

if not selected:
    st.warning("Nothing matches this scope.")
    st.stop()


# -------------------------------------------------------------------- flat
def flat_row(record: ProductRecord) -> dict[str, object]:
    """One commerce-ready row, honouring the confidence floor and content toggle."""
    row_out: dict[str, object] = {
        "sku": record.sku,
        "category": record.category,
        "category_label": schema.label_for(record.category) if record.category else None,
    }
    fields = dict(record.attributes)
    if include_content:
        fields.update(record.content)

    for key, av in fields.items():
        if av.is_empty or av.penalised_confidence() < min_conf:
            continue
        value = ", ".join(str(v) for v in av.value) if isinstance(av.value, list) else av.value
        row_out[key] = value
        if av.unit:
            row_out[f"{key}__unit"] = av.unit

    row_out["quality_score"] = record.quality_after.overall if record.quality_after else None
    row_out["quality_grade"] = record.quality_after.grade() if record.quality_after else None
    row_out["review_state"] = record.review_state.value
    row_out["fields_flagged"] = len(record.needs_review())
    return row_out


stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

flat_df = pd.DataFrame([flat_row(r) for r in selected])

# Stable column order: identity, common attributes, category attributes, content, meta.
ordered = ["sku", "category", "category_label"]
ordered += [k for k in schema.common if k in flat_df.columns and k not in ordered]
ordered += [c for c in flat_df.columns
            if c not in ordered and not c.endswith("__unit")
            and c not in schema.content
            and c not in {"quality_score", "quality_grade", "review_state", "fields_flagged"}]
if include_content:
    ordered += [k for k in schema.content if k in flat_df.columns]
ordered += [c for c in flat_df.columns if c.endswith("__unit")]
ordered += ["quality_score", "quality_grade", "review_state", "fields_flagged"]
flat_df = flat_df[[c for c in dict.fromkeys(ordered) if c in flat_df.columns]]

tab_delivery, tab_flat, tab_audit, tab_queue = st.tabs(
    ["Delivery template (CSV)", "Flat catalogue (CSV)", "Audit trail (JSON)",
     "Open review queue (CSV)"]
)

with tab_delivery:
    st.caption("The Unilog delivery contract: taxonomy as Dept / Class / Fine, "
               "specifications packed into LABEL / VALUE / UOM triplets, and the "
               "six description variants. Column order is read from the template "
               "file itself, so a revised template needs no code change.")

    delivery_df = to_delivery_frame(selected, schema, min_confidence=min_conf)
    coverage = coverage_report(delivery_df)

    cols = st.columns(5)
    cols[0].metric("Rows", coverage["rows"])
    cols[1].metric("Template columns", coverage["columns"])
    cols[2].metric("Taxonomy filled", f"{coverage['taxonomy_pct']:.0f}%")
    cols[3].metric("Content filled", f"{coverage['content_pct']:.0f}%")
    cols[4].metric("Attribute slots used",
                   f"{coverage['attribute_slots_used_avg']:.1f}",
                   help=f"out of {DELIVERY_ATTRIBUTE_SLOTS} available per product; "
                        f"most used on one product: {coverage['attribute_slots_used_max']}")

    preview_cols = [c for c in delivery_df.columns
                    if delivery_df[c].astype(str).str.strip().ne("").any()]
    st.dataframe(delivery_df[preview_cols].head(30), use_container_width=True,
                 hide_index=True)
    st.caption(f"Preview hides the {len(delivery_df.columns) - len(preview_cols)} "
               "template columns nothing filled — the download keeps all "
               f"{len(delivery_df.columns)}, in template order.")

    delivery_bytes = delivery_df.to_csv(index=False).encode("utf-8-sig")
    col1, col2 = st.columns(2)
    col1.download_button("Download delivery CSV", delivery_bytes,
                         f"delivery_{stamp}.csv", "text/csv", type="primary",
                         use_container_width=True)
    if col2.button("Save to outputs/exports", key="save_delivery",
                   use_container_width=True):
        target = EXPORT_DIR / f"delivery_{stamp}.csv"
        target.write_bytes(delivery_bytes)
        st.success(f"Written to {target}")

with tab_flat:
    st.dataframe(flat_df.head(50), use_container_width=True, hide_index=True)
    st.caption(f"{len(flat_df)} rows × {len(flat_df.columns)} columns. Values below "
               f"{min_conf:.0%} confidence are held back rather than published.")

    csv_bytes = flat_df.to_csv(index=False).encode("utf-8")
    col1, col2 = st.columns(2)
    col1.download_button("Download CSV", csv_bytes, f"catalog_{stamp}.csv",
                         "text/csv", type="primary", use_container_width=True)
    if col2.button("Save to outputs/exports", use_container_width=True):
        target = EXPORT_DIR / f"catalog_{stamp}.csv"
        target.write_bytes(csv_bytes)
        st.success(f"Written to {target}")

with tab_audit:
    audit = [r.to_audit_dict() for r in selected]
    st.caption("Every field with its source, evidence snippet, method, confidence, "
               "issues and rejected alternatives — the record of why each value is "
               "what it is.")
    st.json(json.loads(json.dumps(audit[0], default=str)), expanded=False)

    audit_bytes = json.dumps(audit, indent=2, ensure_ascii=False, default=str).encode("utf-8")
    col1, col2 = st.columns(2)
    col1.download_button("Download JSON", audit_bytes, f"audit_{stamp}.json",
                         "application/json", type="primary", use_container_width=True)
    if col2.button("Save to outputs/exports", key="save_audit", use_container_width=True):
        target = EXPORT_DIR / f"audit_{stamp}.json"
        target.write_bytes(audit_bytes)
        st.success(f"Written to {target}")

with tab_queue:
    pending = [
        {
            "sku": record.sku,
            "product": record.value_of("product_name"),
            "field": av.key,
            "machine_value": av.value,
            "unit": av.unit,
            "confidence": av.penalised_confidence(),
            "source": av.provenance.source_type.value,
            "method": av.provenance.method.value,
            "evidence": av.provenance.evidence,
            "issues": "; ".join(f"{i.code}: {i.message}" for i in av.issues),
        }
        for record in selected
        for av in record.needs_review()
    ]
    if not pending:
        st.success("Nothing pending in this scope.")
    else:
        queue_df = pd.DataFrame(pending)
        st.dataframe(queue_df, use_container_width=True, hide_index=True)
        st.download_button("Download queue CSV",
                           queue_df.to_csv(index=False).encode("utf-8"),
                           f"review_queue_{stamp}.csv", "text/csv",
                           use_container_width=True)

st.divider()
st.info(
    "**Mapping to a required output template:** the flat export is generated from "
    "the canonical keys in `schemas/attributes.json`. To emit a different column "
    "contract, add the target names as `aliases` on each attribute and rename at "
    "this layer — the pipeline itself does not change."
)
