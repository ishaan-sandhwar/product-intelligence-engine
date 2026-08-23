"""Product Intelligence Engine — Streamlit entry point.

Run with:  streamlit run Overview.py

The home page answers the only question that matters at a glance: how good is
this catalogue now, how good was it when it arrived, and what still needs a
human.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src import store
from src.score import quality
from src.ui.common import GRADE_COLORS, bootstrap, session_records, sidebar_status

bootstrap("Overview")
sidebar_status()

st.title("🏭 Product Intelligence Engine")
st.caption(
    "Turns a part number and a line of distributor shorthand into a complete, "
    "validated, delivery-ready product record — with the evidence for every "
    "value kept attached."
)

records = session_records()

if not records:
    st.info("No catalogue loaded yet.")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
            #### Start here
            1. **Ingest & Run** — upload a catalogue file (CSV/XLSX), optionally
               attach datasheets, images or product URLs, then run the pipeline.
            2. **Catalog** — inspect any product field by field, with the source
               snippet behind every value.
            3. **Review Queue** — approve, edit or reject what the engine was not
               sure about.
            4. **Quality** — see where the score comes from and which fields the
               reviewers keep correcting.
            5. **Export** — flat commerce CSV, or the full audit JSON.
            """
        )
    with col_b:
        st.markdown(
            """
            #### The dataset
            The challenge catalogue lives in `data/raw/` — 1,000 rows of part
            number, a 35-character description and three brand columns that are
            usually placeholders. Pick it on the Ingest page.

            No API key? The deterministic path still runs: trade-shorthand
            decoding, unit conversion, rules and catalogue peers. Content
            generation is the part that needs a model.
            """
        )
    st.stop()

stats = quality.score_catalog(records)

# ------------------------------------------------------------------ headline
cols = st.columns(5)
cols[0].metric("Products", stats["count"])
cols[1].metric("Quality score", f"{stats['avg_after']:.0f}",
               delta=f"{stats['uplift']:+.1f} vs input")
cols[2].metric("Input score", f"{stats['avg_before']:.0f}")
cols[3].metric("Auto-approved fields", f"{stats['auto_approved_pct']:.0f}%")
cols[4].metric("Awaiting review", stats["fields_flagged"])

st.divider()

# ------------------------------------------------------------------- charts
left, right = st.columns([3, 2])

with left:
    st.subheader("Quality dimensions")
    dims = pd.DataFrame(
        {
            "Dimension": ["Completeness", "Accuracy", "Consistency", "Richness"],
            "Score": [
                stats["avg_completeness"],
                stats["avg_accuracy"],
                stats["avg_consistency"],
                stats["avg_richness"],
            ],
        }
    )
    fig = px.bar(dims, x="Score", y="Dimension", orientation="h",
                 range_x=[0, 100], text="Score")
    fig.update_traces(marker_color="#2563eb", texttemplate="%{text:.0f}")
    fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0),
                      yaxis_title=None, xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Grades")
    grades = stats.get("grades") or {}
    if grades:
        grade_df = pd.DataFrame(
            {"Grade": list(grades), "Products": list(grades.values())}
        )
        fig = px.bar(grade_df, x="Grade", y="Products", text="Products",
                     color="Grade", color_discrete_map=GRADE_COLORS)
        fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0),
                          showlegend=False, xaxis_title=None, yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Not scored yet.")

left, right = st.columns(2)

with left:
    st.subheader("Categories")
    cats = stats.get("categories") or {}
    if cats:
        cat_df = pd.DataFrame({"Category": list(cats), "Products": list(cats.values())})
        fig = px.pie(cat_df, names="Category", values="Products", hole=0.45)
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Most common issues")
    issues = stats.get("top_issues") or {}
    if issues:
        issue_df = pd.DataFrame(
            {"Issue": list(issues), "Count": list(issues.values())}
        ).sort_values("Count")
        fig = px.bar(issue_df, x="Count", y="Issue", orientation="h", text="Count")
        fig.update_traces(marker_color="#dc2626")
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                          yaxis_title=None, xaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No validation issues outstanding.")

# --------------------------------------------------------------------- runs
st.divider()
st.subheader("Run history")
runs = store.list_runs(limit=10)
if runs:
    run_df = pd.DataFrame(runs)[
        ["run_id", "started_at", "total", "succeeded", "failed",
         "avg_quality_before", "avg_quality_after", "llm_calls", "cache_hits"]
    ].rename(
        columns={
            "run_id": "Run", "started_at": "Started", "total": "Products",
            "succeeded": "OK", "failed": "Failed", "avg_quality_before": "Before",
            "avg_quality_after": "After", "llm_calls": "LLM calls",
            "cache_hits": "Cache hits",
        }
    )
    st.dataframe(run_df, use_container_width=True, hide_index=True)
else:
    st.caption("No runs recorded yet.")
