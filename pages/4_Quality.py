"""Quality analytics: where the score comes from, and whether the engine's
confidence actually matches what reviewers decide."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src import store
from src.score import quality
from src.ui.common import bootstrap, get_schema, require_records, sidebar_status

bootstrap("Quality", "📊")
sidebar_status()

st.title("📊 Quality & Calibration")
st.caption("A quality score nobody can decompose is a vanity metric. Every number "
           "here traces back to fields, rules and reviewer decisions.")

schema = get_schema()
records = require_records()
stats = quality.score_catalog(records)

tab_score, tab_fields, tab_issues, tab_calibration = st.tabs(
    ["Score", "Field coverage", "Issues", "Reviewer calibration"]
)

# ------------------------------------------------------------------- score
with tab_score:
    cols = st.columns(4)
    cols[0].metric("Average score", f"{stats['avg_after']:.1f}",
                   delta=f"{stats['uplift']:+.1f}")
    cols[1].metric("Completeness", f"{stats['avg_completeness']:.0f}")
    cols[2].metric("Accuracy", f"{stats['avg_accuracy']:.0f}")
    cols[3].metric("Consistency / Richness",
                   f"{stats['avg_consistency']:.0f} / {stats['avg_richness']:.0f}")

    uplift_df = pd.DataFrame(
        [
            {
                "SKU": r.sku,
                "Before": r.quality_before.overall if r.quality_before else 0.0,
                "After": r.quality_after.overall if r.quality_after else 0.0,
                "Category": r.category or "unclassified",
            }
            for r in records
        ]
    )
    uplift_df["Uplift"] = uplift_df["After"] - uplift_df["Before"]

    st.subheader("Per-product uplift")
    fig = px.scatter(
        uplift_df, x="Before", y="After", color="Category", hover_name="SKU",
        range_x=[0, 100], range_y=[0, 100],
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=100, y1=100,
                  line=dict(dash="dot", color="#9ca3af"))
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Points above the diagonal gained quality. Points on it were already "
               "complete, or the engine had nothing new to work with.")

    st.subheader("Weakest products")
    st.dataframe(
        uplift_df.sort_values("After").head(15),
        use_container_width=True, hide_index=True,
        column_config={
            "After": st.column_config.ProgressColumn("After", min_value=0,
                                                     max_value=100, format="%.0f"),
        },
    )

# ----------------------------------------------------------- field coverage
with tab_fields:
    st.caption("Fill rate per canonical attribute, and the average confidence of "
               "the values that did get filled.")

    coverage: dict[str, dict[str, float]] = {}
    for record in records:
        for key, spec in schema.specs_for(record.category).items():
            entry = coverage.setdefault(
                key, {"label": spec.label, "required": spec.required,
                      "applicable": 0, "filled": 0, "confidence_sum": 0.0}
            )
            entry["applicable"] += 1
            av = record.attributes.get(key)
            if av and not av.is_empty:
                entry["filled"] += 1
                entry["confidence_sum"] += av.penalised_confidence()

    coverage_df = pd.DataFrame(
        [
            {
                "Field": data["label"],
                "Key": key,
                "Required": "yes" if data["required"] else "",
                "Fill rate": round(100.0 * data["filled"] / data["applicable"], 1)
                if data["applicable"] else 0.0,
                "Filled": data["filled"],
                "Applicable": data["applicable"],
                "Avg confidence": round(data["confidence_sum"] / data["filled"], 2)
                if data["filled"] else 0.0,
            }
            for key, data in coverage.items()
        ]
    ).sort_values("Fill rate")

    worst = coverage_df.head(20)
    fig = px.bar(worst, x="Fill rate", y="Field", orientation="h", range_x=[0, 100],
                 color="Avg confidence", color_continuous_scale="RdYlGn",
                 range_color=[0, 1])
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0), yaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(coverage_df, use_container_width=True, hide_index=True,
                 column_config={
                     "Fill rate": st.column_config.ProgressColumn(
                         "Fill rate", min_value=0, max_value=100, format="%.0f%%"),
                 })

# ------------------------------------------------------------------ issues
with tab_issues:
    issues = [
        {
            "SKU": record.sku,
            "Severity": issue.severity.value,
            "Code": issue.code,
            "Field": issue.field_key or "—",
            "Message": issue.message,
            "Raised by": issue.raised_by,
        }
        for record in records
        for issue in record.all_issues()
    ]
    if not issues:
        st.success("No open validation issues across the catalogue.")
    else:
        issue_df = pd.DataFrame(issues)
        counts = issue_df.groupby(["Code", "Severity"]).size().reset_index(name="Count")
        fig = px.bar(counts.sort_values("Count"), x="Count", y="Code",
                     orientation="h", color="Severity",
                     color_discrete_map={"error": "#dc2626", "warning": "#d97706",
                                         "info": "#2563eb"})
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0), yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)

        severity = st.multiselect("Severity", ["error", "warning", "info"],
                                  default=["error", "warning"])
        st.dataframe(issue_df[issue_df["Severity"].isin(severity)],
                     use_container_width=True, hide_index=True)

# ------------------------------------------------------------- calibration
with tab_calibration:
    decisions = store.decision_stats()
    reviewed = decisions["total_reviewed"]

    if not reviewed:
        st.info("No human decisions recorded yet. Work through the **Review Queue** "
                "and this page starts showing whether the confidence thresholds "
                "are set correctly.")
    else:
        cols = st.columns(3)
        cols[0].metric("Fields reviewed", reviewed)
        cols[1].metric("Agreement rate", f"{decisions['agreement_rate']:.0f}%",
                       help="Share of reviewed fields a human approved unchanged.")
        by_decision = decisions["by_decision"]
        cols[2].metric("Edited / rejected",
                       by_decision.get("edited", 0) + by_decision.get("rejected", 0))

        left, right = st.columns(2)
        with left:
            st.subheader("Decisions")
            dec_df = pd.DataFrame(
                {"Decision": list(by_decision), "Count": list(by_decision.values())}
            )
            fig = px.pie(dec_df, names="Decision", values="Count", hole=0.45,
                         color="Decision",
                         color_discrete_map={"approved": "#16a34a", "edited": "#d97706",
                                             "rejected": "#dc2626"})
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.subheader("Approval rate by extraction method")
            by_method = decisions["by_method"]
            if by_method:
                method_df = pd.DataFrame(by_method)
                method_df["Approval %"] = (
                    100.0 * method_df["approved"] / method_df["reviewed"]
                ).round(1)
                fig = px.bar(method_df.sort_values("Approval %"), x="Approval %",
                             y="method", orientation="h", range_x=[0, 100],
                             text="reviewed")
                fig.update_traces(marker_color="#2563eb",
                                  texttemplate="n=%{text}")
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                                  yaxis_title=None)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("A method reviewers keep correcting is a method whose base "
                           "confidence in `config.METHOD_BASE_CONFIDENCE` is too high.")

        st.subheader("Fields humans correct most")
        worst = decisions["most_corrected_fields"]
        if worst:
            worst_df = pd.DataFrame(worst)
            worst_df["Correction rate %"] = (
                100.0 * worst_df["corrected"] / worst_df["reviewed"]
            ).round(1)
            st.dataframe(worst_df, use_container_width=True, hide_index=True)
        else:
            st.caption("Not enough decisions yet to rank fields (needs 2+ per field).")
