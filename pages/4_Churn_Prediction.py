from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from modules.data_manager import get_active_dataset, render_data_source_banner
from utils.groq_client import ask_groq
from utils.helpers import (
    apply_filters,
    compute_churn,
    compute_clustering,
    compute_rfm,
    empty_state,
    render_kpi_row,
    sidebar_filters,
)
from utils.styling import apply_theme

PLOTLY_TEMPLATE = "plotly_dark"
AI_TOGGLE_LABEL = "Enable AI Features"


def configure_page() -> None:
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="CustomerLens | Churn Prediction",
        page_icon=":rotating_light:",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def prepare_data() -> Dict[str, object]:
    """Load data and train the churn model."""
    with st.spinner("Loading customer data..."):
        base_df = get_active_dataset()
    with st.spinner("Computing RFM segments..."):
        rfm_df = compute_rfm(base_df)
    with st.spinner("Running clustering analysis..."):
        cluster_result = compute_clustering(rfm_df)
    with st.spinner("Training churn model..."):
        churn_result = compute_churn(cluster_result["df"])

    return {"df": churn_result["df"].copy(), "churn_result": churn_result}


def build_kpis(filtered: pd.DataFrame, baseline: pd.DataFrame, roc_auc: float) -> List[Dict[str, object]]:
    """Build KPI metrics for churn analysis."""
    churn_rate = filtered["churn_flag"].mean()
    avg_prob = filtered["churn_probability"].mean()
    high_risk = (filtered["churn_probability"] >= 0.7).sum()

    return [
        {
            "label": "ROC-AUC",
            "value": roc_auc,
            "display": f"{roc_auc:.3f}",
            "baseline": roc_auc,
        },
        {
            "label": "Churn Rate",
            "value": float(churn_rate),
            "display": f"{churn_rate * 100:.1f}%",
            "baseline": float(baseline["churn_flag"].mean()),
        },
        {
            "label": "Avg Churn Probability",
            "value": float(avg_prob),
            "display": f"{avg_prob * 100:.1f}%",
            "baseline": float(baseline["churn_probability"].mean()),
        },
        {
            "label": "High Risk Customers",
            "value": float(high_risk),
            "display": f"{int(high_risk):,}",
            "baseline": float((baseline["churn_probability"] >= 0.7).sum()),
        },
    ]


def build_cluster_gauges(df: pd.DataFrame) -> go.Figure:
    """Build gauge charts for churn risk by cluster."""
    cluster_scores = (
        df.groupby("cluster_id")["churn_probability"].mean().reset_index().sort_values("cluster_id")
    )
    clusters = cluster_scores["cluster_id"].tolist()
    cols = 3
    rows = math.ceil(len(clusters) / cols)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=[[{"type": "indicator"}] * cols for _ in range(rows)],
    )

    for idx, (_, row) in enumerate(cluster_scores.iterrows()):
        r = idx // cols + 1
        c = idx % cols + 1
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=row["churn_probability"] * 100,
                title={"text": f"Cluster {int(row['cluster_id'])}"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#6C63FF"},
                    "steps": [
                        {"range": [0, 30], "color": "#1C1F26"},
                        {"range": [30, 60], "color": "#2C3340"},
                        {"range": [60, 100], "color": "#3A4455"},
                    ],
                },
            ),
            row=r,
            col=c,
        )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=rows * 250,
        transition_duration=500,
        margin=dict(t=40, b=20),
    )
    return fig


def predict_what_if(churn_result: Dict[str, object], overrides: Dict[str, float]) -> float:
    """Predict churn probability for a what-if scenario."""
    feature_cols = churn_result["feature_cols"]
    data = churn_result["df"]
    baseline = data[feature_cols].median(numeric_only=True).to_dict()
    baseline.update(overrides)

    input_df = pd.DataFrame([baseline])[feature_cols]
    scaler = churn_result["scaler"]
    model = churn_result["model"]

    proba = model.predict_proba(scaler.transform(input_df))[0, 1]
    return float(proba)


def render_page() -> None:
    """Render the churn prediction page."""
    configure_page()
    apply_theme()

    st.title("Churn Prediction")
    st.caption("Predict churn risk and simulate retention interventions.")
    render_data_source_banner()

    data_bundle = prepare_data()
    full_df = data_bundle["df"]
    churn_result = data_bundle["churn_result"]

    filters = sidebar_filters(full_df, cluster_options=sorted(full_df["cluster_id"].unique().tolist()))
    filtered = apply_filters(full_df, filters)

    if filtered.empty:
        empty_state("No customers match the current filters. Adjust the sidebar to continue.")
        return

    render_kpi_row(build_kpis(filtered, full_df, churn_result["roc_auc"]))

    st.markdown("### Model Performance")
    col1, col2 = st.columns(2)

    with col1:
        roc = churn_result["roc_curve"]
        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name="ROC"))
        roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Baseline", line=dict(dash="dash")))
        roc_fig.update_layout(
            template=PLOTLY_TEMPLATE,
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            transition_duration=500,
        )
        st.plotly_chart(roc_fig, use_container_width=True)

    with col2:
        importance = churn_result["feature_importance"]
        fig = px.bar(
            importance,
            x="importance",
            y="feature",
            orientation="h",
            template=PLOTLY_TEMPLATE,
        )
        fig.update_layout(title="Feature Importance", transition_duration=500)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Churn Probability Distribution")
    hist_fig = px.histogram(filtered, x="churn_probability", nbins=30, template=PLOTLY_TEMPLATE)
    hist_fig.update_layout(title="Churn Probability Histogram", transition_duration=500)
    st.plotly_chart(hist_fig, use_container_width=True)

    st.markdown("### High-Risk Customers")
    st.dataframe(
        filtered.sort_values("churn_probability", ascending=False).head(50)[
            [
                "customer_id",
                "rfm_segment",
                "cluster_id",
                "churn_probability",
                "total_spend",
                "satisfaction_score",
            ]
        ],
        use_container_width=True,
        height=360,
    )

    st.markdown("### AI Retention Playbook")
    ai_enabled = st.toggle(AI_TOGGLE_LABEL, value=True, key="ai_retention_playbook")

    segment_summary = (
        filtered.groupby("rfm_segment")
        .agg(
            size=("customer_id", "count"),
            churn_rate=("churn_probability", "mean"),
            avg_recency=("recency", "mean"),
            avg_satisfaction=("satisfaction_score", "mean"),
            avg_clv=("clv_12m", "mean"),
        )
        .reset_index()
    )
    top_segment = segment_summary.sort_values("churn_rate", ascending=False).iloc[0]
    cache_key = (
        "churn_playbook:"
        f"{top_segment['rfm_segment']}:"
        f"{int(top_segment['size'])}:"
        f"{top_segment['churn_rate']:.3f}:"
        f"{top_segment['avg_recency']:.1f}:"
        f"{top_segment['avg_satisfaction']:.1f}:"
        f"{top_segment['avg_clv']:.1f}"
    )
    cached_playbook = st.session_state.get("ai_cache", {}).get(cache_key)
    playbook_text = cached_playbook
    cached_used = cached_playbook is not None

    if ai_enabled:
        if st.button("AI Retention Playbook"):
            if cached_playbook:
                playbook_text = cached_playbook
                cached_used = True
            else:
                prompt = (
                    "Create a 5-step retention playbook for the highest churn-risk segment.\n"
                    f"Segment: {top_segment['rfm_segment']}\n"
                    f"Size: {int(top_segment['size'])}\n"
                    f"Churn probability: {top_segment['churn_rate'] * 100:.1f}%\n"
                    f"Avg recency: {top_segment['avg_recency']:.1f} days\n"
                    f"Avg satisfaction: {top_segment['avg_satisfaction']:.1f}/10\n"
                    f"Avg CLV: ${top_segment['avg_clv']:.2f}\n"
                    "Return numbered steps with clear actions."
                )
                system = "You are a retention strategist. Be concise. Max 400 words."
                playbook = ask_groq(prompt, system, temperature=0.7)
                if playbook:
                    st.session_state.setdefault("ai_cache", {})[cache_key] = playbook
                    playbook_text = playbook
                    cached_used = False
    else:
        playbook_text = (
            "1. Identify high-risk customers and prioritize personal outreach.\n"
            "2. Offer targeted incentives aligned with their recent purchase gaps.\n"
            "3. Improve support follow-ups to resolve recent issues quickly.\n"
            "4. Use lifecycle reminders to bring them back within 14 days.\n"
            "5. Monitor churn lift weekly and refine by segment response."
        )
        cached_used = False

    if playbook_text:
        formatted = playbook_text.replace("\n", "<br/>")
        st.markdown(
            f"""
            <div style="background: rgba(245, 176, 65, 0.12); border-left: 4px solid var(--warning); padding: 16px; border-radius: 12px;">
                <strong>Top Risk Segment:</strong> {top_segment['rfm_segment']}<br/>
                {formatted}
            </div>
            """,
            unsafe_allow_html=True,
        )
        if cached_used:
            st.caption("(cached)")

    st.markdown("### Cluster Risk Gauges")
    gauge_fig = build_cluster_gauges(filtered)
    st.plotly_chart(gauge_fig, use_container_width=True)

    st.markdown("### What-If Simulator")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        recency = st.slider("Recency (days)", int(full_df["recency"].min()), int(full_df["recency"].max()), 30)
    with col_b:
        satisfaction = st.slider(
            "Satisfaction score",
            float(full_df["satisfaction_score"].min()),
            float(full_df["satisfaction_score"].max()),
            7.0,
        )
    with col_c:
        tickets = st.slider(
            "Support tickets",
            int(full_df["support_tickets_raised"].min()),
            int(full_df["support_tickets_raised"].max()),
            1,
        )

    what_if_prob = predict_what_if(
        churn_result,
        {
            "recency": recency,
            "satisfaction_score": satisfaction,
            "support_tickets_raised": tickets,
        },
    )

    risk_color = "green" if what_if_prob < 0.3 else "yellow" if what_if_prob < 0.6 else "red"
    st.markdown(
        f"""
        <div class="strategy-box">
            <strong>Predicted churn probability:</strong>
            <span style="color: var(--{risk_color}); font-size: 1.2rem;">{what_if_prob * 100:.1f}%</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


render_page()
