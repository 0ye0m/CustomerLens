from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.groq_client import ask_groq
from modules.data_manager import get_active_dataset, render_data_source_banner
from utils.helpers import (
    apply_filters,
    compute_clustering,
    compute_recommendations,
    compute_rfm,
    empty_state,
    format_number,
    generate_strategy_pdf,
    render_kpi_row,
    sidebar_filters,
)
from utils.styling import apply_theme

PLOTLY_TEMPLATE = "plotly_dark"
AI_TOGGLE_LABEL = "Enable AI Features"


def configure_page() -> None:
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="CustomerLens | Strategy Engine",
        page_icon=":sparkles:",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def prepare_data() -> Dict[str, object]:
    """Load data and compute required customer features."""
    with st.spinner("Loading customer data..."):
        base_df = get_active_dataset()
    with st.spinner("Computing RFM segments..."):
        rfm_df = compute_rfm(base_df)
    with st.spinner("Running clustering analysis..."):
        cluster_result = compute_clustering(rfm_df)
    return {"df": cluster_result["df"].copy()}


def build_kpis(filtered: pd.DataFrame, strategy_df: pd.DataFrame) -> List[Dict[str, object]]:
    """Build KPI metrics for the strategy engine."""
    segments = strategy_df["rfm_segment"].nunique()
    avg_response = strategy_df["response_rate"].mean()
    avg_order = filtered["avg_order_value"].mean()
    avg_budget = strategy_df["budget_pct"].mean()

    return [
        {
            "label": "Active Segments",
            "value": float(segments),
            "display": f"{segments}",
            "baseline": float(segments),
        },
        {
            "label": "Avg Response Rate",
            "value": float(avg_response),
            "display": f"{avg_response * 100:.1f}%",
            "baseline": float(avg_response),
        },
        {
            "label": "Avg Order Value",
            "value": float(avg_order),
            "display": f"${format_number(avg_order)}",
            "baseline": float(avg_order),
        },
        {
            "label": "Avg Budget Split",
            "value": float(avg_budget),
            "display": f"{avg_budget:.1f}%",
            "baseline": float(avg_budget),
        },
    ]


def build_allocation(segment_strategies: pd.DataFrame, total_budget: float) -> pd.DataFrame:
    """Allocate budget across clusters for a segment."""
    weights = segment_strategies["customer_count"] * segment_strategies["response_rate"]
    if weights.sum() == 0:
        weights = segment_strategies["customer_count"]

    allocation_pct = weights / weights.sum() * 100
    allocation = segment_strategies[["cluster_id", "customer_count"]].copy()
    allocation["allocation_pct"] = allocation_pct
    allocation["allocation_amount"] = allocation_pct / 100 * total_budget
    return allocation


def render_page() -> None:
    """Render the strategy engine page."""
    configure_page()
    apply_theme()

    st.title("Strategy Engine")
    st.caption("AI-assisted recommendations that turn segments into growth campaigns.")
    render_data_source_banner()

    data_bundle = prepare_data()
    full_df = data_bundle["df"]

    filters = sidebar_filters(full_df, cluster_options=sorted(full_df["cluster_id"].unique().tolist()))
    filtered = apply_filters(full_df, filters)

    if filtered.empty:
        empty_state("No customers match the current filters. Adjust the sidebar to continue.")
        return

    with st.spinner("Refreshing strategy recommendations..."):
        filtered_strategies = compute_recommendations(filtered)
    if filtered_strategies.empty:
        empty_state("No strategy data available for the current filters.")
        return

    render_kpi_row(build_kpis(filtered, filtered_strategies))

    segment_options = sorted(filtered_strategies["rfm_segment"].unique().tolist())
    selected_segment = st.selectbox("Select segment", options=segment_options)

    segment_strategies = filtered_strategies[filtered_strategies["rfm_segment"] == selected_segment]
    primary = segment_strategies.sort_values("customer_count", ascending=False).iloc[0]

    budget = st.slider("Total budget", min_value=5000, max_value=200000, value=50000, step=5000)
    allocation = build_allocation(segment_strategies, budget)

    avg_order_value = filtered.loc[filtered["rfm_segment"] == selected_segment, "avg_order_value"].mean()
    expected_orders = segment_strategies["customer_count"].sum() * primary["response_rate"]
    budget_multiplier = min(2.0, 0.6 + budget / 100000)
    expected_lift = expected_orders * avg_order_value * budget_multiplier

    st.markdown("### Recommended Actions")
    st.markdown(
        f"""
        <div class="strategy-box">
            <strong>Channel:</strong> {primary['channel']}<br/>
            <strong>Offer:</strong> {primary['offer']}<br/>
            <strong>Tone:</strong> {primary['tone']}<br/>
            <strong>Expected Response Rate:</strong> {primary['response_rate'] * 100:.1f}%<br/>
            <strong>Estimated Revenue Lift:</strong> ${format_number(expected_lift)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Budget Allocation by Cluster")
    allocation_fig = px.bar(
        allocation,
        x="cluster_id",
        y="allocation_pct",
        text="allocation_pct",
        template=PLOTLY_TEMPLATE,
    )
    allocation_fig.update_layout(transition_duration=500, yaxis_title="Allocation %")
    st.plotly_chart(allocation_fig, use_container_width=True)

    st.dataframe(
        allocation.sort_values("allocation_pct", ascending=False),
        use_container_width=True,
        height=280,
    )

    st.markdown("### Email Template Preview")
    ai_enabled = st.toggle(AI_TOGGLE_LABEL, value=True, key="ai_strategy_email")

    cache_key = (
        "strategy_email:"
        f"{selected_segment}:"
        f"{primary['offer']}:"
        f"{primary['tone']}:"
        f"{primary['channel']}:"
        f"{primary['response_rate']:.2f}"
    )
    cached_email = st.session_state.get("ai_cache", {}).get(cache_key)
    email_text = cached_email
    cached_used = cached_email is not None

    if ai_enabled:
        if st.button("Generate AI Email"):
            if cached_email:
                email_text = cached_email
                cached_used = True
            else:
                behavior_profile = (
                    f"avg spend ${filtered.loc[filtered['rfm_segment'] == selected_segment, 'total_spend'].mean():.2f}, "
                    f"avg recency {filtered.loc[filtered['rfm_segment'] == selected_segment, 'recency'].mean():.1f} days, "
                    f"avg frequency {filtered.loc[filtered['rfm_segment'] == selected_segment, 'frequency'].mean():.2f}"
                )
                prompt = (
                    "Write a marketing email for this segment. Include a subject line and a short preview line.\n"
                    f"Segment: {selected_segment}\n"
                    f"Goal: {primary['offer']}\n"
                    f"Offer type: {primary['offer']}\n"
                    f"Tone: {primary['tone']}\n"
                    f"Customer behavior: {behavior_profile}\n"
                    "Keep it under 200 words."
                )
                system = (
                    "You are an expert email copywriter specializing in retention and growth marketing. "
                    "Be concise. Max 400 words."
                )
                response = ask_groq(prompt, system, temperature=0.7)
                if response:
                    st.session_state.setdefault("ai_cache", {})[cache_key] = response
                    email_text = response
                    cached_used = False
    else:
        email_text = (
            "Subject: Your next best offer is ready\n"
            "Preview: Exclusive perks tailored for your next purchase.\n\n"
            "Hi there,\n\n"
            "We noticed you have been exploring our newest collection. As one of our valued customers, you have "
            "early access to a curated offer designed for your preferences. Click below to activate your perks and "
            "unlock a personalized experience.\n\n"
            "Regards,\nCustomerLens Growth Team"
        )
        cached_used = False

    if email_text:
        formatted_email = email_text.replace("\n", "<br/>")
        st.markdown(
            f"""
            <div style="background: #FFFFFF; color: #0B0E14; border-radius: 12px; padding: 18px; border: 1px solid rgba(0,0,0,0.08);">
                <div style="font-weight: 600;">From: CustomerLens Growth Team</div>
                <div style="margin-bottom: 10px;">{formatted_email}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        word_count = len(email_text.split())
        st.caption(f"Word count: {word_count}")
        if cached_used:
            st.caption("(cached)")

    report_bytes = generate_strategy_pdf(
        selected_segment,
        {
            "channel": primary["channel"],
            "offer": primary["offer"],
            "tone": primary["tone"],
            "response_rate": f"{primary['response_rate'] * 100:.1f}%",
        },
        budget,
        allocation,
    )

    st.download_button(
        "Download strategy report (PDF)",
        data=report_bytes,
        file_name="customerlens_strategy_report.pdf",
        mime="application/pdf",
    )


render_page()
