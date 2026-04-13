from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from modules.data_manager import get_active_dataset, render_data_source_banner
from utils.groq_client import ask_groq
from utils.helpers import (
    apply_filters,
    compute_clustering,
    compute_rfm,
    empty_state,
    format_number,
    render_kpi_row,
    sidebar_filters,
)
from utils.styling import apply_theme

PERSONAS: Dict[str, Dict[str, str]] = {
    "Champions": {
        "icon": "&#x1F3C6;",
        "description": "High spenders with recent, frequent purchases. They love exclusivity.",
        "health": "green",
    },
    "Loyal Customers": {
        "icon": "&#x1F4AA;",
        "description": "Consistent and engaged. They respond well to loyalty perks.",
        "health": "green",
    },
    "Potential Loyalists": {
        "icon": "&#x1F680;",
        "description": "Rising engagement with growing value. Nurture to unlock loyalty.",
        "health": "yellow",
    },
    "New Customers": {
        "icon": "&#x1F389;",
        "description": "Recently acquired with limited history. Focus on onboarding journeys.",
        "health": "yellow",
    },
    "At Risk": {
        "icon": "&#x26A0;",
        "description": "High value but cooling off. Prioritize win-back and care outreach.",
        "health": "red",
    },
    "Can't Lose Them": {
        "icon": "&#x1F6A8;",
        "description": "Valuable yet disengaging quickly. Immediate action needed.",
        "health": "red",
    },
    "Hibernating": {
        "icon": "&#x1F6D1;",
        "description": "Low engagement and low spend. Re-activate with targeted offers.",
        "health": "yellow",
    },
    "Lost Customers": {
        "icon": "&#x1F480;",
        "description": "Lapsed and inactive. Consider reactivation or list hygiene.",
        "health": "red",
    },
}

AI_TOGGLE_LABEL = "Enable AI Features"


def configure_page() -> None:
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="CustomerLens | Segment Personas",
        page_icon=":busts_in_silhouette:",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def prepare_data() -> pd.DataFrame:
    """Load data and compute RFM segmentation."""
    with st.spinner("Loading customer data..."):
        base_df = get_active_dataset()
    with st.spinner("Computing RFM segments..."):
        rfm_df = compute_rfm(base_df)
    with st.spinner("Running clustering analysis..."):
        cluster_result = compute_clustering(rfm_df)

    return cluster_result["df"].copy()


def build_kpis(filtered: pd.DataFrame, baseline: pd.DataFrame) -> List[Dict[str, object]]:
    """Build KPI metrics for the persona page."""
    segment_count = filtered["rfm_segment"].nunique()
    avg_spend = filtered["total_spend"].mean()
    avg_recency = filtered["recency"].mean()
    healthy_share = filtered["rfm_segment"].isin(["Champions", "Loyal Customers"]).mean()

    return [
        {
            "label": "Segments Covered",
            "value": float(segment_count),
            "display": f"{segment_count}",
            "baseline": float(baseline["rfm_segment"].nunique()),
        },
        {
            "label": "Avg Spend",
            "value": float(avg_spend),
            "display": f"${format_number(avg_spend)}",
            "baseline": float(baseline["total_spend"].mean()),
        },
        {
            "label": "Avg Recency",
            "value": float(avg_recency),
            "display": f"{avg_recency:.1f} days",
            "baseline": float(baseline["recency"].mean()),
        },
        {
            "label": "Healthy Share",
            "value": float(healthy_share),
            "display": f"{healthy_share * 100:.1f}%",
            "baseline": float(
                baseline["rfm_segment"].isin(["Champions", "Loyal Customers"]).mean()
            ),
        },
    ]


def render_persona_cards(filtered: pd.DataFrame, ai_enabled: bool) -> None:
    """Render persona cards in a 4x2 grid."""
    segment_stats = (
        filtered.groupby("rfm_segment")
        .agg(
            count=("customer_id", "count"),
            avg_spend=("total_spend", "mean"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
        )
        .reset_index()
    )

    total = segment_stats["count"].sum()

    segments = [
        "Champions",
        "Loyal Customers",
        "Potential Loyalists",
        "New Customers",
        "At Risk",
        "Can't Lose Them",
        "Hibernating",
        "Lost Customers",
    ]

    rows = [segments[i : i + 4] for i in range(0, len(segments), 4)]
    for row in rows:
        cols = st.columns(4)
        for col, segment in zip(cols, row):
            with col:
                stats = segment_stats.loc[segment_stats["rfm_segment"] == segment]
                if stats.empty:
                    count = 0
                    avg_spend = 0.0
                    avg_recency = 0.0
                    avg_frequency = 0.0
                else:
                    stats_row = stats.iloc[0]
                    count = int(stats_row["count"])
                    avg_spend = float(stats_row["avg_spend"])
                    avg_recency = float(stats_row["avg_recency"])
                    avg_frequency = float(stats_row["avg_frequency"])

                persona = PERSONAS.get(segment, PERSONAS["New Customers"])
                pct = (count / total * 100) if total else 0
                cache_key = (
                    f"persona_story:{segment}:{count}:{avg_spend:.1f}:{avg_recency:.1f}:{avg_frequency:.2f}"
                )
                cached_story = st.session_state.get("ai_cache", {}).get(cache_key)
                story_to_show = cached_story
                cached_used = cached_story is not None

                st.markdown(
                    f"""
                    <div class="segment-card">
                        <div style="font-size: 1.6rem;">{persona['icon']} {segment}</div>
                        <div style="color: var(--muted); margin-bottom: 8px;">{count:,} customers ({pct:.1f}%)</div>
                        <div><strong>Avg Spend:</strong> ${format_number(avg_spend)}</div>
                        <div><strong>Avg Recency:</strong> {avg_recency:.1f} days</div>
                        <div><strong>Avg Frequency:</strong> {avg_frequency:.1f}</div>
                        <div style="margin-top: 8px;">{persona['description']}</div>
                        <div style="margin-top: 10px;"><span class="badge {persona['health']}">{persona['health']}</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if ai_enabled:
                    if st.button("AI Insight", key=f"ai_insight_{segment}"):
                        if cached_story:
                            story_to_show = cached_story
                            cached_used = True
                        else:
                            prompt = (
                                "Create a 2-sentence customer story. Include age, shopping cadence, "
                                "and a realistic motivation."
                                f" Segment: {segment}. Avg spend: ${avg_spend:.2f}. "
                                f"Avg recency: {avg_recency:.1f} days. Avg frequency: {avg_frequency:.2f}."
                            )
                            system = "You are a customer insights copywriter. Be concise. Max 400 words."
                            story = ask_groq(prompt, system, temperature=0.7)
                            if story:
                                st.session_state.setdefault("ai_cache", {})[cache_key] = story
                                story_to_show = story
                                cached_used = False
                else:
                    story_to_show = f"Typical {segment} customers are {persona['description'].lower()}"
                    cached_used = False

                if story_to_show:
                    st.markdown(f"*{story_to_show}*")
                    if cached_used:
                        st.caption("(cached)")


def render_page() -> None:
    """Render the segment personas page."""
    configure_page()
    apply_theme()

    st.title("Segment Personas")
    st.caption("Rich persona cards for each RFM segment with behavior cues.")
    render_data_source_banner()

    full_df = prepare_data()
    filters = sidebar_filters(full_df, cluster_options=sorted(full_df["cluster_id"].unique().tolist()))
    filtered = apply_filters(full_df, filters)

    if filtered.empty:
        empty_state("No customers match the current filters. Adjust the sidebar to continue.")
        return

    render_kpi_row(build_kpis(filtered, full_df))
    st.markdown("### Persona Cards")
    ai_enabled = st.toggle(AI_TOGGLE_LABEL, value=True, key="ai_persona_insights")
    render_persona_cards(filtered, ai_enabled)


render_page()
