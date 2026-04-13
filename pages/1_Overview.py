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

from utils.helpers import (
    apply_filters,
    compute_clustering,
    compute_clv,
    compute_rfm,
    empty_state,
    format_number,
    load_data,
    render_kpi_row,
    sidebar_filters,
)
from utils.styling import apply_theme

PLOTLY_TEMPLATE = "plotly_dark"


def configure_page() -> None:
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="CustomerLens | Overview",
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def prepare_data() -> pd.DataFrame:
    """Load and enrich the dataset with RFM, clustering, and CLV."""
    with st.spinner("Loading customer data..."):
        base_df = load_data()
    with st.spinner("Computing RFM segments..."):
        rfm_df = compute_rfm(base_df)
    with st.spinner("Running clustering analysis..."):
        cluster_result = compute_clustering(rfm_df)
    with st.spinner("Forecasting CLV..."):
        clv_result = compute_clv(cluster_result["df"])

    return clv_result["df"].copy()


def build_kpis(filtered: pd.DataFrame, baseline: pd.DataFrame) -> List[Dict[str, object]]:
    """Build KPI metrics for the overview page."""
    total_customers = len(filtered)
    avg_clv = filtered["clv_12m"].mean()
    churn_rate = filtered["churn_flag"].mean()
    avg_satisfaction = filtered["satisfaction_score"].mean()

    return [
        {
            "label": "Total Customers",
            "value": float(total_customers),
            "display": f"{total_customers:,}",
            "baseline": float(len(baseline)),
        },
        {
            "label": "Avg CLV (12m)",
            "value": float(avg_clv),
            "display": f"${format_number(avg_clv)}",
            "baseline": float(baseline["clv_12m"].mean()),
        },
        {
            "label": "Churn Rate",
            "value": float(churn_rate),
            "display": f"{churn_rate * 100:.1f}%",
            "baseline": float(baseline["churn_flag"].mean()),
        },
        {
            "label": "Avg Satisfaction",
            "value": float(avg_satisfaction),
            "display": f"{avg_satisfaction:.1f}/10",
            "baseline": float(baseline["satisfaction_score"].mean()),
        },
    ]


def render_charts(filtered: pd.DataFrame) -> None:
    """Render the overview charts."""
    segment_counts = filtered["rfm_segment"].value_counts().reset_index()
    segment_counts.columns = ["rfm_segment", "count"]

    revenue = (
        filtered.groupby("rfm_segment")["total_spend"]
        .sum()
        .reset_index()
        .sort_values("total_spend", ascending=False)
    )

    country_counts = filtered["country"].value_counts().reset_index()
    country_counts.columns = ["country", "count"]

    signups = (
        filtered.set_index("signup_date")
        .resample("M")
        .size()
        .reset_index(name="signups")
    )

    heatmap_data = (
        filtered.assign(satisfaction_bucket=filtered["satisfaction_score"].round().astype(int))
        .pivot_table(
            index="rfm_segment",
            columns="satisfaction_bucket",
            values="total_spend",
            aggfunc="mean",
        )
        .fillna(0)
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(
            segment_counts,
            names="rfm_segment",
            values="count",
            hole=0.5,
            template=PLOTLY_TEMPLATE,
        )
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(title="RFM Segment Distribution", transition_duration=500)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            revenue,
            x="rfm_segment",
            y="total_spend",
            color="rfm_segment",
            template=PLOTLY_TEMPLATE,
        )
        fig.update_layout(title="Revenue by Segment", transition_duration=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = px.choropleth(
            country_counts,
            locations="country",
            locationmode="country names",
            color="count",
            color_continuous_scale="Blues",
            template=PLOTLY_TEMPLATE,
        )
        fig.update_layout(title="Customers by Country", transition_duration=500)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.line(
            signups,
            x="signup_date",
            y="signups",
            markers=True,
            template=PLOTLY_TEMPLATE,
        )
        fig.update_layout(title="New Customer Signups", transition_duration=500)
        st.plotly_chart(fig, use_container_width=True)

    fig = px.imshow(
        heatmap_data,
        aspect="auto",
        color_continuous_scale="Purples",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(
        title="Spend vs Satisfaction by Segment",
        xaxis_title="Satisfaction Score",
        yaxis_title="RFM Segment",
        transition_duration=500,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_page() -> None:
    """Render the overview dashboard page."""
    configure_page()
    apply_theme()

    st.title("Overview Dashboard")
    st.caption("Executive summary of customer health, revenue, and engagement trends.")

    full_df = prepare_data()
    filters = sidebar_filters(full_df, cluster_options=sorted(full_df["cluster_id"].unique().tolist()))
    filtered = apply_filters(full_df, filters)

    if filtered.empty:
        empty_state("No customers match the current filters. Adjust the sidebar to continue.")
        return

    render_kpi_row(build_kpis(filtered, full_df))
    st.markdown("### Segmentation Insights")
    render_charts(filtered)


render_page()
