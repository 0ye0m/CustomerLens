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

from modules.data_manager import get_active_dataset, render_data_source_banner
from utils.helpers import (
    apply_filters,
    compute_churn,
    compute_clustering,
    compute_clv,
    compute_rfm,
    empty_state,
    format_number,
    render_kpi_row,
    sidebar_filters,
)
from utils.styling import apply_theme

PLOTLY_TEMPLATE = "plotly_dark"


def configure_page() -> None:
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="CustomerLens | CLV Forecast",
        page_icon=":money_with_wings:",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def prepare_data() -> Dict[str, object]:
    """Load data and compute CLV forecasting outputs."""
    with st.spinner("Loading customer data..."):
        base_df = get_active_dataset()
    with st.spinner("Computing RFM segments..."):
        rfm_df = compute_rfm(base_df)
    with st.spinner("Running clustering analysis..."):
        cluster_result = compute_clustering(rfm_df)
    with st.spinner("Forecasting CLV..."):
        clv_result = compute_clv(cluster_result["df"])
    with st.spinner("Scoring churn probabilities..."):
        churn_result = compute_churn(clv_result["df"])

    return {"df": churn_result["df"].copy()}


def build_kpis(filtered: pd.DataFrame, baseline: pd.DataFrame) -> List[Dict[str, object]]:
    """Build KPI metrics for CLV analysis."""
    avg_clv = filtered["clv_12m"].mean()
    platinum_share = (filtered["clv_tier"] == "Platinum").mean()
    growth = filtered["clv_12m"].mean() / filtered["historical_clv"].mean() - 1
    top_clv = filtered["clv_12m"].max()

    return [
        {
            "label": "Avg 12m CLV",
            "value": float(avg_clv),
            "display": f"${format_number(avg_clv)}",
            "baseline": float(baseline["clv_12m"].mean()),
        },
        {
            "label": "Platinum Share",
            "value": float(platinum_share),
            "display": f"{platinum_share * 100:.1f}%",
            "baseline": float((baseline["clv_tier"] == "Platinum").mean()),
        },
        {
            "label": "CLV Growth",
            "value": float(growth),
            "display": f"{growth * 100:.1f}%",
            "baseline": float(
                baseline["clv_12m"].mean() / baseline["historical_clv"].mean() - 1
            ),
        },
        {
            "label": "Top CLV",
            "value": float(top_clv),
            "display": f"${format_number(top_clv)}",
            "baseline": float(baseline["clv_12m"].max()),
        },
    ]


def render_page() -> None:
    """Render the CLV forecasting page."""
    configure_page()
    apply_theme()

    st.title("CLV Forecast")
    st.caption("Forecast long-term value and connect it to churn risk.")
    render_data_source_banner()

    full_df = prepare_data()["df"]
    filters = sidebar_filters(full_df, cluster_options=sorted(full_df["cluster_id"].unique().tolist()))
    filtered = apply_filters(full_df, filters)

    if filtered.empty:
        empty_state("No customers match the current filters. Adjust the sidebar to continue.")
        return

    render_kpi_row(build_kpis(filtered, full_df))

    st.markdown("### CLV Tier Distribution")
    treemap_fig = px.treemap(
        filtered,
        path=["clv_tier", "cluster_id"],
        values="clv_12m",
        color="clv_tier",
        template=PLOTLY_TEMPLATE,
    )
    treemap_fig.update_layout(transition_duration=500)
    st.plotly_chart(treemap_fig, use_container_width=True)

    st.markdown("### Predicted 12-Month CLV by Segment")
    segment_clv = (
        filtered.groupby(["rfm_segment", "cluster_id"])["clv_12m"]
        .mean()
        .reset_index()
    )
    bar_fig = px.bar(
        segment_clv,
        x="rfm_segment",
        y="clv_12m",
        color="cluster_id",
        barmode="group",
        template=PLOTLY_TEMPLATE,
    )
    bar_fig.update_layout(transition_duration=500, xaxis_title="Segment", yaxis_title="CLV")
    st.plotly_chart(bar_fig, use_container_width=True)

    st.markdown("### CLV vs Churn Risk")
    scatter_fig = px.scatter(
        filtered,
        x="clv_12m",
        y="churn_probability",
        size="total_orders",
        color="rfm_segment",
        template=PLOTLY_TEMPLATE,
        hover_data=["customer_id", "cluster_id"],
    )
    scatter_fig.update_layout(transition_duration=500, xaxis_title="CLV (12m)", yaxis_title="Churn Probability")
    st.plotly_chart(scatter_fig, use_container_width=True)

    st.markdown("### Top 20 Highest CLV Customers")
    st.dataframe(
        filtered.sort_values("clv_12m", ascending=False).head(20)[
            [
                "customer_id",
                "clv_12m",
                "clv_tier",
                "total_spend",
                "total_orders",
                "rfm_segment",
            ]
        ],
        use_container_width=True,
        height=360,
    )


render_page()
