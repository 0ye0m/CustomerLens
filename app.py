from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
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


def configure_page() -> None:
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="CustomerLens",
        page_icon=":sparkles:",
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


def build_kpis(filtered, baseline) -> List[Dict[str, object]]:
    """Build KPI metrics for the landing page."""
    total_customers = len(filtered)
    segments = filtered["rfm_segment"].nunique()
    avg_clv = filtered["clv_12m"].mean()
    churn_rate = filtered["churn_flag"].mean()

    return [
        {
            "label": "Customers",
            "value": float(total_customers),
            "display": f"{total_customers:,}",
            "baseline": float(len(baseline)),
        },
        {
            "label": "Active Segments",
            "value": float(segments),
            "display": f"{segments}",
            "baseline": float(baseline["rfm_segment"].nunique()),
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
    ]


def render_page() -> None:
    """Render the main landing page."""
    configure_page()
    apply_theme()

    st.title("CustomerLens")
    st.caption("Multi-Dimensional Segmentation Intelligence Platform")

    full_df = prepare_data()
    filters = sidebar_filters(full_df, cluster_options=sorted(full_df["cluster_id"].unique().tolist()))
    filtered = apply_filters(full_df, filters)

    if filtered.empty:
        empty_state("No customers match the current filters. Adjust the sidebar to continue.")
        return

    render_kpi_row(build_kpis(filtered, full_df))

    st.markdown("### Welcome")
    st.markdown(
        """
        CustomerLens turns raw customer data into executive-ready segmentation intelligence. Explore the
        dedicated pages to analyze RFM behavior, compare clustering models, forecast CLV, and launch
        retention strategies.
        """
    )

    st.markdown("### Quick Actions")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="card">
                <strong>Overview Dashboard</strong><br/>
                High-level KPIs, revenue distribution, and geographic coverage.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="card">
                <strong>Cluster Explorer</strong><br/>
                Compare K-Means, DBSCAN, and Hierarchical segmentation models.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="card">
                <strong>Strategy Engine</strong><br/>
                Generate personalized outreach strategies and export reports.
            </div>
            """,
            unsafe_allow_html=True,
        )


render_page()
