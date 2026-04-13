from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from modules import rfm_analysis
from modules.data_manager import get_active_dataset, render_data_source_banner
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

PLOTLY_TEMPLATE = "plotly_dark"


def configure_page() -> None:
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="CustomerLens | RFM Analysis",
        page_icon=":mag:",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def prepare_data() -> pd.DataFrame:
    """Load and enrich the dataset with RFM and clustering."""
    with st.spinner("Loading customer data..."):
        base_df = get_active_dataset()
    with st.spinner("Computing RFM segments..."):
        rfm_df = compute_rfm(base_df)
    with st.spinner("Running clustering analysis..."):
        cluster_result = compute_clustering(rfm_df)

    return cluster_result["df"].copy()


def build_kpis(filtered: pd.DataFrame, baseline: pd.DataFrame) -> List[Dict[str, object]]:
    """Build KPI metrics for the RFM page."""
    avg_recency = filtered["recency"].mean()
    avg_frequency = filtered["frequency"].mean()
    avg_monetary = filtered["monetary"].mean()
    top_segment = filtered["rfm_segment"].value_counts(normalize=True).max()

    return [
        {
            "label": "Avg Recency",
            "value": float(avg_recency),
            "display": f"{avg_recency:.1f} days",
            "baseline": float(baseline["recency"].mean()),
        },
        {
            "label": "Avg Frequency",
            "value": float(avg_frequency),
            "display": f"{avg_frequency:.1f}",
            "baseline": float(baseline["frequency"].mean()),
        },
        {
            "label": "Avg Monetary",
            "value": float(avg_monetary),
            "display": f"${format_number(avg_monetary)}",
            "baseline": float(baseline["monetary"].mean()),
        },
        {
            "label": "Top Segment Share",
            "value": float(top_segment),
            "display": f"{top_segment * 100:.1f}%",
            "baseline": float(baseline["rfm_segment"].value_counts(normalize=True).max()),
        },
    ]


def build_sankey(filtered: pd.DataFrame) -> go.Figure:
    """Build a Sankey diagram that approximates segment migration."""
    prev_df = filtered.copy()
    prev_df["last_purchase_date"] = prev_df["last_purchase_date"] - pd.to_timedelta(30, unit="D")
    prev_rfm = rfm_analysis.calculate_rfm(prev_df)

    flow = pd.DataFrame(
        {
            "source": prev_rfm["rfm_segment"].values,
            "target": filtered["rfm_segment"].values,
        }
    )
    counts = flow.value_counts().reset_index(name="value")

    labels = sorted(set(counts["source"]).union(set(counts["target"])))
    label_index = {label: idx for idx, label in enumerate(labels)}

    sankey = go.Figure(
        data=[
            go.Sankey(
                node={"label": labels, "pad": 12, "thickness": 14},
                link={
                    "source": counts["source"].map(label_index),
                    "target": counts["target"].map(label_index),
                    "value": counts["value"],
                },
            )
        ]
    )
    sankey.update_layout(template=PLOTLY_TEMPLATE, transition_duration=500)
    return sankey


def render_page() -> None:
    """Render the RFM analysis page."""
    configure_page()
    apply_theme()

    st.title("RFM Analysis")
    st.caption("Deep dive into recency, frequency, and monetary value segmentation.")
    render_data_source_banner()

    full_df = prepare_data()
    filters = sidebar_filters(full_df, cluster_options=sorted(full_df["cluster_id"].unique().tolist()))
    filtered = apply_filters(full_df, filters)

    if filtered.empty:
        empty_state("No customers match the current filters. Adjust the sidebar to continue.")
        return

    render_kpi_row(build_kpis(filtered, full_df))

    st.markdown("### RFM Behavior Landscape")
    fig = px.scatter_3d(
        filtered,
        x="recency",
        y="frequency",
        z="monetary",
        color="rfm_segment",
        template=PLOTLY_TEMPLATE,
        height=600,
    )
    fig.update_layout(transition_duration=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Score Distributions")
    cols = st.columns(3)
    for col, score, title in zip(cols, ["r_score", "f_score", "m_score"], ["Recency", "Frequency", "Monetary"]):
        with col:
            fig = px.histogram(filtered, x=score, nbins=5, template=PLOTLY_TEMPLATE)
            fig.update_layout(title=f"{title} Score", transition_duration=500)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Segment Migration Flow")
    with st.spinner("Modeling segment movement..."):
        sankey_fig = build_sankey(filtered)
    st.plotly_chart(sankey_fig, use_container_width=True)

    st.markdown("### Segment Explorer")
    segments = sorted(filtered["rfm_segment"].unique().tolist())
    selected_segments = st.multiselect("Segments", options=segments, default=segments)
    search = st.text_input("Search by customer_id, city, or country")

    table_df = filtered.copy()
    if selected_segments:
        table_df = table_df[table_df["rfm_segment"].isin(selected_segments)]
    if search:
        search_lower = search.lower()
        mask = (
            table_df["customer_id"].str.lower().str.contains(search_lower)
            | table_df["city"].str.lower().str.contains(search_lower)
            | table_df["country"].str.lower().str.contains(search_lower)
        )
        table_df = table_df[mask]

    st.dataframe(
        table_df.sort_values("rfm_score", ascending=False)[
            [
                "customer_id",
                "rfm_segment",
                "r_score",
                "f_score",
                "m_score",
                "rfm_score",
                "total_spend",
                "total_orders",
            ]
        ],
        use_container_width=True,
        height=420,
    )

    csv_data = table_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download segment CSV",
        data=csv_data,
        file_name="customerlens_rfm_segments.csv",
        mime="text/csv",
    )


render_page()
