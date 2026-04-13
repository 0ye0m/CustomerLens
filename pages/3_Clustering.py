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

from utils.helpers import (
    apply_filters,
    compute_clustering,
    compute_clv,
    compute_dimensionality,
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
        page_title="CustomerLens | Cluster Explorer",
        page_icon=":compass:",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def prepare_data() -> Dict[str, object]:
    """Load data and compute clustering plus embeddings."""
    with st.spinner("Loading customer data..."):
        base_df = load_data()
    with st.spinner("Computing RFM segments..."):
        rfm_df = compute_rfm(base_df)
    with st.spinner("Running clustering analysis..."):
        cluster_result = compute_clustering(rfm_df)
    with st.spinner("Projecting dimensionality..."):
        dim_result = compute_dimensionality(cluster_result["features"]["matrix"])

    cluster_df = cluster_result["df"].copy()
    cluster_df["pca_x"] = dim_result["pca_2d"][:, 0]
    cluster_df["pca_y"] = dim_result["pca_2d"][:, 1]
    cluster_df["pca_z"] = dim_result["pca_3d"][:, 2]
    cluster_df["tsne_x"] = dim_result["tsne_2d"][:, 0]
    cluster_df["tsne_y"] = dim_result["tsne_2d"][:, 1]

    return {"df": cluster_df, "cluster_result": cluster_result}


def build_kpis(comparison: pd.DataFrame, algorithm: str) -> List[Dict[str, object]]:
    """Build KPI metrics for clustering performance."""
    algo_row = comparison.loc[comparison["algorithm"] == algorithm].iloc[0]
    return [
        {
            "label": "Algorithm",
            "value": float(algo_row["score"]),
            "display": algorithm,
            "baseline": float(comparison["score"].max()),
        },
        {
            "label": "Clusters",
            "value": float(algo_row["n_clusters"]),
            "display": f"{int(algo_row['n_clusters'])}",
            "baseline": float(comparison["n_clusters"].max()),
        },
        {
            "label": "Silhouette",
            "value": float(algo_row["silhouette"] or 0),
            "display": f"{(algo_row['silhouette'] or 0):.3f}",
            "baseline": float(comparison["silhouette"].max() or 0),
        },
        {
            "label": "Davies-Bouldin",
            "value": float(algo_row["davies_bouldin"] or 0),
            "display": f"{(algo_row['davies_bouldin'] or 0):.3f}",
            "baseline": float(comparison["davies_bouldin"].min() or 0),
        },
    ]


def render_comparison_table(comparison: pd.DataFrame) -> None:
    """Render the algorithm comparison table with highlights."""
    styled = (
        comparison.style.format({
            "silhouette": "{:.3f}",
            "davies_bouldin": "{:.3f}",
            "calinski_harabasz": "{:.1f}",
            "score": "{:.3f}",
        })
        .highlight_max(subset=["silhouette", "calinski_harabasz", "score"], color="rgba(108, 99, 255, 0.35)")
        .highlight_min(subset=["davies_bouldin"], color="rgba(231, 76, 60, 0.25)")
    )
    st.dataframe(styled, use_container_width=True)


def render_cluster_cards(stats: pd.DataFrame) -> None:
    """Render cluster size cards."""
    stats = stats.sort_values("cluster_size", ascending=False).head(8)
    rows = [stats.iloc[i : i + 4] for i in range(0, len(stats), 4)]
    for row in rows:
        cols = st.columns(4)
        for col, (_, cluster) in zip(cols, row.iterrows()):
            with col:
                st.markdown(
                    f"""
                    <div class="card">
                        <strong>Cluster {int(cluster['cluster_id'])}</strong><br/>
                        Size: {int(cluster['cluster_size']):,}<br/>
                        Avg Spend: ${format_number(cluster['total_spend'])}<br/>
                        Avg Satisfaction: {cluster['satisfaction_score']:.1f}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_page() -> None:
    """Render the cluster explorer page."""
    configure_page()
    apply_theme()

    st.title("Cluster Explorer")
    st.caption("Compare clustering algorithms and inspect segment structures in 2D and 3D.")

    data_bundle = prepare_data()
    base_df = data_bundle["df"].copy()
    cluster_result = data_bundle["cluster_result"]

    algorithm = st.radio(
        "Select algorithm",
        options=["K-Means", "DBSCAN", "Hierarchical"],
        horizontal=True,
    )

    base_df["cluster_id"] = cluster_result["labels"][algorithm]

    with st.spinner("Forecasting CLV for cluster profiles..."):
        clv_result = compute_clv(base_df)
    full_df = clv_result["df"].copy()

    filters = sidebar_filters(full_df, cluster_options=sorted(full_df["cluster_id"].unique().tolist()))
    filtered = apply_filters(full_df, filters)

    if filtered.empty:
        empty_state("No customers match the current filters. Adjust the sidebar to continue.")
        return

    render_kpi_row(build_kpis(cluster_result["comparison"], algorithm))

    st.markdown("### Algorithm Comparison")
    render_comparison_table(cluster_result["comparison"])

    st.markdown("### Cluster Embeddings")
    filtered["cluster_label"] = filtered["cluster_id"].astype(str)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(
            filtered,
            x="tsne_x",
            y="tsne_y",
            color="cluster_label",
            template=PLOTLY_TEMPLATE,
        )
        fig.update_layout(title="t-SNE 2D", transition_duration=500, legend_title_text="Cluster")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter_3d(
            filtered,
            x="pca_x",
            y="pca_y",
            z="pca_z",
            color="cluster_label",
            template=PLOTLY_TEMPLATE,
            height=520,
        )
        fig.update_layout(title="PCA 3D", transition_duration=500, legend_title_text="Cluster")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Cluster Feature Profile")
    cluster_stats = (
        filtered.groupby("cluster_id")
        .agg(
            total_spend=("total_spend", "mean"),
            frequency=("frequency", "mean"),
            recency=("recency", "mean"),
            satisfaction_score=("satisfaction_score", "mean"),
            clv_12m=("clv_12m", "mean"),
            cluster_size=("customer_id", "count"),
        )
        .reset_index()
    )

    render_cluster_cards(cluster_stats)

    selected_cluster = st.selectbox(
        "Select cluster for radar view",
        options=cluster_stats["cluster_id"].tolist(),
    )

    radar_features = ["total_spend", "frequency", "recency", "satisfaction_score", "clv_12m"]
    max_vals = cluster_stats[radar_features].max().replace(0, 1)
    cluster_row = cluster_stats.loc[cluster_stats["cluster_id"] == selected_cluster].iloc[0]
    radar_values = (cluster_row[radar_features] / max_vals).values.tolist()

    radar_fig = go.Figure()
    radar_fig.add_trace(
        go.Scatterpolar(
            r=radar_values,
            theta=["Spend", "Frequency", "Recency", "Satisfaction", "CLV"],
            fill="toself",
            name=f"Cluster {int(selected_cluster)}",
        )
    )
    radar_fig.update_layout(
        template=PLOTLY_TEMPLATE,
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        transition_duration=500,
    )
    st.plotly_chart(radar_fig, use_container_width=True)


render_page()
