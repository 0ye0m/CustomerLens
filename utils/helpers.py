from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from fpdf import FPDF

from data.generate_data import generate_dataset
from modules import churn_model, clv_model, clustering, dimensionality, recommender, rfm_analysis
from utils.groq_client import render_groq_sidebar


@dataclass(frozen=True)
class FilterState:
    """Represents the current sidebar filter selections."""

    date_range: Tuple[pd.Timestamp, pd.Timestamp]
    countries: List[str]
    clusters: List[int]


def get_project_root() -> Path:
    """Return the project root folder based on this file's location."""
    return Path(__file__).resolve().parents[1]


def get_data_path() -> Path:
    """Return the CSV path for the synthetic dataset."""
    return get_project_root() / "data" / "customers.csv"


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the customer dataset, generating it if missing."""
    data_path = get_data_path()
    if not data_path.exists():
        generate_dataset(data_path)

    df = pd.read_csv(data_path)
    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
    df["last_purchase_date"] = pd.to_datetime(df["last_purchase_date"], errors="coerce")
    return df


@st.cache_data
def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Compute RFM scores and segments for the provided dataset."""
    return rfm_analysis.calculate_rfm(df)


@st.cache_data
def compute_clustering(df: pd.DataFrame) -> Dict[str, Any]:
    """Run clustering models and return labels, metrics, and features."""
    return clustering.run_clustering(df)


@st.cache_data
def compute_dimensionality(feature_matrix: np.ndarray) -> Dict[str, Any]:
    """Run PCA and t-SNE to generate embeddings."""
    return dimensionality.run_dimensionality(feature_matrix)


@st.cache_data
def compute_churn(df: pd.DataFrame) -> Dict[str, Any]:
    """Train the churn model and score customers."""
    return churn_model.train_churn_model(df)


@st.cache_data
def compute_clv(df: pd.DataFrame) -> Dict[str, Any]:
    """Train the CLV model and return tiers and forecasts."""
    return clv_model.train_clv_model(df)


@st.cache_data
def compute_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """Generate recommendation strategies for each segment and cluster."""
    return recommender.build_strategy_table(df)


def render_sidebar_nav() -> None:
    """Render sidebar navigation links for the multipage app."""
    nav_items = [
        ("app.py", "Home"),
        ("pages/1_Overview.py", "Overview"),
        ("pages/2_RFM_Analysis.py", "RFM Analysis"),
        ("pages/3_Clustering.py", "Clustering"),
        ("pages/4_Churn_Prediction.py", "Churn Prediction"),
        ("pages/5_CLV_Forecast.py", "CLV Forecast"),
        ("pages/6_Segment_Personas.py", "Segment Personas"),
        ("pages/7_Strategy_Engine.py", "Strategy Engine"),
        ("pages/8_AI_Analyst.py", "AI Analyst"),
    ]

    try:
        for path, label in nav_items:
            st.sidebar.page_link(path, label=label)
    except Exception:
        for _, label in nav_items:
            st.sidebar.markdown(f"- {label}")


def sidebar_filters(df: pd.DataFrame, cluster_options: Iterable[int] | None = None) -> FilterState:
    """Render the sidebar filters and return current selections."""
    st.sidebar.markdown("## CustomerLens")
    st.sidebar.caption("Multi-Dimensional Segmentation Intelligence Platform")
    st.sidebar.markdown("---")
    render_sidebar_nav()
    st.sidebar.markdown("---")

    min_date = df["signup_date"].min()
    max_date = df["signup_date"].max()
    date_range = st.sidebar.date_input(
        "Signup date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    available_countries = sorted(df["country"].dropna().unique().tolist())
    selected_countries = st.sidebar.multiselect(
        "Country",
        options=available_countries,
        default=available_countries,
    )

    cluster_list = sorted(set(cluster_options or []))
    if cluster_list:
        selected_clusters = st.sidebar.multiselect(
            "Cluster",
            options=cluster_list,
            default=cluster_list,
        )
    else:
        st.sidebar.info("No clusters available for the current view.")
        selected_clusters = []

    st.sidebar.markdown("---")
    render_groq_sidebar()

    start_date, end_date = date_range if isinstance(date_range, tuple) else (min_date, max_date)
    return FilterState(
        date_range=(pd.to_datetime(start_date), pd.to_datetime(end_date)),
        countries=selected_countries,
        clusters=selected_clusters,
    )


def apply_filters(df: pd.DataFrame, filters: FilterState) -> pd.DataFrame:
    """Filter the dataset based on sidebar selections."""
    filtered = df.copy()
    filtered = filtered[
        (filtered["signup_date"] >= filters.date_range[0]) & (filtered["signup_date"] <= filters.date_range[1])
    ]

    if filters.countries:
        filtered = filtered[filtered["country"].isin(filters.countries)]

    if filters.clusters:
        filtered = filtered[filtered["cluster_id"].isin(filters.clusters)]

    return filtered


def format_number(value: float, suffix: str = "") -> str:
    """Format large numbers with suffixes."""
    if np.isnan(value):
        return "0"
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M{suffix}"
    if abs_value >= 1_000:
        return f"{value / 1_000:.2f}K{suffix}"
    return f"{value:.2f}{suffix}" if isinstance(value, float) else f"{value}{suffix}"


def delta_badge(current: float, baseline: float) -> Tuple[str, str]:
    """Return delta text and color class compared to baseline."""
    if baseline == 0:
        return "0%", "yellow"
    delta_pct = (current - baseline) / abs(baseline) * 100
    color = "green" if delta_pct >= 0 else "red"
    return f"{delta_pct:+.1f}%", color


def render_kpi_row(metrics: List[Dict[str, Any]]) -> None:
    """Render a KPI row using HTML cards."""
    cols = st.columns(len(metrics))
    for col, metric in zip(cols, metrics):
        with col:
            delta_text, delta_color = delta_badge(metric["value"], metric.get("baseline", metric["value"]))
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">{metric['label']}</div>
                    <div class="kpi-value">{metric['display']}</div>
                    <div class="kpi-delta" style="color: var(--{delta_color});">{delta_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def empty_state(message: str) -> None:
    """Render a friendly empty state message."""
    st.warning(message)


def generate_strategy_pdf(
    segment: str,
    strategy: Dict[str, Any],
    budget: float,
    allocation: pd.DataFrame,
) -> bytes:
    """Generate a PDF summary for the selected strategy."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()

    pdf.set_font("Helvetica", style="B", size=16)
    pdf.cell(0, 10, "CustomerLens Strategy Report", ln=True)

    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 8, f"Segment: {segment}", ln=True)
    pdf.cell(0, 8, f"Total Budget: ${budget:,.0f}", ln=True)

    pdf.ln(4)
    pdf.set_font("Helvetica", style="B", size=12)
    pdf.cell(0, 8, "Recommended Actions", ln=True)

    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 6, f"Channel: {strategy['channel']}")
    pdf.multi_cell(0, 6, f"Offer: {strategy['offer']}")
    pdf.multi_cell(0, 6, f"Tone: {strategy['tone']}")
    pdf.multi_cell(0, 6, f"Expected Response Rate: {strategy['response_rate']}")

    pdf.ln(4)
    pdf.set_font("Helvetica", style="B", size=12)
    pdf.cell(0, 8, "Budget Allocation", ln=True)

    pdf.set_font("Helvetica", size=10)
    for _, row in allocation.iterrows():
        pdf.cell(0, 6, f"Cluster {row['cluster_id']}: {row['allocation_pct']:.1f}%", ln=True)

    return pdf.output(dest="S").encode("latin-1")
