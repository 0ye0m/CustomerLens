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


AI_TOGGLE_LABEL = "Enable AI Features"
STRATEGY_HEADINGS = [
    "\U0001F3AF Campaign Objective",
    "\U0001F4E3 Recommended Channels (with % budget split)",
    "\U0001F4E7 Message & Offer Ideas (3 specific examples)",
    "\U0001F4C5 Timeline & Frequency",
    "\U0001F4CA Success Metrics to Track",
    "\U0001F53D Risks to Avoid",
]


def configure_page() -> None:
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="CustomerLens | AI Analyst",
        page_icon=":robot_face:",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def prepare_data() -> pd.DataFrame:
    """Load the dataset and compute core enrichments."""
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

    return churn_result["df"].copy()


def build_kpis(filtered: pd.DataFrame, baseline: pd.DataFrame) -> List[Dict[str, object]]:
    """Build KPI metrics for the AI analyst page."""
    churn_rate = filtered["churn_flag"].mean()
    avg_clv = filtered["clv_12m"].mean()
    avg_recency = filtered["recency"].mean()
    total_customers = len(filtered)

    return [
        {
            "label": "Customers",
            "value": float(total_customers),
            "display": f"{total_customers:,}",
            "baseline": float(len(baseline)),
        },
        {
            "label": "Avg CLV",
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
            "label": "Avg Recency",
            "value": float(avg_recency),
            "display": f"{avg_recency:.1f} days",
            "baseline": float(baseline["recency"].mean()),
        },
    ]


def _segment_stats(df: pd.DataFrame, segment: str, by: str) -> Dict[str, float]:
    """Compute summary stats for a segment or cluster."""
    if by == "rfm_segment":
        scoped = df[df["rfm_segment"] == segment]
    else:
        scoped = df[df["cluster_id"].astype(str) == segment]

    if scoped.empty:
        return {
            "segment": segment,
            "size": 0,
            "avg_spend": 0.0,
            "avg_recency": 0.0,
            "avg_frequency": 0.0,
            "churn_rate": 0.0,
            "avg_clv": 0.0,
        }

    return {
        "segment": segment,
        "size": int(len(scoped)),
        "avg_spend": float(scoped["total_spend"].mean()),
        "avg_recency": float(scoped["recency"].mean()),
        "avg_frequency": float(scoped["frequency"].mean()),
        "churn_rate": float(scoped["churn_flag"].mean()),
        "avg_clv": float(scoped["clv_12m"].mean()),
    }


def _fingerprint(stats: Dict[str, float]) -> str:
    """Create a compact cache fingerprint for a stats dict."""
    return "-".join(
        [
            stats["segment"],
            str(stats["size"]),
            f"{stats['avg_spend']:.1f}",
            f"{stats['avg_recency']:.1f}",
            f"{stats['avg_frequency']:.2f}",
            f"{stats['churn_rate']:.3f}",
            f"{stats['avg_clv']:.1f}",
        ]
    )


def _ai_cache_key(prefix: str, stats: Dict[str, float]) -> str:
    """Build a session cache key for AI responses."""
    return f"{prefix}:{_fingerprint(stats)}"


def _store_ai_response(key: str, response: str) -> None:
    """Store AI response in session cache."""
    st.session_state.setdefault("ai_cache", {})
    st.session_state["ai_cache"][key] = response


def _get_ai_response(key: str) -> str | None:
    """Fetch AI response from session cache."""
    cache = st.session_state.get("ai_cache", {})
    return cache.get(key)


def _render_cached_label(cached: bool) -> None:
    """Render a cached label when applicable."""
    if cached:
        st.caption("(cached)")


def _parse_strategy_sections(text: str) -> List[tuple[str, str]]:
    """Parse strategy text into titled sections."""
    sections: List[tuple[str, str]] = []
    current_title = ""
    current_body: List[str] = []

    for line in text.splitlines():
        line_stripped = line.strip()
        heading_match = next((h for h in STRATEGY_HEADINGS if line_stripped.startswith(h)), None)
        if heading_match:
            if current_title:
                sections.append((current_title, "\n".join(current_body).strip()))
            current_title = line_stripped
            current_body = []
        else:
            current_body.append(line)

    if current_title:
        sections.append((current_title, "\n".join(current_body).strip()))

    if not sections and text.strip():
        sections.append(("Strategy", text.strip()))

    return sections


def _summary_fingerprint(df: pd.DataFrame) -> str:
    """Create a summary fingerprint for caching chat/report calls."""
    churn_rate = df["churn_flag"].mean()
    avg_clv = df["clv_12m"].mean()
    top_cluster = df.groupby("cluster_id")["total_spend"].sum().idxmax()
    risky_segment = df.groupby("rfm_segment")["churn_flag"].mean().idxmax()
    return f"{len(df)}|{churn_rate:.3f}|{avg_clv:.1f}|{top_cluster}|{risky_segment}"


def tab_segment_explainer(df: pd.DataFrame) -> None:
    """Render the Segment Explainer tab."""
    st.subheader("Segment Explainer")
    ai_enabled = st.toggle(AI_TOGGLE_LABEL, value=True, key="ai_segment_explainer")

    segment_type = st.radio("Segment type", options=["RFM Segment", "Cluster"], horizontal=True)
    by = "rfm_segment" if segment_type == "RFM Segment" else "cluster_id"

    if by == "rfm_segment":
        options = sorted(df["rfm_segment"].unique().tolist())
    else:
        options = sorted(df["cluster_id"].astype(str).unique().tolist())

    selected = st.selectbox("Select segment", options=options)
    stats = _segment_stats(df, selected, by)

    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    metrics_col1.metric("Avg Spend", f"${format_number(stats['avg_spend'])}")
    metrics_col2.metric("Avg Recency", f"{stats['avg_recency']:.1f} days")
    metrics_col3.metric("Avg Frequency", f"{stats['avg_frequency']:.2f}")

    metrics_col4, metrics_col5, metrics_col6 = st.columns(3)
    metrics_col4.metric("Churn Rate", f"{stats['churn_rate'] * 100:.1f}%")
    metrics_col5.metric("Avg CLV", f"${format_number(stats['avg_clv'])}")
    metrics_col6.metric("Segment Size", f"{stats['size']:,}")

    cache_key = _ai_cache_key("segment_explainer", stats)
    cached = _get_ai_response(cache_key)

    if not ai_enabled:
        st.info("AI features are disabled. Enable the toggle to generate insights.")
        return

    col_a, col_b = st.columns([1, 1])
    response_text = cached
    cached_used = cached is not None
    with col_a:
        if st.button("Explain This Segment"):
            if cached:
                response_text = cached
                cached_used = True
            else:
                prompt = (
                    "Segment stats:\n"
                    f"- Segment: {stats['segment']}\n"
                    f"- Size: {stats['size']}\n"
                    f"- Avg spend: ${stats['avg_spend']:.2f}\n"
                    f"- Avg recency: {stats['avg_recency']:.1f} days\n"
                    f"- Avg frequency: {stats['avg_frequency']:.2f}\n"
                    f"- Churn rate: {stats['churn_rate'] * 100:.1f}%\n"
                    f"- Avg CLV: ${stats['avg_clv']:.2f}\n"
                    "Explain: who these customers are, why they behave this way, the biggest risk or "
                    "opportunity, and one surprising insight."
                )
                system = (
                    "You are a senior customer analytics expert. Explain customer segments in clear, actionable "
                    "business language. Be specific and insightful. Be concise. Max 400 words."
                )
                response = ask_groq(prompt, system, temperature=0.2)
                if response:
                    _store_ai_response(cache_key, response)
                    response_text = response
                    cached_used = False

    with col_b:
        if st.button("Regenerate"):
            prompt = (
                "Provide a fresh perspective.\n"
                f"Segment: {stats['segment']}\n"
                f"Size: {stats['size']}\n"
                f"Avg spend: ${stats['avg_spend']:.2f}\n"
                f"Avg recency: {stats['avg_recency']:.1f} days\n"
                f"Avg frequency: {stats['avg_frequency']:.2f}\n"
                f"Churn rate: {stats['churn_rate'] * 100:.1f}%\n"
                f"Avg CLV: ${stats['avg_clv']:.2f}\n"
                "Explain: who these customers are, why they behave this way, the biggest risk or "
                "opportunity, and one surprising insight."
            )
            system = (
                "You are a senior customer analytics expert. Explain customer segments in clear, actionable "
                "business language. Be specific and insightful. Be concise. Max 400 words."
            )
            response = ask_groq(prompt, system, temperature=0.2)
            if response:
                _store_ai_response(cache_key, response)
                response_text = response
                cached_used = False

    if response_text:
        st.info(response_text)
        _render_cached_label(cached_used)
        st.code(response_text, language="text")


def tab_strategy_generator(df: pd.DataFrame) -> None:
    """Render the Strategy Generator tab."""
    st.subheader("Strategy Generator")
    ai_enabled = st.toggle(AI_TOGGLE_LABEL, value=True, key="ai_strategy_generator")

    segment_options = sorted(df["rfm_segment"].unique().tolist())
    segment = st.selectbox("Target segment", options=segment_options)
    budget = st.slider("Budget ($)", min_value=1000, max_value=100000, value=25000, step=1000)
    goal = st.selectbox("Campaign goal", options=["Retention", "Acquisition", "Upsell", "Win-back"])

    stats = _segment_stats(df, segment, "rfm_segment")
    cache_key = _ai_cache_key("strategy_generator", stats) + f":{budget}:{goal}"

    if not ai_enabled:
        st.info("AI features are disabled. Enable the toggle to generate strategies.")
        return

    cached = _get_ai_response(cache_key)
    generate_clicked = st.button("Generate Strategy")
    response_text = cached
    cached_used = cached is not None

    if generate_clicked and not cached:
        prompt = (
            "Segment stats:\n"
            f"- Segment: {stats['segment']}\n"
            f"- Size: {stats['size']}\n"
            f"- Avg spend: ${stats['avg_spend']:.2f}\n"
            f"- Avg recency: {stats['avg_recency']:.1f} days\n"
            f"- Avg frequency: {stats['avg_frequency']:.2f}\n"
            f"- Churn rate: {stats['churn_rate'] * 100:.1f}%\n"
            f"- Avg CLV: ${stats['avg_clv']:.2f}\n"
            f"Budget: ${budget:,}\n"
            f"Goal: {goal}"
        )
        system = (
            "You are a growth marketing strategist specializing in data-driven customer campaigns. Output "
            "structured, specific, actionable strategies. Be concise. Max 400 words. Use this exact format: "
            "\n\n"
            "\U0001F3AF Campaign Objective\n"
            "\U0001F4E3 Recommended Channels (with % budget split)\n"
            "\U0001F4E7 Message & Offer Ideas (3 specific examples)\n"
            "\U0001F4C5 Timeline & Frequency\n"
            "\U0001F4CA Success Metrics to Track\n"
            "\U0001F53D Risks to Avoid"
        )
        response = ask_groq(prompt, system, temperature=0.7)
        if response:
            _store_ai_response(cache_key, response)
            response_text = response
            cached_used = False

    if response_text:
        sections = _parse_strategy_sections(response_text)
        for title, body in sections:
            with st.expander(title, expanded=True):
                st.write(body)
        _render_cached_label(cached_used)

        st.download_button(
            "Download Strategy as .txt",
            data=response_text.encode("utf-8"),
            file_name="customerlens_strategy.txt",
            mime="text/plain",
        )


def tab_customer_chat(df: pd.DataFrame) -> None:
    """Render the Customer Chat tab."""
    header_col, action_col = st.columns([6, 1])
    with header_col:
        st.subheader("Chat with Your Data")
        st.caption("Ask anything about your customer segments")
    with action_col:
        if st.button("Clear Chat"):
            st.session_state.pop("ai_chat_history", None)

    ai_enabled = st.toggle(AI_TOGGLE_LABEL, value=True, key="ai_customer_chat")

    history = st.session_state.get("ai_chat_history", [])
    if not history:
        st.session_state["ai_chat_history"] = []
        history = st.session_state["ai_chat_history"]

    segment_list = sorted(df["rfm_segment"].unique().tolist())
    churn_rate = df["churn_flag"].mean() * 100
    avg_clv = df["clv_12m"].mean()
    top_cluster = (
        df.groupby("cluster_id")["total_spend"].sum().sort_values(ascending=False).index[0]
    )
    risky_segment = df.groupby("rfm_segment")["churn_flag"].mean().sort_values(ascending=False).index[0]

    system_prompt = (
        "You are a data analyst assistant for CustomerLens. "
        "Here is the current dataset summary:\n"
        f"- Total customers: {len(df)}\n"
        f"- Segments: {', '.join(segment_list)}\n"
        f"- Churn rate: {churn_rate:.1f}%\n"
        f"- Avg CLV: ${avg_clv:.2f}\n"
        f"- Top cluster by revenue: {top_cluster}\n"
        f"- Highest churn risk segment: {risky_segment}\n"
        "Answer questions about this customer data accurately. "
        "If asked something outside this data, say so clearly. Be concise. Max 400 words."
    )

    chips = [
        "Which segment should I focus on first?",
        "What's driving churn in our data?",
        "How do I increase CLV for at-risk customers?",
    ]
    chip_cols = st.columns(3)
    chip_clicked = None
    for col, chip in zip(chip_cols, chips):
        with col:
            if st.button(chip):
                chip_clicked = chip

    summary_key = _summary_fingerprint(df)

    def _send_message(text: str) -> None:
        history.append({"role": "user", "content": text})
        if not ai_enabled:
            history.append({"role": "assistant", "content": "AI features are disabled.", "cached": False})
            return
        cache_key = f"chat:{summary_key}:{text.strip().lower()}"
        cached = _get_ai_response(cache_key)
        if cached:
            history.append({"role": "assistant", "content": cached, "cached": True})
            return
        response = ask_groq(text, system_prompt, temperature=0.2)
        response_text = response or "No response."
        _store_ai_response(cache_key, response_text)
        history.append({"role": "assistant", "content": response_text, "cached": False})

    if chip_clicked:
        _send_message(chip_clicked)

    for message in history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("cached"):
                st.caption("(cached)")

    user_input = st.chat_input("Ask about your data")
    if user_input:
        _send_message(user_input)


def tab_exec_report(df: pd.DataFrame) -> None:
    """Render the Executive Report Writer tab."""
    st.subheader("Auto-Generate Executive Report")
    ai_enabled = st.toggle(AI_TOGGLE_LABEL, value=True, key="ai_exec_report")

    if not ai_enabled:
        st.info("AI features are disabled. Enable the toggle to generate reports.")
        return

    generate = st.button("Generate Full Report")
    if not generate:
        return

    summary_key = _summary_fingerprint(df)
    report_key = f"exec_report:{summary_key}"
    cached = st.session_state.get("ai_cache", {}).get(report_key)
    if cached:
        st.markdown(cached)
        _render_cached_label(True)
        st.download_button(
            "Download Report as .txt",
            data=cached.encode("utf-8"),
            file_name="customerlens_executive_report.txt",
            mime="text/plain",
        )
        st.code(cached, language="text")
        return

    summary_metrics = {
        "total_customers": len(df),
        "churn_rate": df["churn_flag"].mean() * 100,
        "avg_clv": df["clv_12m"].mean(),
        "top_segment": df["rfm_segment"].value_counts().idxmax(),
        "top_cluster": df.groupby("cluster_id")["total_spend"].sum().idxmax(),
    }

    progress = st.progress(0)

    with st.spinner("Generating executive summary..."):
        prompt = (
            "Create a 3-paragraph executive summary for a C-suite audience. Include the top 3 findings. "
            f"Metrics: total_customers={summary_metrics['total_customers']}, "
            f"churn_rate={summary_metrics['churn_rate']:.1f}%, "
            f"avg_clv=${summary_metrics['avg_clv']:.2f}, "
            f"top_segment={summary_metrics['top_segment']}, "
            f"top_cluster={summary_metrics['top_cluster']}."
        )
        system = "You are a senior analytics leader. Be concise. Max 400 words."
        executive_summary = ask_groq(prompt, system, temperature=0.3)
    progress.progress(33)

    with st.spinner("Identifying risks and opportunities..."):
        prompt = (
            "Based on churn rates, CLV distribution, and segment health, identify top 3 risks and top 3 growth "
            "opportunities with reasoning. Use bullet points."
        )
        system = "You are a senior analytics leader. Be concise. Max 400 words."
        risks = ask_groq(prompt, system, temperature=0.3)
    progress.progress(66)

    with st.spinner("Drafting recommended actions..."):
        prompt = (
            "Write a prioritized action plan with 5 specific steps for the next 30/60/90 days. Use headings for "
            "30 days, 60 days, 90 days."
        )
        system = "You are a senior analytics leader. Be concise. Max 400 words."
        actions = ask_groq(prompt, system, temperature=0.5)
    progress.progress(100)

    report = (
        "### Executive Summary\n"
        f"{executive_summary}\n\n"
        "---\n"
        "### Key Risks & Opportunities\n"
        f"{risks}\n\n"
        "---\n"
        "### Recommended Actions\n"
        f"{actions}"
    )

    st.markdown(report)
    st.session_state.setdefault("ai_cache", {})[report_key] = report

    st.download_button(
        "Download Report as .txt",
        data=report.encode("utf-8"),
        file_name="customerlens_executive_report.txt",
        mime="text/plain",
    )
    st.code(report, language="text")


def render_page() -> None:
    """Render the AI Analyst page."""
    configure_page()
    apply_theme()

    st.title("AI Analyst")
    st.caption("AI-powered insights, strategies, and executive reporting.")
    render_data_source_banner()

    full_df = prepare_data()
    filters = sidebar_filters(full_df, cluster_options=sorted(full_df["cluster_id"].unique().tolist()))
    filtered = apply_filters(full_df, filters)

    if filtered.empty:
        empty_state("No customers match the current filters. Adjust the sidebar to continue.")
        return

    render_kpi_row(build_kpis(filtered, full_df))

    tab_labels = ["Segment Explainer", "Strategy Generator", "Customer Chat", "Executive Report"]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        tab_segment_explainer(filtered)
    with tabs[1]:
        tab_strategy_generator(filtered)
    with tabs[2]:
        tab_customer_chat(filtered)
    with tabs[3]:
        tab_exec_report(filtered)


render_page()
