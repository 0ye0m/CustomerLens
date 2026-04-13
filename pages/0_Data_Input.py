from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from modules.data_manager import (
    STANDARD_COLUMNS,
    auto_detect_columns,
    build_demo_dataset,
    enrich_dataframe,
    render_data_source_banner,
    validate_dataframe,
)
from utils.styling import apply_theme

SKIP_OPTION = "-- skip this column --"


def configure_page() -> None:
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="CustomerLens | Your Data",
        page_icon=":open_file_folder:",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def _reset_mapping(columns: List[str]) -> None:
    """Reset mapping selections for the provided columns."""
    for col in columns:
        st.session_state[f"map_{col}"] = SKIP_OPTION


def _mapping_status(selected: str, detected: str | None) -> str:
    """Return status HTML for the mapping selection."""
    if selected == SKIP_OPTION:
        return "<span style='color: #9AA4B2;'>&#x2014; skipped</span>"
    if detected and selected == detected:
        return "<span style='color: #2ECC71;'>&#x2705; matched</span>"
    return "<span style='color: #F5B041;'>&#x2753; needs review</span>"


def _mapping_ui(df: pd.DataFrame) -> Dict[str, str]:
    """Render the column mapping UI and return the selected mapping."""
    detected = auto_detect_columns(df)

    st.markdown("#### Column Mapping")
    st.caption("Map your columns to the required CustomerLens fields.")

    header_cols = st.columns([3, 3, 2])
    header_cols[0].markdown("**Your column**")
    header_cols[1].markdown("**Maps to**")
    header_cols[2].markdown("**Status**")

    options = STANDARD_COLUMNS + [SKIP_OPTION]
    mapping: Dict[str, str] = {}

    for col in df.columns:
        row_cols = st.columns([3, 3, 2])
        row_cols[0].write(col)

        default_value = detected.get(col, SKIP_OPTION)
        if f"map_{col}" not in st.session_state:
            st.session_state[f"map_{col}"] = default_value

        selected = row_cols[1].selectbox(
            f"Map {col}",
            options=options,
            key=f"map_{col}",
            label_visibility="collapsed",
        )

        status_html = _mapping_status(selected, detected.get(col))
        row_cols[2].markdown(status_html, unsafe_allow_html=True)
        mapping[col] = selected

    if st.button("Reset mapping"):
        _reset_mapping(list(df.columns))
        st.experimental_rerun()

    return mapping


def _apply_mapping(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """Apply the selected mapping to the dataframe."""
    rename_map = {col: target for col, target in mapping.items() if target != SKIP_OPTION}
    return df.rename(columns=rename_map)


def _render_source_banner() -> None:
    """Render the active data source banner."""
    render_data_source_banner()


def tab_upload() -> None:
    """Upload CSV or Excel files."""
    st.subheader("Upload CSV or Excel")
    st.info(
        "Your file should have at minimum: a customer ID, a purchase date, number of orders, "
        "and total spend. Column names do not need to be exact; you can map them below."
    )

    uploaded = st.file_uploader(
        "Drag your file here or click to browse",
        type=["csv", "xlsx", "xls"],
    )

    if not uploaded:
        return

    with st.spinner("Reading your file..."):
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)

    if df.shape[1] == 0:
        st.error("The uploaded file has no columns. Please upload a valid dataset.")
        return

    if df.empty:
        st.error("The uploaded file has no rows. Please upload a file with data.")
        return

    if st.session_state.get("uploaded_file_name") != uploaded.name:
        st.session_state.pop("mapped_df", None)
        st.session_state.pop("mapping_validation", None)
        st.session_state.pop("upload_loaded", None)
        st.session_state["uploaded_file_name"] = uploaded.name

    st.session_state["uploaded_df"] = df

    mapping = _mapping_ui(df)

    if st.button("Confirm Mapping"):
        mapped_targets = [value for value in mapping.values() if value != SKIP_OPTION]
        if len(mapped_targets) != len(set(mapped_targets)):
            st.error("Each standard column can only be mapped once. Resolve duplicates and try again.")
            return

        mapped_df = _apply_mapping(df, mapping)
        validation = validate_dataframe(mapped_df)

        if not validation["is_valid"]:
            st.error(
                "Missing required columns: "
                + ", ".join(validation["missing_required"])
            )
            return

        st.session_state["mapped_df"] = mapped_df
        st.session_state["mapping_validation"] = validation

    mapped_df = st.session_state.get("mapped_df")
    validation = st.session_state.get("mapping_validation")

    if mapped_df is not None and validation is not None:
        if validation["warnings"]:
            st.warning(
                "Analysis will work but some features may be limited. "
                + " ".join(validation["warnings"])
            )
        if validation["type_issues"]:
            st.warning(
                "Columns with potential type issues: "
                + ", ".join(validation["type_issues"])
            )

        st.markdown("#### Data Preview")
        st.dataframe(mapped_df.head(10), use_container_width=True)

        nulls_found = sum(validation["null_summary"].values()) if validation["null_summary"] else 0
        st.caption(
            f"{validation['row_count']:,} rows, {mapped_df.shape[1]} columns, {nulls_found:,} nulls found"
        )

        if st.button("Load This Data", type="primary"):
            enriched = enrich_dataframe(mapped_df)
            st.session_state["user_data"] = enriched
            st.session_state["data_source"] = "uploaded"
            st.session_state["upload_loaded"] = True
            st.success(
                "Your data is loaded. All analysis pages are now using your data."
            )

        if st.session_state.get("upload_loaded"):
            if st.button("Go to Overview"):
                st.switch_page("pages/1_Overview.py")


def _init_manual_rows() -> None:
    """Initialize the manual entry rows in session state."""
    if "manual_rows" not in st.session_state:
        today = pd.Timestamp.today().normalize()
        st.session_state["manual_rows"] = [
            {
                "customer_id": "",
                "last_purchase_date": today,
                "total_orders": 1,
                "total_spend": 0.0,
            }
            for _ in range(3)
        ]


def tab_manual_entry() -> None:
    """Manual entry for small datasets."""
    st.subheader("Enter customer records manually")
    st.caption("Good for small datasets or testing")

    _init_manual_rows()
    rows = st.session_state["manual_rows"]

    for idx, row in enumerate(rows):
        cols = st.columns(4)
        row["customer_id"] = cols[0].text_input(
            "Customer ID",
            value=row.get("customer_id", ""),
            key=f"manual_customer_id_{idx}",
        )
        row["last_purchase_date"] = cols[1].date_input(
            "Last purchase date",
            value=row.get("last_purchase_date"),
            key=f"manual_purchase_date_{idx}",
        )
        row["total_orders"] = cols[2].number_input(
            "Total orders",
            min_value=1,
            value=int(row.get("total_orders", 1)),
            key=f"manual_total_orders_{idx}",
        )
        row["total_spend"] = cols[3].number_input(
            "Total spend",
            min_value=0.0,
            value=float(row.get("total_spend", 0.0)),
            format="%.2f",
            key=f"manual_total_spend_{idx}",
        )

    st.session_state["manual_rows"] = rows

    actions = st.columns(2)
    if actions[0].button("+ Add another customer"):
        rows.append(
            {
                "customer_id": "",
                "last_purchase_date": pd.Timestamp.today().normalize(),
                "total_orders": 1,
                "total_spend": 0.0,
            }
        )
        st.session_state["manual_rows"] = rows
        st.experimental_rerun()

    if actions[1].button("Remove last row"):
        if len(rows) > 1:
            rows.pop()
            st.session_state["manual_rows"] = rows
            st.experimental_rerun()

    manual_df = pd.DataFrame(rows)
    st.markdown("#### Preview")
    st.dataframe(manual_df, use_container_width=True)

    if st.button("Load These Customers", type="primary"):
        trimmed = manual_df[manual_df["customer_id"].astype(str).str.len() > 0]
        if len(trimmed) < 3:
            st.error("Please enter at least 3 complete customer rows before loading.")
            return

        validation = validate_dataframe(trimmed)
        if not validation["is_valid"]:
            st.error(
                "Missing required columns: "
                + ", ".join(validation["missing_required"])
            )
            return

        enriched = enrich_dataframe(trimmed)
        st.session_state["user_data"] = enriched
        st.session_state["data_source"] = "uploaded"
        st.session_state["manual_loaded"] = True
        st.success("Your customers are loaded. All analysis pages are now using your data.")

    if st.session_state.get("manual_loaded"):
        if st.button("Go to Overview"):
            st.switch_page("pages/1_Overview.py")

    st.download_button(
        "Download as CSV",
        data=manual_df.to_csv(index=False).encode("utf-8"),
        file_name="customerlens_manual_data.csv",
        mime="text/csv",
    )


def _demo_card(title: str, subtitle: str, button_label: str) -> None:
    """Render a demo dataset card with a load button."""
    st.markdown(
        f"""
        <div class="card">
            <strong>{title}</strong><br/>
            <span style="color: var(--muted);">{subtitle}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    key_suffix = "".join(ch.lower() for ch in title if ch.isalnum())
    if st.button(button_label, type="primary", key=f"load_demo_{key_suffix}"):
        if "E-commerce" in title:
            demo_df = build_demo_dataset("ecommerce")
            st.session_state["data_source"] = "demo_ecommerce"
        elif "SaaS" in title:
            demo_df = build_demo_dataset("saas")
            st.session_state["data_source"] = "demo_saas"
        else:
            demo_df = build_demo_dataset("retail")
            st.session_state["data_source"] = "demo_retail"

        st.session_state["user_data"] = demo_df
        st.session_state["demo_loaded"] = True
        st.success("Demo data loaded. You can start exploring now.")


def tab_demo_data() -> None:
    """Offer demo datasets."""
    st.subheader("Explore with demo data")
    st.caption("No file needed. Great for testing the app.")

    col1, col2, col3 = st.columns(3)
    with col1:
        _demo_card(
            "E-commerce store",
            "5,000 customers, online retail, international. High churn, 3 product categories.",
            "Load E-commerce Demo",
        )
    with col2:
        _demo_card(
            "SaaS company",
            "2,000 customers, subscription model, trial users, feature adoption focus.",
            "Load SaaS Demo",
        )
    with col3:
        _demo_card(
            "Retail chain",
            "3,000 customers, in-store + online, loyalty program, regional spread.",
            "Load Retail Demo",
        )

    if st.session_state.get("demo_loaded"):
        if st.button("Start Exploring", key="start_demo_explore"):
            st.switch_page("pages/1_Overview.py")


def render_page() -> None:
    """Render the data input page."""
    configure_page()
    apply_theme()

    st.title("Your Data")
    st.caption("Upload your customer data or use demo data")
    _render_source_banner()

    tabs = st.tabs(["Upload File", "Manual Entry", "Use Demo Data"])
    with tabs[0]:
        tab_upload()
    with tabs[1]:
        tab_manual_entry()
    with tabs[2]:
        tab_demo_data()


render_page()
