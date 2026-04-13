from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from data.generate_data import generate_dataset

REQUIRED_COLUMNS = [
    "customer_id",
    "last_purchase_date",
    "total_orders",
    "total_spend",
    "days_since_last_purchase",
]

OPTIONAL_COLUMNS = [
    "age",
    "gender",
    "country",
    "city",
    "satisfaction_score",
    "churn_flag",
    "avg_order_value",
    "channel",
    "loyalty_points",
    "support_tickets_raised",
]

STANDARD_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

DEFAULT_VALUES = {
    "age": 35,
    "gender": "Unknown",
    "country": "Unknown",
    "city": "Unknown",
    "satisfaction_score": 7.0,
    "churn_flag": 0,
    "avg_order_value": 0.0,
    "channel": "web",
    "loyalty_points": 0,
    "support_tickets_raised": 0,
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "customers.csv"


def _normalize_column_name(name: str) -> str:
    """Normalize column name for matching."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _parse_dates(series: pd.Series) -> pd.Series:
    """Parse dates with multiple format fallbacks."""
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=False)
    if parsed.isna().mean() > 0.3:
        parsed_alt = pd.to_datetime(series, errors="coerce", dayfirst=True)
        parsed = parsed.fillna(parsed_alt)
    return parsed


def _clean_currency(series: pd.Series) -> pd.Series:
    """Strip currency symbols and parse numeric values."""
    cleaned = series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def auto_detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Smart fuzzy column name matcher."""
    synonyms = {
        "last_purchase_date": [
            "lastpurchase",
            "lastorder",
            "purchase_date",
            "orderdate",
            "last_order_date",
            "last_purchase",
            "last_order",
            "purchase",
            "date",
        ],
        "total_orders": [
            "totalorders",
            "ordercount",
            "numorders",
            "orders",
            "purchases",
            "transactions",
        ],
        "total_spend": [
            "totalspend",
            "totalsales",
            "revenue",
            "amount",
            "total",
            "spend",
            "sales",
        ],
        "customer_id": [
            "customerid",
            "custid",
            "userid",
            "clientid",
            "customer",
            "user",
            "id",
        ],
        "days_since_last_purchase": [
            "dayssincelastpurchase",
            "dayssincepurchase",
            "dayssince",
            "recency",
            "dayslastpurchase",
        ],
        "avg_order_value": [
            "avgordervalue",
            "averageordervalue",
            "aov",
            "avgspend",
        ],
        "satisfaction_score": [
            "satisfactionscore",
            "satisfaction",
            "csat",
            "rating",
            "score",
        ],
        "churn_flag": [
            "churnflag",
            "churn",
            "churned",
            "attrition",
        ],
        "loyalty_points": [
            "loyaltypoints",
            "rewardpoints",
            "points",
        ],
        "support_tickets_raised": [
            "supporttickets",
            "tickets",
            "cases",
        ],
        "channel": [
            "channel",
            "platform",
            "source",
        ],
        "country": ["country", "nation"],
        "city": ["city", "town"],
        "gender": ["gender", "sex"],
        "age": ["age", "years"],
    }

    mapping: Dict[str, str] = {}
    for col in df.columns:
        normalized = _normalize_column_name(col)
        best_match = ""
        best_score = 0
        for standard, keys in synonyms.items():
            for key in keys:
                if key and key in normalized and len(key) > best_score:
                    best_match = standard
                    best_score = len(key)
        if best_match:
            mapping[col] = best_match

    return mapping


def validate_dataframe(df: pd.DataFrame) -> Dict[str, object]:
    """Validate required and optional columns with type checks."""
    missing_required = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    warnings: List[str] = []
    if "days_since_last_purchase" in missing_required and "last_purchase_date" in df.columns:
        missing_required.remove("days_since_last_purchase")
        warnings.append("days_since_last_purchase will be computed from last_purchase_date")

    for col in REQUIRED_COLUMNS:
        if col in df.columns and df[col].isna().all():
            if col not in missing_required:
                missing_required.append(col)

    missing_optional = [col for col in OPTIONAL_COLUMNS if col not in df.columns]
    warnings.extend([f"Missing optional column: {col}" for col in missing_optional])

    for col in OPTIONAL_COLUMNS:
        if col in df.columns and df[col].isna().all():
            warnings.append(f"Optional column has all nulls: {col}")

    row_count = int(len(df))
    if row_count == 0:
        warnings.append("Dataset is empty")
    elif row_count < 3:
        warnings.append("Dataset has fewer than 3 rows")

    null_summary = {
        col: int(df[col].isna().sum())
        for col in df.columns
        if col in STANDARD_COLUMNS
    }

    type_issues: List[str] = []
    if "last_purchase_date" in df.columns:
        parsed = _parse_dates(df["last_purchase_date"])
        if parsed.isna().mean() > 0.4:
            type_issues.append("last_purchase_date")

    for col in ["total_orders", "days_since_last_purchase", "age", "loyalty_points", "support_tickets_raised"]:
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.isna().mean() > 0.4:
                type_issues.append(col)

    for col in ["total_spend", "avg_order_value"]:
        if col in df.columns:
            numeric = _clean_currency(df[col])
            if numeric.isna().mean() > 0.4:
                type_issues.append(col)

    for col in ["churn_flag", "satisfaction_score"]:
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.isna().mean() > 0.6:
                type_issues.append(col)

    is_valid = len(missing_required) == 0 and row_count > 0

    return {
        "is_valid": is_valid,
        "missing_required": missing_required,
        "warnings": warnings,
        "row_count": row_count,
        "null_summary": null_summary,
        "type_issues": type_issues,
    }


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived columns and fill missing values."""
    data = df.copy()

    if "customer_id" in data.columns:
        data["customer_id"] = data["customer_id"].astype(str)
        data = data.drop_duplicates(subset="customer_id", keep="last")

    if "total_spend" in data.columns:
        data["total_spend"] = _clean_currency(data["total_spend"])

    if "total_orders" in data.columns:
        data["total_orders"] = pd.to_numeric(data["total_orders"], errors="coerce")

    if "last_purchase_date" in data.columns:
        data["last_purchase_date"] = _parse_dates(data["last_purchase_date"])

    if "days_since_last_purchase" not in data.columns:
        data["days_since_last_purchase"] = np.nan

    today = pd.Timestamp.today().normalize()
    if "last_purchase_date" in data.columns:
        derived_days = (today - data["last_purchase_date"]).dt.days
        data["days_since_last_purchase"] = data["days_since_last_purchase"].fillna(derived_days)

    if "avg_order_value" not in data.columns:
        data["avg_order_value"] = np.nan

    orders = data["total_orders"].replace(0, np.nan)
    data["avg_order_value"] = data["avg_order_value"].fillna(data["total_spend"] / orders)

    if "churn_flag" not in data.columns:
        data["churn_flag"] = (data["days_since_last_purchase"] > 180).astype(int)

    for col, default in DEFAULT_VALUES.items():
        if col not in data.columns:
            data[col] = default

    if "signup_date" not in data.columns and "last_purchase_date" in data.columns:
        data["signup_date"] = data["last_purchase_date"]

    numeric_cols = data.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        median = data[col].median()
        if np.isnan(median):
            median = 0
        data[col] = data[col].fillna(median)

    categorical_cols = [
        col
        for col in data.columns
        if data[col].dtype == "object" or data[col].dtype.name.startswith("string")
    ]
    for col in categorical_cols:
        data[col] = data[col].fillna("Unknown")

    return data


def _load_synthetic_data() -> pd.DataFrame:
    """Load or generate the synthetic dataset."""
    if not DATA_PATH.exists():
        generate_dataset(DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    df["signup_date"] = pd.to_datetime(df.get("signup_date"), errors="coerce")
    df["last_purchase_date"] = pd.to_datetime(df.get("last_purchase_date"), errors="coerce")
    return df


def render_data_source_banner() -> None:
    """Show a banner describing the active data source."""
    source = st.session_state.get("data_source", "synthetic")
    st.session_state["data_banner_rendered"] = True
    if source == "synthetic":
        st.info(
            "You are viewing synthetic demo data. Upload your own data on the **Your Data** page for real insights."
        )
    elif source.startswith("demo"):
        label = {
            "demo_ecommerce": "E-commerce",
            "demo_saas": "SaaS",
            "demo_retail": "Retail",
        }.get(source, source.replace("demo_", "").replace("_", " ").title())
        st.info(f"Using demo dataset: {label}")
    else:
        st.success("Showing analysis on your uploaded data.")


def render_data_source_widget() -> None:
    """Render the sidebar data status widget."""
    st.sidebar.markdown("### Data source")
    user_data = st.session_state.get("user_data")
    source = st.session_state.get("data_source", "synthetic")

    if user_data is not None:
        count = len(user_data)
        label = "Uploaded" if source == "uploaded" else {
            "demo_ecommerce": "Demo: E-commerce",
            "demo_saas": "Demo: SaaS",
            "demo_retail": "Demo: Retail",
        }.get(source, source.replace("demo_", "Demo: ").title())
        st.sidebar.markdown("<span style='color: #2ecc71;'>Active</span>", unsafe_allow_html=True)
        st.sidebar.caption(f"{count:,} customers loaded")
        st.sidebar.caption(label)
        st.sidebar.page_link("pages/0_Data_Input.py", label="Change data")
    else:
        st.sidebar.markdown("<span style='color: #9AA4B2;'>Demo data (synthetic)</span>", unsafe_allow_html=True)
        st.sidebar.caption("5,000 synthetic customers")
        st.sidebar.page_link("pages/0_Data_Input.py", label="Upload your data")


def get_active_dataset() -> pd.DataFrame:
    """Return the active dataset, falling back to synthetic data."""
    if not st.session_state.get("data_banner_rendered"):
        render_data_source_banner()

    if "user_data" in st.session_state:
        st.session_state.setdefault("data_source", "uploaded")
        return st.session_state["user_data"].copy()

    st.session_state.setdefault("data_source", "synthetic")
    df = _load_synthetic_data()
    return enrich_dataframe(df)


def _base_demo_frame(n_customers: int, seed: int) -> pd.DataFrame:
    """Create a base demo dataframe with customer ids and purchase dates."""
    rng = np.random.default_rng(seed)
    customer_id = [f"USR-{i:05d}" for i in range(1, n_customers + 1)]
    last_purchase_date = pd.Timestamp.today().normalize() - pd.to_timedelta(
        rng.integers(1, 365, size=n_customers), unit="D"
    )
    return pd.DataFrame({"customer_id": customer_id, "last_purchase_date": last_purchase_date})


def _demo_ecommerce() -> pd.DataFrame:
    """Generate an e-commerce demo dataset."""
    rng = np.random.default_rng(12)
    n_customers = 5000
    df = _base_demo_frame(n_customers, seed=12)
    df["total_orders"] = rng.poisson(lam=3.5, size=n_customers).clip(1, None)
    df["total_spend"] = rng.lognormal(mean=3.2, sigma=0.6, size=n_customers) * 120
    df["country"] = rng.choice(["United States", "Canada", "United Kingdom"], size=n_customers)
    df["city"] = rng.choice(["New York", "Toronto", "London", "Austin", "Vancouver"], size=n_customers)
    df["channel"] = rng.choice(["web", "mobile"], size=n_customers, p=[0.6, 0.4])
    df["satisfaction_score"] = np.clip(rng.normal(6.8, 1.6, size=n_customers), 1, 10)
    df["churn_flag"] = (rng.random(n_customers) < 0.25).astype(int)
    df["loyalty_points"] = (df["total_spend"] * 0.05 + df["total_orders"] * 8).astype(int)
    df["support_tickets_raised"] = rng.poisson(0.8, size=n_customers)
    df["avg_order_value"] = df["total_spend"] / df["total_orders"]
    df["signup_date"] = df["last_purchase_date"] - pd.to_timedelta(
        rng.integers(30, 900, size=n_customers), unit="D"
    )
    return df


def _demo_saas() -> pd.DataFrame:
    """Generate a SaaS demo dataset."""
    rng = np.random.default_rng(24)
    n_customers = 2000
    df = _base_demo_frame(n_customers, seed=24)
    df["total_orders"] = rng.poisson(lam=6.0, size=n_customers).clip(1, None)
    df["total_spend"] = rng.lognormal(mean=2.6, sigma=0.4, size=n_customers) * 300
    df["country"] = rng.choice(["United States", "Germany", "India"], size=n_customers)
    df["city"] = rng.choice(["San Francisco", "Berlin", "Bengaluru", "Austin"], size=n_customers)
    df["channel"] = rng.choice(["web", "mobile"], size=n_customers, p=[0.8, 0.2])
    df["satisfaction_score"] = np.clip(rng.normal(7.6, 1.1, size=n_customers), 1, 10)
    df["churn_flag"] = (rng.random(n_customers) < 0.18).astype(int)
    df["loyalty_points"] = (df["total_spend"] * 0.03 + df["total_orders"] * 5).astype(int)
    df["support_tickets_raised"] = rng.poisson(0.4, size=n_customers)
    df["avg_order_value"] = df["total_spend"] / df["total_orders"]
    df["signup_date"] = df["last_purchase_date"] - pd.to_timedelta(
        rng.integers(60, 1200, size=n_customers), unit="D"
    )
    return df


def _demo_retail() -> pd.DataFrame:
    """Generate a retail demo dataset."""
    rng = np.random.default_rng(36)
    n_customers = 3000
    df = _base_demo_frame(n_customers, seed=36)
    df["total_orders"] = rng.poisson(lam=4.2, size=n_customers).clip(1, None)
    df["total_spend"] = rng.lognormal(mean=3.0, sigma=0.5, size=n_customers) * 90
    df["country"] = rng.choice(["United States", "Canada"], size=n_customers)
    df["city"] = rng.choice(["Chicago", "Seattle", "Montreal", "Calgary"], size=n_customers)
    df["channel"] = rng.choice(["store", "web", "mobile"], size=n_customers, p=[0.5, 0.3, 0.2])
    df["satisfaction_score"] = np.clip(rng.normal(7.0, 1.4, size=n_customers), 1, 10)
    df["churn_flag"] = (rng.random(n_customers) < 0.2).astype(int)
    df["loyalty_points"] = (df["total_spend"] * 0.08 + df["total_orders"] * 10).astype(int)
    df["support_tickets_raised"] = rng.poisson(0.6, size=n_customers)
    df["avg_order_value"] = df["total_spend"] / df["total_orders"]
    df["signup_date"] = df["last_purchase_date"] - pd.to_timedelta(
        rng.integers(45, 1000, size=n_customers), unit="D"
    )
    return df


def build_demo_dataset(kind: str) -> pd.DataFrame:
    """Create a demo dataset based on the selected scenario."""
    kind_lower = kind.lower()
    if kind_lower == "ecommerce":
        df = _demo_ecommerce()
    elif kind_lower == "saas":
        df = _demo_saas()
    elif kind_lower == "retail":
        df = _demo_retail()
    else:
        df = _demo_ecommerce()
    return enrich_dataframe(df)
