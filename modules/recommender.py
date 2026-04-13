from __future__ import annotations

from typing import Dict, List
import pandas as pd

SEGMENT_PROFILES = {
    "Champions": {
        "channel": "email",
        "offer": "exclusive access",
        "tone": "premium",
        "response_rate": 0.32,
        "budget_pct": 16,
    },
    "Loyal Customers": {
        "channel": "email",
        "offer": "loyalty points",
        "tone": "nurturing",
        "response_rate": 0.27,
        "budget_pct": 14,
    },
    "Potential Loyalists": {
        "channel": "push",
        "offer": "loyalty points",
        "tone": "nurturing",
        "response_rate": 0.22,
        "budget_pct": 13,
    },
    "New Customers": {
        "channel": "email",
        "offer": "discount",
        "tone": "nurturing",
        "response_rate": 0.18,
        "budget_pct": 10,
    },
    "At Risk": {
        "channel": "sms",
        "offer": "discount",
        "tone": "urgent",
        "response_rate": 0.15,
        "budget_pct": 18,
    },
    "Can't Lose Them": {
        "channel": "email",
        "offer": "exclusive access",
        "tone": "urgent",
        "response_rate": 0.21,
        "budget_pct": 17,
    },
    "Hibernating": {
        "channel": "sms",
        "offer": "win-back",
        "tone": "reactivation",
        "response_rate": 0.1,
        "budget_pct": 8,
    },
    "Lost Customers": {
        "channel": "sms",
        "offer": "win-back",
        "tone": "reactivation",
        "response_rate": 0.06,
        "budget_pct": 4,
    },
}

def build_strategy_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create strategy recommendations by segment and cluster."""
    data = df.copy()
    base = (
        data.groupby(["rfm_segment", "cluster_id"])
        .agg(
            customer_count=("customer_id", "count"),
            avg_spend=("total_spend", "mean"),
            avg_recency=("recency", "mean"),
            avg_satisfaction=("satisfaction_score", "mean"),
        )
        .reset_index()
    )

    rows: List[Dict[str, object]] = []
    for _, row in base.iterrows():
        profile = SEGMENT_PROFILES.get(row["rfm_segment"], SEGMENT_PROFILES["New Customers"])
        response_rate = profile["response_rate"]

        strategy = {
            "rfm_segment": row["rfm_segment"],
            "cluster_id": int(row["cluster_id"]),
            "channel": profile["channel"],
            "offer": profile["offer"],
            "tone": profile["tone"],
            "budget_pct": profile["budget_pct"],
            "response_rate": response_rate,
            "customer_count": int(row["customer_count"]),
        }

        if row["avg_spend"] > data["total_spend"].median():
            strategy["offer"] = "exclusive access"
            strategy["tone"] = "premium"
            strategy["budget_pct"] += 2
        if row["avg_recency"] > data["recency"].median():
            strategy["channel"] = "sms"
            strategy["tone"] = "urgent"
        if row["avg_satisfaction"] < data["satisfaction_score"].median():
            strategy["tone"] = "nurturing"

        strategy["expected_response"] = strategy["response_rate"] * strategy["customer_count"]
        rows.append(strategy)

    return pd.DataFrame(rows)
