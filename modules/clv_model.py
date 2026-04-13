from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


def _assign_tier(series: pd.Series) -> pd.Series:
    """Assign CLV tiers based on quantiles."""
    quantiles = series.quantile([0.1, 0.3, 0.6]).values

    def label(value: float) -> str:
        if value >= quantiles[2]:
            return "Platinum"
        if value >= quantiles[1]:
            return "Gold"
        if value >= quantiles[0]:
            return "Silver"
        return "Bronze"

    return series.apply(label)


def train_clv_model(df: pd.DataFrame, random_state: int = 42) -> Dict[str, object]:
    """Train a CLV forecasting model and return predictions."""
    data = df.copy()
    data["historical_clv"] = data["avg_order_value"] * data["total_orders"]

    feature_cols = [
        "recency",
        "frequency",
        "monetary",
        "total_orders",
        "avg_order_value",
        "satisfaction_score",
        "loyalty_points",
        "days_since_last_purchase",
        "cluster_id",
    ]

    X = data[feature_cols].fillna(0)
    rng = np.random.default_rng(random_state)
    y = data["historical_clv"] * rng.normal(1.05, 0.12, size=len(data))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    data["clv_12m"] = model.predict(X)
    data["clv_tier"] = _assign_tier(data["clv_12m"])

    clv_by_cluster = (
        data.groupby(["cluster_id", "clv_tier"])["clv_12m"]
        .mean()
        .reset_index()
        .sort_values(["cluster_id", "clv_tier"])
    )

    return {
        "df": data,
        "model": model,
        "feature_cols": feature_cols,
        "clv_by_cluster": clv_by_cluster,
    }
