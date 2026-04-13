from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def _score_quantiles(series: pd.Series, reverse: bool = False) -> pd.Series:
    """Score a series into 1-5 quantile bins."""
    ranked = series.rank(method="first")
    try:
        scores = pd.qcut(ranked, 5, labels=[1, 2, 3, 4, 5])
    except ValueError:
        scores = pd.cut(ranked, bins=5, labels=[1, 2, 3, 4, 5])

    scores = scores.astype(int)
    if reverse:
        scores = 6 - scores
    return scores


def _segment_from_scores(r: int, f: int, m: int) -> str:
    """Map RFM scores to a named segment."""
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    if r >= 3 and f >= 4 and m >= 3:
        return "Loyal Customers"
    if r >= 4 and f >= 2 and m >= 2:
        return "Potential Loyalists"
    if r >= 4 and f <= 2 and m <= 2:
        return "New Customers"
    if r <= 2 and f >= 4 and m >= 4:
        return "Can't Lose Them"
    if r <= 2 and f >= 3 and m >= 3:
        return "At Risk"
    if r <= 2 and f <= 2 and m <= 2:
        return "Hibernating"
    return "Lost Customers"


def calculate_rfm(df: pd.DataFrame, today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Calculate RFM scores and segment labels.

    Args:
        df: Input customer dataframe.
        today: Optional reference date for recency calculation.

    Returns:
        Enriched dataframe with RFM scores and segments.
    """
    data = df.copy()
    reference_date = today or pd.Timestamp.today().normalize()

    data["recency"] = (reference_date - pd.to_datetime(data["last_purchase_date"]).fillna(reference_date)).dt.days
    data["frequency"] = data["total_orders"].fillna(0)
    data["monetary"] = data["total_spend"].fillna(0.0)

    data["r_score"] = _score_quantiles(data["recency"], reverse=True)
    data["f_score"] = _score_quantiles(data["frequency"], reverse=False)
    data["m_score"] = _score_quantiles(data["monetary"], reverse=False)

    data["rfm_score"] = data[["r_score", "f_score", "m_score"]].sum(axis=1)
    data["rfm_segment"] = [
        _segment_from_scores(r, f, m) for r, f, m in zip(data["r_score"], data["f_score"], data["m_score"])
    ]
    return data
