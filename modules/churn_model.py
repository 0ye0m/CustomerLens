from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_churn_model(df: pd.DataFrame, random_state: int = 42) -> Dict[str, object]:
    """Train a churn prediction model and return scores and metadata."""
    data = df.copy()
    feature_cols = [
        "recency",
        "frequency",
        "monetary",
        "satisfaction_score",
        "support_tickets_raised",
        "days_since_last_purchase",
        "cluster_id",
    ]

    X = data[feature_cols].fillna(0)
    y = data["churn_flag"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
    )
    model.fit(X_train_scaled, y_train)

    proba = model.predict_proba(scaler.transform(X))[:, 1]
    data["churn_probability"] = proba

    y_pred = model.predict(X_test_scaled)
    y_score = model.predict_proba(X_test_scaled)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_score)
    fpr, tpr, thresholds = roc_curve(y_test, y_score)

    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    top_risk = data.sort_values("churn_probability", ascending=False).head(100)

    return {
        "df": data,
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "report": report,
        "roc_auc": roc_auc,
        "roc_curve": {"fpr": fpr, "tpr": tpr, "thresholds": thresholds},
        "feature_importance": importance,
        "top_risk": top_risk,
    }
