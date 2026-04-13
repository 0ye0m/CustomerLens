from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _select_features(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Select numeric and categorical features for clustering."""
    numeric_features = [
        "age",
        "total_orders",
        "total_spend",
        "avg_order_value",
        "loyalty_points",
        "support_tickets_raised",
        "satisfaction_score",
        "referral_count",
        "days_since_last_purchase",
    ]
    for col in ["recency", "frequency", "monetary", "rfm_score"]:
        if col in df.columns:
            numeric_features.append(col)

    categorical_features = [
        "gender",
        "country",
        "city",
        "product_category_preference",
        "channel",
        "payment_method",
        "discount_sensitivity",
        "rfm_segment",
    ]
    categorical_features = [col for col in categorical_features if col in df.columns]

    return numeric_features, categorical_features


def prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str], ColumnTransformer]:
    """Prepare the feature matrix for clustering models."""
    numeric_features, categorical_features = _select_features(df)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="drop",
    )

    X = preprocessor.fit_transform(df)

    feature_names = numeric_features
    if categorical_features:
        cat_features = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features)
        feature_names = numeric_features + list(cat_features)

    return X, feature_names, preprocessor


def _cluster_metrics(X: np.ndarray, labels: np.ndarray) -> Tuple[float | None, float | None, float | None]:
    """Safely compute clustering metrics."""
    unique_labels = set(labels)
    if -1 in unique_labels:
        mask = labels != -1
    else:
        mask = np.ones(len(labels), dtype=bool)

    filtered_labels = labels[mask]
    filtered_X = X[mask]
    if len(set(filtered_labels)) < 2:
        return None, None, None

    return (
        float(silhouette_score(filtered_X, filtered_labels)),
        float(davies_bouldin_score(filtered_X, filtered_labels)),
        float(calinski_harabasz_score(filtered_X, filtered_labels)),
    )


def _normalize(values: List[float | None], higher_better: bool = True) -> List[float]:
    """Normalize a list of metrics to 0-1 range."""
    arr = np.array([np.nan if v is None else v for v in values], dtype=float)
    if np.all(np.isnan(arr)):
        return [0.0 for _ in values]

    min_val = np.nanmin(arr)
    max_val = np.nanmax(arr)
    if min_val == max_val:
        norm = np.ones_like(arr) * 0.5
    else:
        norm = (arr - min_val) / (max_val - min_val)

    if not higher_better:
        norm = 1 - norm

    return [float(v) if not np.isnan(v) else 0.0 for v in norm]


def _run_kmeans(X: np.ndarray, k_range: range, random_state: int) -> Dict[str, Any]:
    """Run KMeans across a range of k values."""
    results = []
    for k in k_range:
        model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = model.fit_predict(X)
        silhouette, dbi, ch = _cluster_metrics(X, labels)
        results.append(
            {
                "k": k,
                "model": model,
                "labels": labels,
                "inertia": model.inertia_,
                "silhouette": silhouette,
                "dbi": dbi,
                "ch": ch,
            }
        )

    inertia_norm = _normalize([r["inertia"] for r in results], higher_better=False)
    silhouette_norm = _normalize([r["silhouette"] for r in results], higher_better=True)
    ch_norm = _normalize([r["ch"] for r in results], higher_better=True)
    dbi_norm = _normalize([r["dbi"] for r in results], higher_better=False)

    for idx, result in enumerate(results):
        result["score"] = 0.45 * silhouette_norm[idx] + 0.2 * ch_norm[idx] + 0.2 * dbi_norm[idx] + 0.15 * inertia_norm[idx]

    best = max(results, key=lambda r: r["score"])
    return {
        "labels": best["labels"],
        "model": best["model"],
        "metrics": best,
        "history": results,
    }


def _run_dbscan(X: np.ndarray, min_samples: int) -> Dict[str, Any]:
    """Auto-tune DBSCAN epsilon via k-distance percentiles."""
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors.fit(X)
    distances, _ = neighbors.kneighbors(X)
    k_distances = np.sort(distances[:, -1])

    eps_candidates = np.percentile(k_distances, [70, 75, 80, 85, 90, 92, 95])
    eps_candidates = np.unique(np.round(eps_candidates, 4))

    best_result = None
    results = []
    for eps in eps_candidates:
        model = DBSCAN(eps=float(eps), min_samples=min_samples)
        labels = model.fit_predict(X)
        silhouette, dbi, ch = _cluster_metrics(X, labels)
        result = {
            "eps": float(eps),
            "labels": labels,
            "model": model,
            "silhouette": silhouette,
            "dbi": dbi,
            "ch": ch,
        }
        results.append(result)
        if best_result is None or (result["silhouette"] or -1) > (best_result["silhouette"] or -1):
            best_result = result

    return {
        "labels": best_result["labels"],
        "model": best_result["model"],
        "metrics": best_result,
        "history": results,
    }


def _run_agglomerative(X: np.ndarray, n_range: range) -> Dict[str, Any]:
    """Run Agglomerative clustering with Ward linkage."""
    results = []
    for n_clusters in n_range:
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        labels = model.fit_predict(X)
        silhouette, dbi, ch = _cluster_metrics(X, labels)
        results.append(
            {
                "n_clusters": n_clusters,
                "labels": labels,
                "model": model,
                "silhouette": silhouette,
                "dbi": dbi,
                "ch": ch,
            }
        )

    best = max(results, key=lambda r: r["silhouette"] or -1)
    return {
        "labels": best["labels"],
        "model": best["model"],
        "metrics": best,
        "history": results,
    }


def _count_clusters(labels: np.ndarray) -> int:
    """Count clusters excluding noise if present."""
    unique = set(labels)
    unique.discard(-1)
    return len(unique)


def run_clustering(df: pd.DataFrame, random_state: int = 42) -> Dict[str, Any]:
    """Run multiple clustering algorithms and compare results."""
    X, feature_names, preprocessor = prepare_features(df)

    kmeans_result = _run_kmeans(X, range(2, 11), random_state)
    dbscan_result = _run_dbscan(X, min_samples=5)
    agglom_result = _run_agglomerative(X, range(2, 11))

    comparison = pd.DataFrame(
        [
            {
                "algorithm": "K-Means",
                "n_clusters": _count_clusters(kmeans_result["labels"]),
                "silhouette": kmeans_result["metrics"]["silhouette"],
                "davies_bouldin": kmeans_result["metrics"]["dbi"],
                "calinski_harabasz": kmeans_result["metrics"]["ch"],
            },
            {
                "algorithm": "DBSCAN",
                "n_clusters": _count_clusters(dbscan_result["labels"]),
                "silhouette": dbscan_result["metrics"]["silhouette"],
                "davies_bouldin": dbscan_result["metrics"]["dbi"],
                "calinski_harabasz": dbscan_result["metrics"]["ch"],
            },
            {
                "algorithm": "Hierarchical",
                "n_clusters": _count_clusters(agglom_result["labels"]),
                "silhouette": agglom_result["metrics"]["silhouette"],
                "davies_bouldin": agglom_result["metrics"]["dbi"],
                "calinski_harabasz": agglom_result["metrics"]["ch"],
            },
        ]
    )

    silhouette_norm = _normalize(comparison["silhouette"].tolist(), higher_better=True)
    dbi_norm = _normalize(comparison["davies_bouldin"].tolist(), higher_better=False)
    ch_norm = _normalize(comparison["calinski_harabasz"].tolist(), higher_better=True)

    comparison["score"] = (
        np.array(silhouette_norm) * 0.5 + np.array(dbi_norm) * 0.25 + np.array(ch_norm) * 0.25
    )
    best_idx = int(comparison["score"].idxmax())
    best_algorithm = comparison.loc[best_idx, "algorithm"]

    if best_algorithm == "DBSCAN":
        best_labels = dbscan_result["labels"]
    elif best_algorithm == "Hierarchical":
        best_labels = agglom_result["labels"]
    else:
        best_labels = kmeans_result["labels"]

    data = df.copy()
    data["kmeans_label"] = kmeans_result["labels"]
    data["dbscan_label"] = dbscan_result["labels"]
    data["hierarchical_label"] = agglom_result["labels"]
    data["cluster_id"] = best_labels

    numeric_features, _ = _select_features(df)
    cluster_stats = (
        data.groupby("cluster_id")[numeric_features]
        .mean()
        .reset_index()
        .merge(data["cluster_id"].value_counts().rename("cluster_size"), on="cluster_id")
        .sort_values("cluster_id")
    )

    return {
        "df": data,
        "comparison": comparison,
        "cluster_stats": cluster_stats,
        "features": {"matrix": X, "names": feature_names, "preprocessor": preprocessor},
        "labels": {
            "K-Means": kmeans_result["labels"],
            "DBSCAN": dbscan_result["labels"],
            "Hierarchical": agglom_result["labels"],
        },
        "best_algorithm": best_algorithm,
    }
