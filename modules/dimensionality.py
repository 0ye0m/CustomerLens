from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def run_dimensionality(feature_matrix: np.ndarray, random_state: int = 42) -> Dict[str, np.ndarray]:
    """Compute PCA and t-SNE embeddings for visualization.

    Args:
        feature_matrix: Preprocessed feature matrix.
        random_state: Random state for reproducibility.

    Returns:
        Dictionary of embeddings and explained variance ratios.
    """
    pca_2d = PCA(n_components=2, random_state=random_state)
    pca_3d = PCA(n_components=3, random_state=random_state)

    pca_2d_coords = pca_2d.fit_transform(feature_matrix)
    pca_3d_coords = pca_3d.fit_transform(feature_matrix)

    perplexity = min(40, max(5, feature_matrix.shape[0] // 50))
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity, init="pca", learning_rate="auto")
    tsne_coords = tsne.fit_transform(feature_matrix)

    return {
        "pca_2d": pca_2d_coords,
        "pca_3d": pca_3d_coords,
        "pca_2d_var": pca_2d.explained_variance_ratio_,
        "pca_3d_var": pca_3d.explained_variance_ratio_,
        "tsne_2d": tsne_coords,
    }
