from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler


RFM_COLUMNS = ["recency", "frequency", "monetary"]


def prepare_features(rfm: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Apply log1p and standard scaling to the same RFM features for both models."""
    features = rfm[RFM_COLUMNS].astype(float).copy()
    if (features < 0).any().any():
        raise ValueError("RFM features must be non-negative before log1p transformation.")

    transformed = np.log1p(features)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(transformed)
    return scaled, scaler


def run_clustering(
    scaled_features: np.ndarray,
    n_clusters: int = 4,
    linkage: str = "ward",
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Fit both algorithms on identical preprocessed data and return labels and runtimes."""
    if n_clusters < 2 or n_clusters >= len(scaled_features):
        raise ValueError("n_clusters must be at least 2 and smaller than the number of customers.")

    start = perf_counter()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    kmeans_labels = kmeans.fit_predict(scaled_features)
    kmeans_seconds = perf_counter() - start

    start = perf_counter()
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    hierarchical_labels = hierarchical.fit_predict(scaled_features)
    hierarchical_seconds = perf_counter() - start

    return kmeans_labels, hierarchical_labels, kmeans_seconds, hierarchical_seconds
