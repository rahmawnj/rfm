from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def evaluate_labels(
    scaled_features,
    labels,
    algorithm: str,
    runtime_seconds: float,
) -> dict:
    unique_labels = set(labels)
    if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
        raise ValueError(f"{algorithm} produced an invalid number of clusters for metric evaluation.")

    return {
        "algorithm": algorithm,
        "n_clusters": len(unique_labels),
        "silhouette_score": silhouette_score(scaled_features, labels),
        "davies_bouldin_index": davies_bouldin_score(scaled_features, labels),
        "calinski_harabasz_score": calinski_harabasz_score(scaled_features, labels),
        "runtime_seconds": runtime_seconds,
    }


def save_pca_plot(scaled_features, labels, algorithm: str, output_path: str | Path) -> None:
    projection = PCA(n_components=2, random_state=42).fit_transform(scaled_features)
    frame = pd.DataFrame({"pc1": projection[:, 0], "pc2": projection[:, 1], "cluster": labels})

    fig, ax = plt.subplots(figsize=(9, 6))
    for cluster_id, group in frame.groupby("cluster"):
        ax.scatter(group["pc1"], group["pc2"], s=12, alpha=0.65, label=f"Cluster {cluster_id}")
    ax.set_title(f"{algorithm} Customer Segments (PCA projection)")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend(title="Segment")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
