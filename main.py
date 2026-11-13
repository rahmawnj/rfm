import argparse
from pathlib import Path

import pandas as pd

from src.clustering import prepare_features, run_clustering
from src.data_loader import load_olist_data
from src.evaluation import evaluate_labels, save_pca_plot
from src.preprocessing import build_rfm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare K-Means and Hierarchical Clustering on Olist RFM data."
    )
    parser.add_argument("--data-dir", default="data/raw", help="Directory containing Olist CSV files.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for generated results.")
    parser.add_argument("--clusters", type=int, default=4, help="Number of clusters for both algorithms.")
    parser.add_argument(
        "--linkage",
        choices=["ward", "complete", "average", "single"],
        default="ward",
        help="Agglomerative clustering linkage method.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    orders, customers, payments = load_olist_data(args.data_dir)
    rfm = build_rfm(orders, customers, payments)
    rfm.to_csv(output_dir / "rfm_customers.csv", index=False)

    scaled_features, _ = prepare_features(rfm)
    kmeans_labels, hierarchical_labels, kmeans_time, hierarchical_time = run_clustering(
        scaled_features,
        n_clusters=args.clusters,
        linkage=args.linkage,
    )

    segments = rfm.copy()
    segments["kmeans_cluster"] = kmeans_labels
    segments["hierarchical_cluster"] = hierarchical_labels
    segments.to_csv(output_dir / "customer_segments.csv", index=False)

    metrics = pd.DataFrame(
        [
            evaluate_labels(scaled_features, kmeans_labels, "K-Means", kmeans_time),
            evaluate_labels(
                scaled_features,
                hierarchical_labels,
                f"Hierarchical ({args.linkage})",
                hierarchical_time,
            ),
        ]
    )
    metrics.to_csv(output_dir / "metrics.csv", index=False)

    save_pca_plot(scaled_features, kmeans_labels, "K-Means", output_dir / "kmeans_clusters.png")
    save_pca_plot(
        scaled_features,
        hierarchical_labels,
        f"Hierarchical ({args.linkage})",
        output_dir / "hierarchical_clusters.png",
    )

    print("\nRFM dataset created:", len(rfm), "customers")
    print("\nClustering evaluation:")
    print(metrics.to_string(index=False))
    print("\nResults saved to:", output_dir.resolve())


if __name__ == "__main__":
    main()
