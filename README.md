# Customer Segmentation with RFM: K-Means vs Hierarchical Clustering

A reproducible Python project for comparing K-Means and Agglomerative (Hierarchical) Clustering on the Brazilian E-Commerce Public Dataset by Olist.

## Research objective

Compare both algorithms using the same customer-level RFM features and preprocessing pipeline. The initial experiment uses a configurable number of clusters (default: 4) so the comparison is controlled and easy to reproduce.

## RFM definition

- **Recency (R):** days between the analysis reference date and a customer's most recent delivered order. Lower is better.
- **Frequency (F):** number of distinct delivered orders.
- **Monetary (M):** total payment value of delivered orders in BRL.

The analysis reference date is one day after the latest purchase timestamp in the dataset. Only delivered orders are included in the baseline RFM table. Payments are aggregated by order before joining to prevent duplicate order rows.

## Dataset

Download the Olist dataset from Kaggle: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

Place these files in `data/raw/`:

- `olist_orders_dataset.csv`
- `olist_order_items_dataset.csv` (not required by the baseline pipeline)
- `olist_customers_dataset.csv`
- `olist_order_payments_dataset.csv`

Do not commit the raw dataset to Git.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Windows:
```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

From the repository root:

```bash
python main.py --data-dir data/raw --output-dir outputs --clusters 4 --sample-size 5000
```

Optional arguments:

- `--clusters`: number of clusters used by both algorithms (default: 4)
- `--sample-size`: reproducible number of customers used by both algorithms (default: 5,000); use `0` for all customers
- `--linkage`: hierarchical linkage method: `ward`, `complete`, `average`, or `single` (default: `ward`)

## Outputs

Agglomerative clustering can require quadratic memory/time as the number of customers grows. For that reason, the default experiment draws one reproducible sample of 5,000 customers and runs **both** algorithms on that exact same sample. The full RFM table is still saved. The segment and metric outputs describe the experiment sample, not every customer. You can change the sample size or use `--sample-size 0` only if your machine can handle it. Report the sample size and random seed in the thesis.

The run creates:

- `outputs/rfm_customers.csv`: customer-level RFM features
- `outputs/customer_segments.csv`: RFM data with both algorithm labels
- `outputs/metrics.csv`: internal clustering metrics and runtime
- `outputs/kmeans_clusters.png` and `outputs/hierarchical_clusters.png`: PCA-based 2D visualizations

## Evaluation

The project reports:

- **Silhouette Score:** higher is better; measures separation and cohesion.
- **Davies-Bouldin Index:** lower is better; measures similarity between clusters.
- **Calinski-Harabasz Score:** higher is better; compares between-cluster and within-cluster dispersion.
- **Runtime:** elapsed fitting time in seconds.

Metrics are calculated in the same standardized RFM feature space used for clustering. PCA is used only to visualize the results in two dimensions.

## Project structure

```
rfm/
├── data/
│   └── raw/
├── outputs/
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── clustering.py
│   └── evaluation.py
├── main.py
├── requirements.txt
└── .gitignore
```

## Research notes

This is the baseline implementation, not a final research conclusion. Before using results in a thesis, document the dataset version, missing-value handling, outlier strategy, feature transformation, cluster-count rationale, and linkage choice. Consider testing multiple values of k and multiple hierarchical linkage methods as additional experiments. Interpret each cluster using the original (unscaled) RFM values rather than PCA coordinates alone.
