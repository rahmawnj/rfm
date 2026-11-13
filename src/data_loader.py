from pathlib import Path

import pandas as pd


REQUIRED_FILES = {
    "orders": "olist_orders_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
}


def load_olist_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three Olist CSV files required to build the baseline RFM table."""
    data_path = Path(data_dir)
    missing = [filename for filename in REQUIRED_FILES.values() if not (data_path / filename).is_file()]
    if missing:
        missing_text = "\n".join(f"- {name}" for name in missing)
        raise FileNotFoundError(
            f"Missing required Olist CSV file(s) in {data_path}:\n{missing_text}\n"
            "Download the dataset from Kaggle and place the files in data/raw/."
        )

    orders = pd.read_csv(data_path / REQUIRED_FILES["orders"])
    customers = pd.read_csv(data_path / REQUIRED_FILES["customers"])
    payments = pd.read_csv(data_path / REQUIRED_FILES["payments"])

    return orders, customers, payments
