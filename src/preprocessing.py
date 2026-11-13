import pandas as pd


def build_rfm(
    orders: pd.DataFrame,
    customers: pd.DataFrame,
    payments: pd.DataFrame,
) -> pd.DataFrame:
    """Build one RFM row per customer using delivered orders only."""
    required_order_columns = {
        "order_id",
        "customer_id",
        "order_status",
        "order_purchase_timestamp",
    }
    required_customer_columns = {"customer_id", "customer_unique_id"}
    required_payment_columns = {"order_id", "payment_value"}

    for label, frame, required in [
        ("orders", orders, required_order_columns),
        ("customers", customers, required_customer_columns),
        ("payments", payments, required_payment_columns),
    ]:
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{label} dataset is missing columns: {', '.join(sorted(missing))}")

    delivered = orders.loc[orders["order_status"].eq("delivered")].copy()
    delivered["order_purchase_timestamp"] = pd.to_datetime(
        delivered["order_purchase_timestamp"], errors="coerce"
    )
    delivered = delivered.dropna(subset=["order_purchase_timestamp", "customer_id", "order_id"])

    if delivered.empty:
        raise ValueError("No valid delivered orders were found in the orders dataset.")

    # Aggregate payments first: an order may have multiple payment rows.
    payment_totals = (
        payments.assign(payment_value=pd.to_numeric(payments["payment_value"], errors="coerce"))
        .groupby("order_id", as_index=False)["payment_value"]
        .sum(min_count=1)
    )

    delivered = delivered.merge(
        customers[["customer_id", "customer_unique_id"]].drop_duplicates("customer_id"),
        on="customer_id",
        how="inner",
        validate="many_to_one",
    )
    delivered = delivered.merge(
        payment_totals,
        on="order_id",
        how="left",
        validate="one_to_one",
    )
    delivered["payment_value"] = delivered["payment_value"].fillna(0)

    reference_date = delivered["order_purchase_timestamp"].max().normalize() + pd.Timedelta(days=1)

    rfm = (
        delivered.groupby("customer_unique_id")
        .agg(
            last_purchase=("order_purchase_timestamp", "max"),
            frequency=("order_id", "nunique"),
            monetary=("payment_value", "sum"),
        )
        .reset_index()
    )
    rfm["recency"] = (reference_date - rfm["last_purchase"].dt.normalize()).dt.days
    rfm = rfm[["customer_unique_id", "recency", "frequency", "monetary"]]
    rfm = rfm.dropna(subset=["recency", "frequency", "monetary"])
    rfm = rfm.loc[(rfm["frequency"] > 0) & (rfm["monetary"] >= 0)].reset_index(drop=True)

    if len(rfm) < 3:
        raise ValueError("At least three customers are required for clustering.")

    return rfm
