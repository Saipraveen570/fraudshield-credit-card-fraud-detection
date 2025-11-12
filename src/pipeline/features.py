import pandas as pd
import numpy as np

def add_time_features(df: pd.DataFrame, timestamp_col: str = "timestamp"):
    df["timestamp"] = pd.to_datetime(df[timestamp_col])
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    return df

def add_customer_velocity(df: pd.DataFrame):
    """
    Add time-windowed transaction counts per customer using proper time-based rolling windows.
    Requires df to have 'timestamp' (datetime), 'customer_id', and a unique 'transaction_id'.
    """
    # ensure datetime & sort
    df = df.sort_values(["customer_id", "timestamp"]).copy()

    # We'll use a helper Series that always exists; if transaction_id isn't present, create one.
    if "transaction_id" not in df.columns:
        df["transaction_id"] = np.arange(len(df))

    # 1h rolling count per customer
    tx_1h = (
        df.set_index("timestamp")
          .groupby("customer_id")["transaction_id"]
          .rolling("1h").count()
          .reset_index(level=0, drop=True)
    )

    # 24h rolling count per customer
    tx_24h = (
        df.set_index("timestamp")
          .groupby("customer_id")["transaction_id"]
          .rolling("24h").count()
          .reset_index(level=0, drop=True)
    )

    # Align back to original rows (after the earlier sort)
    df["tx_count_1h"] = tx_1h.values
    df["tx_count_24h"] = tx_24h.values

    return df

def add_amount_stats(df: pd.DataFrame):
    df = df.sort_values(["customer_id", "timestamp"])
    grp = df.groupby("customer_id")["amount"]
    df["avg_amount_7d"] = grp.transform(lambda s: s.rolling(window=20, min_periods=1).mean())
    df["std_amount_7d"] = grp.transform(lambda s: s.rolling(window=20, min_periods=1).std().fillna(0.0))
    return df

def add_geo_device_flags(df: pd.DataFrame):
    # Sort to ensure "first seen" logic is time-consistent
    df = df.sort_values(["customer_id", "timestamp"]).copy()

    # device_new: first time this device is seen for the customer
    df["device_new"] = (
        ~df.groupby("customer_id")["device_id"].apply(lambda s: s.duplicated())
    ).astype(int).values

    # home_country: the first country observed for this customer (no leakage)
    home_first = df.groupby("customer_id")["country"].transform("first")
    df["is_cross_border"] = (df["country"] != home_first).astype(int)

    return df

def add_merchant_risk(df: pd.DataFrame):
    # simple proxy: historical fraud rate per merchant up to that point
    df = df.sort_values(["merchant_id", "timestamp"])
    def cum_fraud_rate(s):
        csum = s.cumsum()
        cnt = np.arange(1, len(s)+1)
        return (csum.shift(fill_value=0) / cnt)
    df["merchant_risk"] = df.groupby("merchant_id")["label"].transform(cum_fraud_rate).fillna(0.0)
    return df

def build_feature_frame(df: pd.DataFrame):
    df = add_time_features(df)
    df = add_customer_velocity(df)
    df = add_amount_stats(df)
    df = add_geo_device_flags(df)
    df = add_merchant_risk(df)
    return df
