import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

def main(rows: int, out_path: str, seed: int = 42):
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1, 0, 0, 0)
    # Catalogs
    countries = ["IN", "US", "GB", "SG", "AE", "DE"]
    mccs = ["5411", "5814", "5999", "5732", "4789", "5942"]  # grocery, fast food, misc, electronics, transport, books
    channels = ["POS", "ECOM", "ATM"]
    n_customers = 6000
    n_merchants = 1200
    n_devices = 8000

    customer_home = {i: rng.choice(countries, p=[0.65,0.08,0.07,0.06,0.08,0.06]) for i in range(n_customers)}
    merchant_country = {i: rng.choice(countries) for i in range(n_merchants)}
    merchant_mcc = {i: rng.choice(mccs, p=[0.25,0.2,0.15,0.2,0.1,0.1]) for i in range(n_merchants)}

    rows_data = []
    ts = start
    for tx_id in range(rows):
        ts += timedelta(minutes=int(rng.integers(0, 10)))
        cust = rng.integers(0, n_customers)
        merch = rng.integers(0, n_merchants)
        device = rng.integers(0, n_devices)

        base_amt = rng.lognormal(mean=3.5, sigma=0.8)  # skewed
        amount = round(float(base_amt), 2)

        # Cross-border likelihood higher for ECOM
        channel = rng.choice(channels, p=[0.6, 0.35, 0.05])
        ctry = merchant_country[merch]
        is_cross = int(ctry != customer_home[cust] and rng.random() < (0.25 if channel=="ECOM" else 0.08))

        # Merchant risk: a few merchants are risky
        merchant_risk = 1 if rng.random() < 0.02 else 0

        # Fraud generation: higher prob on cross-border ecom, high amount, risky merchant, new device
        prob_fraud = (
            0.002
            + 0.02 * is_cross
            + 0.03 * (channel == "ECOM")
            + 0.015 * (amount > 300)
            + 0.04 * merchant_risk
        )
        label = int(rng.random() < prob_fraud)

        rows_data.append({
            "transaction_id": tx_id,
            "timestamp": ts.isoformat(),
            "customer_id": int(cust),
            "merchant_id": int(merch),
            "amount": amount,
            "country": ctry,
            "mcc": merchant_mcc[merch],
            "channel": channel,
            "device_id": f"d{int(device)}",
            "label": label
        })

    df = pd.DataFrame(rows_data)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} rows to {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=50000)
    ap.add_argument("--out", type=str, default="data/transactions.csv")
    args = ap.parse_args()
    main(args.rows, args.out)
