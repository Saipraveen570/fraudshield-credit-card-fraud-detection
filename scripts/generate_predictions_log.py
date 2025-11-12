import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# --- Defaults ---
SRC_DEFAULT = "reports/scored_transactions.csv"
DEST_DEFAULT = "reports/predictions_log.csv"
N_DEFAULT = 2000
TARGET_DEFAULT = {"allow": 0.70, "review": 0.25, "block": 0.05}

RENAME_MAP = {
    "timestamp": "ts",
    "decision": "action",
}

REQUIRED_COLS = [
    "ts",
    "customer_id",
    "merchant_id",
    "amount",
    "country",
    "mcc",
    "channel",
    "device_id",
    "score",
    "action",
]


def parse_args():
    ap = argparse.ArgumentParser(description="Generate a balanced predictions_log.csv from scored_transactions.csv")
    ap.add_argument("--src", default=SRC_DEFAULT, help="Input scored transactions CSV")
    ap.add_argument("--dest", default=DEST_DEFAULT, help="Output predictions log CSV (overwritten)")
    ap.add_argument("--n", type=int, default=N_DEFAULT, help="Number of rows to output")
    ap.add_argument("--allow", type=float, default=TARGET_DEFAULT["allow"], help="Proportion for 'allow'")
    ap.add_argument("--review", type=float, default=TARGET_DEFAULT["review"], help="Proportion for 'review'")
    ap.add_argument("--block", type=float, default=TARGET_DEFAULT["block"], help="Proportion for 'block'")
    ap.add_argument("--no-rebalance", action="store_true", help="Use existing action column (if present) without rebalancing")
    return ap.parse_args()


def to_iso8601_z(s):
    """Best-effort parse and return ISO-8601 with 'Z' suffix for UTC-like display."""
    try:
        ts = pd.to_datetime(s, errors="coerce", utc=True)
        return ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        # If parsing fails, just return as-is string
        return s.astype(str)


def main():
    args = parse_args()
    SRC = Path(args.src)
    DEST = Path(args.dest)
    N = args.n
    target = {"allow": args.allow, "review": args.review, "block": args.block}

    if round(sum(target.values()), 6) != 1.0:
        raise ValueError(f"Target mix must sum to 1.0, got {target}")

    if not SRC.exists():
        raise FileNotFoundError(f"Input not found: {SRC}")

    print(f"Reading: {SRC}")
    df = pd.read_csv(SRC)

    # Normalize column names to predictions_log schema
    df = df.rename(columns=RENAME_MAP).copy()

    # If 'ts' is missing but we have something like 'time', try it
    if "ts" not in df.columns:
        for alt in ["time", "datetime", "event_time"]:
            if alt in df.columns:
                df.rename(columns={alt: "ts"}, inplace=True)
                break

    # Ensure the columns that exist are typed correctly
    # (don't fail if some are missing yet—handled below)
    for col in ["amount", "score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # If 'action' is missing or rebalance requested, derive action from score quantiles
    need_action = ("action" not in df.columns) or args.no_rebalance is False
    if need_action:
        # Compute quantiles to approximate target split
        if "score" not in df.columns:
            raise ValueError("Missing 'score' column; cannot derive actions.")
        # Quantiles: allow below q_allow, block at/above q_block, else review
        q_allow = df["score"].quantile(target["allow"])
        q_block = df["score"].quantile(1 - target["block"])

        def decide(s):
            if pd.isna(s):
                return "review"
            if s >= q_block:
                return "block"
            if s < q_allow:
                return "allow"
            return "review"

        df["action"] = df["score"].apply(decide)

    # Build a working frame with only required columns (create missing if necessary)
    out = pd.DataFrame()
    out["ts"] = df["ts"] if "ts" in df.columns else pd.NaT
    for c in ["customer_id", "merchant_id"]:
        out[c] = df[c] if c in df.columns else None
    for c in ["amount", "country", "mcc", "channel", "device_id", "score", "action"]:
        out[c] = df[c] if c in df.columns else None

    # Drop rows missing essential fields
    out = out.dropna(subset=["score"])
    if len(out) == 0:
        raise ValueError("No rows with a valid 'score' found.")

    # Format timestamps to ISO-8601 Z
    out["ts"] = to_iso8601_z(out["ts"])

    # If fewer than N rows available, cap to available
    if len(out) < N:
        print(f"⚠️ Only {len(out)} rows available; will output all of them instead of {N}.")
        N = len(out)

    # Stratified sampling to hit the target mix as closely as possible
    want = {k: int(v * N) for k, v in target.items()}
    parts = []
    rng = np.random.RandomState(42)
    for a, k in want.items():
        pool = out[out["action"] == a]
        take = min(k, len(pool))
        if take > 0:
            parts.append(pool.sample(take, random_state=rng))
    result = pd.concat(parts) if parts else out.head(0)

    # Top up if we fell short (e.g., class too small)
    if len(result) < N:
        remaining = N - len(result)
        leftovers = out[~out.index.isin(result.index)]
        if remaining > 0 and len(leftovers) > 0:
            extra = leftovers.sample(min(remaining, len(leftovers)), random_state=99)
            result = pd.concat([result, extra])

    # Final trim & shuffle
    result = result.sample(N, random_state=7)

    # Keep only required columns in correct order
    missing = [c for c in REQUIRED_COLS if c not in result.columns]
    if missing:
        # Fill missing required columns with None
        for c in missing:
            result[c] = None
    result = result[REQUIRED_COLS]

    DEST.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(DEST, index=False)

    # Print distribution summary
    print(f"✅ Wrote {len(result):,} rows to {DEST.resolve()}")
    print("Class distribution:")
    print(result["action"].value_counts(normalize=True).round(3).to_string())
    print("Score quantiles (source):")
    print(df["score"].quantile([.1, .2, .5, .8, .9, .95, .99]).round(4).to_string())


if __name__ == "__main__":
    main()