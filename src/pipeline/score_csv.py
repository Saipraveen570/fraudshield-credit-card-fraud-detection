import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd
import yaml

# ✅ Use the same feature engineering used at training time
from src.pipeline.features import build_feature_frame


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_expected_columns(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Guarantee all expected columns exist (categorical + numeric) after feature engineering.
    If any are still missing, create them as None/NaN so the pipeline won't crash.
    """
    expected = cfg["data"]["categorical"] + cfg["data"]["numeric"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        for c in missing:
            df[c] = None
        print(f"[warn] Added missing expected columns (no data): {missing}")
    return df


def read_best_threshold(metrics_path: str, default_thr: float = 0.75) -> float:
    if metrics_path and os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                m = json.load(f)
            if "BestThreshold" in m:
                return float(m["BestThreshold"])
        except Exception:
            pass
    return float(default_thr)


def main(in_csv: str, out_csv: str, config: str):
    cfg = load_config(config)
    paths = cfg["paths"]
    data_cfg = cfg["data"]

    # Load trained pipeline (preprocessor + model)
    pipe = joblib.load(paths["model_file"])

    # Load raw transactions
    df = pd.read_csv(in_csv)

    # Normalize/parse timestamp if present
    # (keeps original column names; training features may derive time parts)
    for ts_col in ["timestamp", "ts"]:
        if ts_col in df.columns:
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
            # optional: sort by time for any rolling features
            df = df.sort_values(by=[ts_col]).reset_index(drop=True)
            break

    # ✅ Apply the SAME feature engineering as training
    df_feat = build_feature_frame(df.copy())

    # Ensure expected columns are present for the model
    df_feat = ensure_expected_columns(df_feat, cfg)

    # Build matrix using the columns defined in config
    use_cols = data_cfg["categorical"] + data_cfg["numeric"]
    X = df_feat[use_cols]

    # Predict probabilities with the saved pipeline
    scores = pipe.predict_proba(X)[:, 1]

    # Compose output: keep original + engineered + scores/decision
    df_out = df_feat.copy()
    df_out["score"] = scores

    # Thresholds
    thr = read_best_threshold(paths.get("metrics_file", "reports/metrics.json"), default_thr=0.75)
    allow_cutoff = float(os.getenv("FS_ALLOW_CUTOFF", "0.20"))

    # Map to actions (allow / review / block)
    #   score >= thr         -> block
    #   allow_cutoff <= s<thr-> review
    #   s < allow_cutoff     -> allow
    def to_action(s: float) -> str:
        if pd.isna(s):
            return "review"
        if s >= thr:
            return "block"
        if s < allow_cutoff:
            return "allow"
        return "review"

    df_out["decision"] = df_out["score"].apply(to_action)

    # Write
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)

    print(
        f"Wrote {len(df_out):,} rows with scores to {out_path} "
        f"(threshold={thr}, allow_cutoff={allow_cutoff})"
    )
    # Quick mix preview
    print(df_out["decision"].value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", default="data/transactions.csv")
    ap.add_argument("--out", dest="out_csv", default="reports/scored_transactions.csv")
    ap.add_argument("--config", default="src/config/config.yaml")
    args = ap.parse_args()
    main(args.in_csv, args.out_csv, args.config)