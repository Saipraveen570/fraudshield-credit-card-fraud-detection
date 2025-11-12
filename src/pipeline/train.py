import argparse, json, os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, precision_recall_curve, auc
import joblib
import yaml

from .features import build_feature_frame

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def temporal_split(df, timestamp_col, valid_fraction_time=0.2):
    df = df.sort_values(timestamp_col)
    cutoff = df[timestamp_col].quantile(1 - valid_fraction_time)
    return df[df[timestamp_col] <= cutoff], df[df[timestamp_col] > cutoff]

def main(config_path):
    cfg = load_config(config_path)
    paths = cfg["paths"]
    data_cfg = cfg["data"]
    split_cfg = cfg["split"]
    model_cfg = cfg["model"]

    df = pd.read_csv(paths["data_csv"])
    df = build_feature_frame(df)

    # Prepare columns
    y = df[data_cfg["label_col"]].astype(int).values
    timestamp_col = pd.to_datetime(df[data_cfg["timestamp_col"]])
    df[data_cfg["timestamp_col"]] = timestamp_col  # ensure datetime
    X = df[data_cfg["categorical"] + data_cfg["numeric"]].copy()

    # Temporal split
    train_df, valid_df = temporal_split(df.assign(_ts=timestamp_col), "_ts", split_cfg["valid_fraction_time"])
    X_train = train_df[data_cfg["categorical"] + data_cfg["numeric"]]
    y_train = train_df[data_cfg["label_col"]].astype(int).values
    X_valid = valid_df[data_cfg["categorical"] + data_cfg["numeric"]]
    y_valid = valid_df[data_cfg["label_col"]].astype(int).values

    # Preprocess: OneHot for categoricals
    cat_cols = data_cfg["categorical"]
    num_cols = data_cfg["numeric"]
    pre = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

    lgbm = lgb.LGBMClassifier(**model_cfg["lightgbm_params"], n_estimators=model_cfg["num_boost_round"])
    pipe = Pipeline([("pre", pre), ("clf", lgbm)])

    pipe.fit(X_train, y_train)

    # Evaluate PR-AUC
    val_scores = pipe.predict_proba(X_valid)[:, 1]
    pr_auc = average_precision_score(y_valid, val_scores)
    p, r, _ = precision_recall_curve(y_valid, val_scores)
    pr_curve_auc = auc(r, p)

    os.makedirs(paths["model_dir"], exist_ok=True)
    joblib.dump(pipe, paths["model_file"])

    # Save feature list (post-transform info is implicit, but we keep raw names)
    with open(paths["feature_list_file"], "w") as f:
        json.dump({"categorical": cat_cols, "numeric": num_cols}, f, indent=2)

    os.makedirs(os.path.dirname(paths["metrics_file"]), exist_ok=True)
    with open(paths["metrics_file"], "w") as f:
        json.dump({"valid_pr_auc": pr_auc, "valid_pr_curve_auc": pr_curve_auc}, f, indent=2)

    print(f"Saved model to {paths['model_file']}")
    print(f"Validation PR-AUC (avg precision): {pr_auc:.4f} | PR-curve AUC: {pr_curve_auc:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="src/config/config.yaml")
    args = ap.parse_args()
    main(args.config)
