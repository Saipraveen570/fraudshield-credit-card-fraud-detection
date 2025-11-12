import argparse, json
import pandas as pd
import joblib
import yaml
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score

from .features import build_feature_frame

def load_config(path):
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

def precision_at_k(y_true, scores, k: int):
    idx = np.argsort(-scores)[:k]
    return float(y_true[idx].mean())

def expected_loss_saved(y_true, scores, amounts, threshold: float, det_rate: float = 0.8):
    # simplistic: if score >= threshold, we "prevent" det_rate of the amount when fraud occurs
    preds = (scores >= threshold).astype(int)
    return float((preds * y_true * amounts * det_rate).sum())

def main(config_path, k):
    cfg = load_config(config_path)
    paths = cfg["paths"]
    data_cfg = cfg["data"]

    df = pd.read_csv(paths["data_csv"])
    df = build_feature_frame(df)

    model = joblib.load(paths["model_file"])
    X = df[data_cfg["categorical"] + data_cfg["numeric"]]
    y = df[data_cfg["label_col"]].astype(int).values
    scores = model.predict_proba(X)[:, 1]
    ap = average_precision_score(y, scores)
    roc = roc_auc_score(y, scores)

    p, r, t = precision_recall_curve(y, scores)
    best_idx = np.argmax(p * r)  # simple harmonic-like
    best_thr = t[best_idx] if best_idx < len(t) else 0.5

    pak = precision_at_k(y, scores, k)
    exp_loss = expected_loss_saved(y, scores, df["amount"].values, threshold=best_thr)

    metrics = {
        "AP": float(ap),
        "ROC_AUC": float(roc),
        "Precision@K": float(pak),
        "BestThreshold": float(best_thr),
        "ExpectedLossSaved_at_BestThr": float(exp_loss)
    }
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="src/config/config.yaml")
    ap.add_argument("--k", type=int, default=200)
    args = ap.parse_args()
    main(args.config, args.k)
