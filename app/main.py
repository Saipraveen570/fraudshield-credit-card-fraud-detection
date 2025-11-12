from fastapi import FastAPI
from typing import List
from datetime import datetime
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import yaml
import os
import json

from src.serving.schema import Transaction

# ---------- Logging (to CSV) ----------
LOG_PATH = Path("reports/predictions_log.csv")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

def log_prediction(payload: dict, score: float, action: str | None = None):
    row = {
        "ts": datetime.utcnow().isoformat(),
        **payload,
        "score": score,
        "action": action,
    }
    import csv
    write_header = not LOG_PATH.exists()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)

# Optional (silence LightGBM warnings in logs)
os.environ.setdefault("NUMEXPR_MAX_THREADS", "8")

CONFIG_PATH = os.getenv("FS_CONFIG", "src/config/config.yaml")

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

cfg = load_config(CONFIG_PATH)
paths = cfg["paths"]
data_cfg = cfg["data"]

# Load the trained Pipeline (preprocessor + model)
# NOTE: We saved a sklearn Pipeline in train.py
pipe = joblib.load(paths["model_file"])

# Extract components for explainability
# Expect pipeline = Pipeline([("pre", ColumnTransformer(...)), ("clf", LGBMClassifier)])
pre = getattr(pipe, "named_steps", {}).get("pre", None)
clf = getattr(pipe, "named_steps", {}).get("clf", None)

# Build feature name list after preprocessing (for SHAP output mapping)
try:
    cat_cols = data_cfg["categorical"]
    num_cols = data_cfg["numeric"]
    cat_names = pre.named_transformers_["cat"].get_feature_names_out(cat_cols)
    FEATURE_NAMES = list(cat_names) + list(num_cols)
except Exception:
    # Fallback if feature names cannot be constructed (shouldn't happen if trained with config)
    FEATURE_NAMES = None

# Decision thresholds (env-tunable)
DEFAULT_THRESHOLD = float(os.getenv("FS_THRESHOLD", "0.75"))      # block at/above this
ALLOW_CUTOFF = float(os.getenv("FS_ALLOW_CUTOFF", "0.20"))        # allow below this
THRESHOLD = DEFAULT_THRESHOLD

# Try to pull BestThreshold from metrics file if present
metrics_path = paths.get("metrics_file", "reports/metrics.json")
if os.path.exists(metrics_path):
    try:
        with open(metrics_path, "r") as f:
            m = json.load(f)
        if "BestThreshold" in m:
            THRESHOLD = float(m["BestThreshold"])
    except Exception:
        pass

app = FastAPI(title="FraudShield Scoring API", version="1.3.0")

# --------------------------- UTILITIES ---------------------------

def ensure_expected_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee all expected model columns exist (cat + numeric)."""
    for c in data_cfg["categorical"] + data_cfg["numeric"]:
        if c not in df.columns:
            df[c] = None
    return df

def score_dataframe(df: pd.DataFrame) -> np.ndarray:
    """Return probabilities for a DF of raw features."""
    df = ensure_expected_columns(df)
    # Pipe handles preprocessing + predict_proba
    return pipe.predict_proba(df[data_cfg["categorical"] + data_cfg["numeric"]])[:, 1]

# --------------------------- ROUTES ---------------------------

@app.get("/")
def root():
    return {
        "message": "✅ FraudShield API is running",
        "docs": "/docs",
        "health": "/health",
        "endpoints": ["/score", "/score-with-decision", "/score-batch", "/explain"],
        "threshold_in_use": THRESHOLD,
        "allow_cutoff_in_use": ALLOW_CUTOFF,
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/score")
def score(tx: Transaction):
    df = pd.DataFrame([tx.dict()])
    s = float(score_dataframe(df)[0])
    # Log prediction (no decision on this endpoint)
    log_prediction(tx.dict(), s, None)
    return {"score": s}

@app.post("/score-with-decision")
def score_with_decision(tx: Transaction):
    """
    Returns both the fraud score and a recommended action:
      - score >= THRESHOLD      -> "block"
      - ALLOW_CUTOFF <= score < THRESHOLD -> "review"
      - score < ALLOW_CUTOFF    -> "allow"
    (THRESHOLD and ALLOW_CUTOFF are tunable via env vars FS_THRESHOLD / FS_ALLOW_CUTOFF.)
    """
    df = pd.DataFrame([tx.dict()])
    s = float(score_dataframe(df)[0])
    if s >= THRESHOLD:
        action = "block"
    elif s < ALLOW_CUTOFF:
        action = "allow"
    else:
        action = "review"
    # Log prediction with decision
    log_prediction(tx.dict(), s, action)
    return {"score": s, "action": action, "threshold": THRESHOLD, "allow_cutoff": ALLOW_CUTOFF}

@app.post("/score-batch")
def score_batch(txs: List[Transaction]):
    df = pd.DataFrame([t.dict() for t in txs])
    scores = score_dataframe(df)
    # Log each prediction (action unknown here)
    for payload, sc in zip([t.dict() for t in txs], scores):
        log_prediction(payload, float(sc), None)
    return {"scores": [float(x) for x in scores]}

@app.post("/explain")
def explain(tx: Transaction, top_k: int = 10):
    """
    Returns top feature contributions (by absolute SHAP value) for a single transaction.
    NOTE:
      - Uses the pipeline's preprocessor to transform input before SHAP.
      - Computes SHAP values against the underlying LightGBM classifier.
    """
    if pre is None or clf is None:
        return {"error": "Explainability not available: pipeline missing expected steps 'pre'/'clf'."}

    # Build a single-row DF and preprocess
    df = pd.DataFrame([tx.dict()])
    df = ensure_expected_columns(df)
    X_raw = df[data_cfg["categorical"] + data_cfg["numeric"]]
    X_proc = pre.transform(X_raw)

    # Build feature names if not already
    feature_names = FEATURE_NAMES
    if feature_names is None:
        try:
            cat_names = pre.named_transformers_["cat"].get_feature_names_out(data_cfg["categorical"])
            feature_names = list(cat_names) + list(data_cfg["numeric"])
        except Exception:
            feature_names = [f"f{i}" for i in range(X_proc.shape[1])]

    # Compute SHAP with TreeExplainer for LightGBM
    import shap
    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X_proc)

    # For binary, shap_values is either a single array or a list [neg_class, pos_class]
    if isinstance(shap_vals, list):
        sv = shap_vals[1]  # positive class
    else:
        sv = shap_vals
    sv_row = np.asarray(sv)[0]

    # Rank top features by absolute impact
    idx = np.argsort(-np.abs(sv_row))[:top_k]
    top_contribs = []
    # Get transformed feature values for the row (to show alongside)
    x_vals = np.asarray(X_proc)[0]
    for i in idx:
        top_contribs.append({
            "feature": str(feature_names[i]) if i < len(feature_names) else f"f{i}",
            "shap_value": float(sv_row[i]),
            "abs_shap": float(abs(sv_row[i])),
            "value": float(x_vals[i]) if isinstance(x_vals[i], (int, float, np.number)) else (x_vals[i].item() if hasattr(x_vals[i], "item") else str(x_vals[i]))
        })

    return {
        "top_features": top_contribs,
        "threshold_in_use": THRESHOLD,
        "allow_cutoff_in_use": ALLOW_CUTOFF,
    }