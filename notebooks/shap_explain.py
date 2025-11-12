import sys, os
sys.path.append(os.path.abspath("."))
import warnings
warnings.filterwarnings("ignore", message="LightGBM binary classifier with TreeExplainer shap values output has changed")

# notebooks/shap_explain.py
import os
from pathlib import Path
import json
import yaml
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# ✅ NEW: bring in the same feature engineering as training
from src.pipeline.features import build_feature_frame

CONFIG_PATH = "src/config/config.yaml"
OUT_DIR = Path("reports/shap")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_feature_names(preprocessor, cat_cols, num_cols):
    try:
        cat = preprocessor.named_transformers_["cat"]
        cat_names = cat.get_feature_names_out(cat_cols)
        return list(cat_names) + list(num_cols)
    except Exception:
        return None

def main():
    cfg = load_cfg(CONFIG_PATH)
    pipe = joblib.load(cfg["paths"]["model_file"])
    pre = pipe.named_steps.get("pre", None)
    clf = pipe.named_steps.get("clf", None)
    if pre is None or clf is None:
        raise RuntimeError("Pipeline missing expected steps: 'pre' and/or 'clf' not found.")

    cat = cfg["data"]["categorical"]
    num = cfg["data"]["numeric"]

    # Load RAW data, then ✅ derive engineered features to match training
    df_raw = pd.read_csv(cfg["paths"]["data_csv"])
    df_feat = build_feature_frame(df_raw.copy())

    # Keep a manageable sample for speed
    df_feat = df_feat.head(2000)

    # Use the exact columns used in training (cat + num after feature engineering)
    X_raw = df_feat[cat + num]

    # Transform with the trained preprocessor
    X_proc = pre.transform(X_raw)

    # Post-transform feature names
    feature_names = build_feature_names(pre, cat, num)
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X_proc.shape[1])]

    # ---- SHAP ----
    try:
        explainer = shap.TreeExplainer(clf)
        shap_vals = explainer.shap_values(X_proc)
        expected_value = explainer.expected_value
    except Exception:
        explainer = shap.Explainer(clf.predict_proba, feature_names=feature_names)
        shap_vals = explainer(X_proc)
        expected_value = getattr(shap_vals, "base_values", None)

    if isinstance(shap_vals, list):
        sv_pos = shap_vals[1]
        base_val = expected_value[1] if isinstance(expected_value, (list, np.ndarray)) else expected_value
    else:
        sv_pos = shap_vals
        base_val = expected_value

    sv_arr = np.asarray(sv_pos)
    Xp_arr = np.asarray(X_proc)

    # Beeswarm
    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv_arr, Xp_arr, feature_names=feature_names, show=False)
    (OUT_DIR / "shap_summary_beeswarm.png").parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_summary_beeswarm.png", dpi=160)
    plt.close()

    # Bar
    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv_arr, Xp_arr, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_summary_bar.png", dpi=160)
    plt.close()

    # Waterfall for top-scored row
    try:
        scores = pipe.predict_proba(X_raw)[:, 1]
        i = int(np.argmax(scores))
    except Exception:
        i = 0

    values_i = sv_arr[i].ravel()
    data_i = Xp_arr[i].ravel()
    try:
        exp = shap.Explanation(values=values_i, base_values=base_val, data=data_i, feature_names=feature_names)
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(exp, show=False, max_display=15)
        plt.savefig(OUT_DIR / "shap_waterfall_top_case.png", dpi=160, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Waterfall plot skipped: {e}")

    # Save top features JSON
    mean_abs = np.mean(np.abs(sv_arr), axis=0)
    order = np.argsort(-mean_abs)
    top_feats = [{"feature": feature_names[j], "mean_abs_shap": float(mean_abs[j])} for j in order[:20]]
    with open(OUT_DIR / "top_features.json", "w") as f:
        json.dump(top_feats, f, indent=2)

if __name__ == "__main__":
    main()