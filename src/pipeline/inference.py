import argparse, json
import pandas as pd
import joblib
import yaml

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main(config_path, json_input):
    cfg = load_config(config_path)
    paths = cfg["paths"]
    data_cfg = cfg["data"]
    model = joblib.load(paths["model_file"])

    tx = pd.DataFrame([json.loads(json_input)])
    # Ensure columns exist
    for c in data_cfg["categorical"] + data_cfg["numeric"]:
        if c not in tx.columns:
            tx[c] = None
    score = float(model.predict_proba(tx[data_cfg["categorical"] + data_cfg["numeric"]])[:,1][0])
    print(json.dumps({"score": score}, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="src/config/config.yaml")
    ap.add_argument("--json", type=str, required=True, help='Single transaction JSON string')
    args = ap.parse_args()
    main(args.config, args.json)
