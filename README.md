# 💳 FraudShield – Credit Card Fraud Detection & Monitoring System

A **real-world, end-to-end Credit Card Fraud Detection project** that simulates how modern financial institutions detect and monitor fraudulent transactions using **Python (LightGBM, FastAPI)** and **Power BI**.

This project integrates **data engineering, machine learning, model deployment, and business intelligence** — exactly what real banking risk analytics teams build.

---

## 🚀 Key Features

✅ **1. Data Generation & Preprocessing**
- Synthetic transaction data (customer, merchant, amount, MCC, country, device, etc.)
- Realistic behavior simulation (cross-border risk, velocity patterns, merchant risk)

✅ **2. Feature Engineering Pipeline**
- Rolling transaction velocity
- Geo & device change flags
- 7-day statistical aggregations
- Encoded categorical variables and standardized numeric features

✅ **3. Model Training**
- Built using **LightGBM** for high performance on imbalanced datasets  
- Hyperparameter tuning & threshold optimization  
- Output model metrics and threshold reports to `/reports`

✅ **4. Model Serving API**
- FastAPI backend for real-time fraud scoring  
- `/score` → returns fraud probability  
- `/score-with-decision` → returns both score & recommended action  
- `/score-batch` → batch scoring for large transaction files  
- `/explain` → SHAP-based feature explainability for a single transaction  

✅ **5. Logging & Monitoring**
- Predictions automatically logged to `reports/predictions_log.csv`  
- Threshold-based decisions logged as “allow”, “review”, or “block”

✅ **6. Power BI Dashboard (Business View)**
- Interactive fraud monitoring dashboard with:
  - 📈 Daily alert trends  
  - 🍩 Action distribution (allow/review/block)  
  - 💰 Expected loss and score distributions  
  - ⚙️ Adjustable thresholds (Decision & Allow cutoffs)
- Designed for real-time fraud risk reporting and executive visibility  

---

## Repo Structure
```
fraudshield_python/
├── app/
│   └── main.py                # FastAPI scoring service
├── data/                      # Raw / generated data
├── models/                    # Saved models & artifacts
├── notebooks/                 # (optional) EDA / experiments
├── reports/                   # Model card, metrics, etc.
├── src/
│   ├── config/
│   │   └── config.yaml        # Paths and model settings
│   ├── pipeline/
│   │   ├── generate_synthetic.py
│   │   ├── features.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── inference.py
│   └── serving/
│       └── schema.py          # Pydantic schema for API
├── tests/
│   ├── test_features.py
│   └── test_api_payload.py
├── requirements.txt
└── README.md
```

## Business-first Metrics
- **Expected Loss Saved** at threshold τ
- **Precision@K** (K = daily alert budget)
- **Recall at FPR=1%**
- PR-AUC

## Replace Synthetic with Real Data
Drop a CSV at `data/transactions.csv` with columns similar to the synthetic generator output
(`timestamp, customer_id, merchant_id, amount, country, mcc, channel, device_id, label`).
Run the same training & evaluation commands.

## Power Users
- Configure via `src/config/config.yaml` (features, LightGBM params, paths).
- Dockerize API by adding a Dockerfile in `app/` (template included below).

## License
MIT
