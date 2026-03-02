# 🔷 Structured Pipeline

ML-based fraud detection using a **17-model ensemble** trained on SEC financial filings. Processes structured JSON financial data and outputs a weighted fraud probability score.

---

## Quick Start

```bash
cd Main_Immplementation
source venv/bin/activate

# Run on a single JSON file
cd stuctured_pipeline
python3 inference_pipeline.py ../Input/0001040719.json

# Or via the unified runner (recommended)
cd ..
python3 run_unified.py
```

Output is saved to `stuctured_pipeline/Output/<filename>_results.json`.

---

## Folder Structure

```
stuctured_pipeline/
├── inference_pipeline.py      # 🎯 Main entry point — runs all 17 models
├── run_inference.py           # CLI wrapper for inference_pipeline.py
│
├── ── Feature Engineering ──
├── json_to_features.py        # Raw JSON → feature DataFrame (43 features)
├── derive_features.py         # Calculates financial ratios from raw fields
├── data_tranform.py           # Scaling, encoding, alignment to training schema
│
├── ── Training ──
├── train_model.py             # Full training script (classification + clustering)
├── classification_pipeline.py # Supervised model training
├── clustering_pipeline.py     # Unsupervised anomaly model training
├── combined_pipeline.py       # Runs both training pipelines together
├── structured_pipeline.py     # End-to-end training pipeline
│
├── ── Risk Scoring ──
├── risk_scorer_structured.py  # Structured risk score calculation
├── structured_pipeline_with_risk.py  # Training pipeline with risk output
│
├── ── Analysis & Reports ──
├── generate_plots.py          # Performance visualizations
├── generate_grouped_plots.py  # Grouped metric bar charts
├── RISK_SCORING_GUIDE.md      # Detailed risk scoring documentation
│
├── MyModels/                  # 🗄️ Trained model files (17 models)
│   ├── autoencoder.pt
│   ├── catboost_model.cbm
│   ├── xgboost_model.json
│   ├── lightgbm_model.txt
│   └── ... (51 files total)
│
├── Input/                     # Input JSON files
├── Output/                    # Per-file inference results
└── model_weights.json         # AUC-based model weights
```

---

## The 17-Model Ensemble

Models are organized into two categories:

### Classification Models (supervised)
| Model | Key | Training AUC |
|-------|-----|-------------|
| CatBoost | `catboost` | 0.9999 |
| XGBoost | `xgboost` | 0.9998 |
| LightGBM | `lightgbm` | 0.9999 |
| Random Forest | `random_forest` | 0.9999 |
| Logistic Regression | `logistic_regression` | 0.9940 |
| SVM | `svm` | 0.9985 |
| Decision Tree | `decision_tree` | 0.9989 |
| DNN | `dnn` | 0.9995 |
| CNN | `cnn` | 0.9994 |

### Anomaly / Clustering Models (unsupervised)
| Model | Key | Training AUC |
|-------|-----|-------------|
| Autoencoder | `autoencoder` | 0.9208 |
| Isolation Forest | `isolation_forest` | 0.8967 |
| One-Class SVM | `oneclass_svm` | 0.7798 |
| LOF | `lof` | 0.7600 |
| DBSCAN | `dbscan` | 0.5766 |
| KMeans | `kmeans` | 0.6438 |
| GMM | `gmm` | 0.9528 |
| PCA Anomaly | `pca_anomaly` | 0.5426 |

> Weights are stored in `model_weights.json` and are proportional to training AUC. Higher AUC = more weight in the ensemble vote.

---

## Data Flow

```
Input JSON (SEC filing)
        │
        ▼
json_to_features.py        → extracts 43 raw financial fields
        │
        ▼
derive_features.py         → computes 30+ derived ratios (D/E, ROA, current ratio...)
        │
        ▼
data_tranform.py           → aligns to training schema, scales features
        │
        ▼
inference_pipeline.py      → runs all 17 models in parallel
        │
        ▼
risk_scorer_structured.py  → weighted ensemble vote → risk score 0–1
        │
        ▼
Output/<file>_results.json → prediction, risk level, per-model breakdown
```

---

## Feature Engineering

### Raw Fields Extracted (`json_to_features.py`)

The pipeline reads three sections from the input JSON:

| Section | Key Fields |
|---------|-----------|
| `balance_sheet` | `total_assets`, `total_liabilities`, `equity`, `cash`, `accounts_receivable`, `inventory`, `fixed_assets_net`, `retained_earnings` |
| `income_statement` | `total_revenues`, `net_income_loss`, `total_operating_expenses`, `interest_expense`, `income_tax_expense`, `depreciation_amortization`, `eps`, `shares_outstanding` |
| `cash_flow` | `net_cash_operating`, `net_cash_financing`, `net_change_cash`, `proceeds_stock_sales` |

### Derived Ratios (`derive_features.py`)

| Ratio | Formula |
|-------|---------|
| `gross_profit_margin` | `(revenue - COGS) / revenue` |
| `return_on_assets` | `net_income / total_assets` |
| `debt_to_equity` | `total_liabilities / equity` |
| `current_ratio` | `current_assets / current_liabilities` |
| `asset_turnover` | `revenue / total_assets` |
| `expense_to_revenue` | `total_operating_expenses / revenue` |
| ... | 25+ more ratios |

---

## Output Format

```json
{
  "input_file": "0001040719.json",
  "risk_score": 0.0837,
  "risk_level": "MINIMAL",
  "overall_prediction": "NORMAL",
  "models_predicting_fraud": ["Dbscan", "Isolation Forest", "Oneclass Svm"],
  "fraud_model_count": 3,
  "normal_model_count": 13,
  "total_models": 17,
  "individual_model_results": {
    "catboost": {
      "prediction": "NORMAL",
      "fraud_probability": 0.0215,
      "weight": 0.94,
      "training_auc": 0.9999
    }
  },
  "success": true
}
```

### Risk Level Mapping

| Risk Score | Level | Interpretation |
|-----------|-------|---------------|
| 0.00–0.19 | MINIMAL | Very unlikely fraud |
| 0.20–0.39 | LOW | Minor anomalies |
| 0.40–0.59 | MEDIUM | Warrants review |
| 0.60–0.79 | HIGH | Strong fraud signals |
| 0.80–1.00 | CRITICAL | Immediate investigation |

---

## Model Weights (`model_weights.json`)

```json
{
  "catboost": 0.94,
  "xgboost": 0.95,
  "lightgbm": 0.93,
  "random_forest": 0.92,
  "autoencoder": 0.80,
  "isolation_forest": 0.75
}
```

The final risk score is a **weighted average** of all models' fraud probabilities, using these AUC-based weights.

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `Model file not found` | Missing `MyModels/` files | Ensure trained model files are present |
| `Feature mismatch` | Input JSON missing fields | Check `json_to_features.py` for required fields |
| `LOF ERROR` | Only 1 sample (normal in inference) | Expected — LOF needs multiple samples |
| `KeyError in derive_features` | Unexpected field name | Check JSON structure matches expected schema |
