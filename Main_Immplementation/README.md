# 🔍 Questor — Unified Fraud Detection Pipeline

A comprehensive, multi-layer fraud detection system for SEC financial filings. It combines **17 trained ML models**, **document-level text analysis**, and **15 specialized financial agents** into a single unified risk score.

---

## 🚀 Quick Start

```bash
cd Main_Immplementation
source venv/bin/activate

# Run the full pipeline on all files in Input/
python3 run_unified.py

# Disable agents (faster)
python3 run_unified.py --no-agents

# Custom input directory
python3 run_unified.py --input-dir /path/to/files
```

Results are saved to `Output/unified_results_<timestamp>.json`.

---

## 📁 Project Structure

```
Main_Immplementation/
│
├── run_unified.py              # 🎯 ENTRY POINT — runs everything
├── score_combiner.py           # Weighted score combination logic
├── config.yaml                 # Global pipeline configuration
├── requirements.txt            # Python dependencies
├── setup.sh                    # Environment setup script
│
├── Input/                      # 📥 Input JSON files (SEC filings by CIK)
├── Output/                     # 📊 Unified pipeline results
│
├── stuctured_pipeline/         # 🔷 17-model ML ensemble
│   ├── inference_pipeline.py   # Main inference entry point
│   ├── json_to_features.py     # Feature extraction from raw JSON
│   ├── derive_features.py      # Financial ratio computation
│   ├── MyModels/               # Trained model files
│   └── README.md               # ← Structured pipeline docs
│
├── unstructured_pipeline/      # 🔶 Document analysis & ChromaDB retrieval
│   ├── pipelines/              # Risk retriever, scorer, data loader
│   ├── databases/              # ChromaDB + Neo4j clients
│   ├── utils/                  # CIK extractor, config, logger
│   └── README.md               # ← Unstructured pipeline docs
│
├── agents/                     # 🤖 15 specialized fraud detection agents
│   ├── orchestrator.py         # Agent coordination & scoring
│   ├── agent_config.yaml       # Per-agent weights and thresholds
│   ├── base_agent.py           # Abstract base class
│   ├── [13 agent files]        # See agents/README.md
│   └── README.md               # ← Agent docs (all 15 described)
│
└── shared/                     # 📦 Shared utilities
    ├── output_schema.py        # RiskAssessment dataclass
    └── utils.py                # Logging, config loading
```

---

## 🏗️ Three-Layer Architecture

```
Input JSON (SEC Filing)
         │
         ├──────────────────────────────────────────┐
         │         [run in parallel]                │
         ▼                                          ▼
🔷 STRUCTURED PIPELINE                   🔶 UNSTRUCTURED PIPELINE
17-model ML ensemble                     ChromaDB document retrieval
   • CatBoost, XGBoost, LightGBM         • Keyword-based risk scoring
   • Random Forest, SVM, DNN, CNN        • Entity extraction
   • Isolation Forest, DBSCAN            • Relationship mapping
   • Autoencoder, GMM, KMeans            • ~2–4s (retrieval mode)
   • ~12s processing time
         │                                          │
         └──────────────┬───────────────────────────┘
                        │  [both complete →]
                        ▼
              🤖 AGENT ORCHESTRATOR
              15 specialized fraud agents
              (uses structured output + raw input JSON)
                • Altman Z-Score      • Tax Rate Anomaly
                • Cash Flow/Earnings  • Financing Red Flags
                • Debt Anomaly        • Asset Quality
                • Expense Padding     • EPS Consistency
                • Related Party       • Negative Equity
                • Benford's Law       • Liquidity Crunch
                • Beneish M-Score     • Depreciation Anomaly
                                      • Cash Flow Composition
                        │
                        ▼
              🔗 SCORE COMBINER
              Weighted final risk score (0–100)
```

---

## 🤖 All 15 Fraud Detection Agents

| Agent | What It Detects |
|-------|----------------|
| `altman_zscore` | Bankruptcy / financial distress (Z < 1.81 = danger) |
| `cashflow_earnings` | Accrual ratio — gap between book income and cash income |
| `debt_anomaly` | D/E ratio, interest coverage ratio, short-term debt concentration |
| `expense_padding` | Expense-to-revenue ratio, gross margin collapse |
| `related_party` | Related-party balances as % of liabilities |
| `tax_rate_anomaly` | Near-zero ETR on profitable company, tax benefit manipulation |
| `financing_red_flags` | Negative CFO covered by stock issuances or new debt |
| `asset_quality` | Other/intangible assets + receivables as % of total assets |
| `eps_consistency` | Reported EPS vs. `net_income ÷ shares_outstanding` |
| `negative_equity` | Negative total equity, accumulated deficit vs. paid-in capital |
| `liquidity_crunch` | Cash ratio, quick ratio, ending cash vs. burn rate |
| `depreciation_anomaly` | D&A rate implying 50+ year useful life (asset inflation) |
| `cashflow_composition` | Operations vs. investing vs. financing as cash sources |
| `benfords_law` | First-digit distribution deviation in financial numbers |
| `beneish_mscore` | 8-ratio earnings manipulation score (requires YoY data) |

> 📖 Full details → [`agents/README.md`](agents/README.md)

---

## 📊 Output Format

Results saved to `Output/unified_results_<timestamp>.json`:

```json
{
  "timestamp": "20260302_163030",
  "total_files": 1,
  "results": [
    {
      "cik": "1040719",
      "filename": "0001040719.json",

      "structured": {
        "risk_score": 0.0837,
        "risk_level": "MINIMAL",
        "overall_prediction": "NORMAL",
        "models_predicting_fraud": ["Dbscan", "Isolation Forest", "Oneclass Svm"],
        "fraud_model_count": 3,
        "total_models": 17
      },

      "unstructured": {
        "mode": "retrieval",
        "documents_retrieved": 1,
        "formatted_output": {
          "risk_assessment": {
            "overall_score": 24.91,
            "risk_level": "LOW",
            "risk_factors": ["High-risk keywords: material weakness, restatement"]
          }
        }
      },

      "agents": {
        "combined_score": 75.4,
        "confidence": 0.82,
        "agents_succeeded": 13,
        "individual_results": {
          "altman_zscore":       { "score": 83.6, "findings": ["Z-Score 0.24 — DISTRESS zone"] },
          "financing_red_flags": { "score": 80.0, "findings": ["Stock issuances funding cash burn"] },
          "cashflow_composition":{ "score": 75.0, "findings": ["Core business is cash-negative"] },
          "debt_anomaly":        { "score": 55.6, "findings": ["ICR -2.64 — cannot cover interest"] }
        }
      },

      "combined": {
        "combined_risk": {
          "overall_risk_score": 52.51,
          "risk_level": "MEDIUM",
          "confidence": 0.72
        }
      }
    }
  ]
}
```

---

## ⚙️ Configuration

### `config.yaml` — Global Settings

```yaml
pipelines:
  structured:
    entry_point: stuctured_pipeline/run_inference.py
  unstructured:
    entry_point: unstructured_pipeline/main.py

output:
  combined_dir: Output/

logging:
  level: INFO
```

### `agents/agent_config.yaml` — Agent Weights & Thresholds

```yaml
agents:
  altman_zscore:
    enabled: true
    weight: 0.20      # contribution to combined agent score
    safe_zone: 2.99

  tax_rate_anomaly:
    enabled: true
    weight: 0.08
```

Set `enabled: false` to disable any agent. Weights are automatically normalized across agents that successfully run.

---

## 🎯 Risk Level Interpretation

| Score | Level | Recommended Action |
|-------|-------|--------------------|
| 0–19 | **MINIMAL** | Routine processing |
| 20–39 | **LOW** | Periodic monitoring |
| 40–59 | **MEDIUM** | Detailed review |
| 60–79 | **HIGH** | Formal assessment + alerts |
| 80–100 | **CRITICAL** | Immediate investigation |

---

## ⚡ Performance

| Component | Mode | Time |
|-----------|------|------|
| Structured Pipeline | ML inference | ~12s |
| Unstructured Pipeline | ChromaDB retrieval | ~2–4s |
| Agent Orchestrator | 15 agents | ~1s |
| Score Combination | Processing | <0.1s |
| **Total** | **End-to-end** | **~15–18s** |

---

## 🔧 Requirements

- **Python**: 3.8+
- **RAM**: 8 GB+ (ML models)
- **Disk**: 15 GB+ (models + ChromaDB)

```
chromadb>=0.4.0
neo4j>=5.0.0            # optional
sentence-transformers
catboost, xgboost, lightgbm
pandas, numpy, scikit-learn
torch
pyyaml
```

Full list → `requirements.txt` | Install → `bash setup.sh`

---

## 📂 Sub-Folder Documentation

| Folder | README |
|--------|--------|
| `agents/` | [`agents/README.md`](agents/README.md) — all 15 agents, base class, config |
| `stuctured_pipeline/` | [`stuctured_pipeline/README.md`](stuctured_pipeline/README.md) — 17 models, feature engineering |
| `unstructured_pipeline/` | [`unstructured_pipeline/README.md`](unstructured_pipeline/README.md) — ChromaDB, risk scoring |

---

## 🔍 Troubleshooting

| Issue | Solution |
|-------|----------|
| Agents show "Data not applicable" | Ensure raw input JSON is in `Input/` — the runner merges it automatically |
| `0 documents retrieved` (unstructured) | CIK not in ChromaDB — ingest documents first |
| `Neo4j warnings` | Normal — system falls back to ChromaDB automatically |
| `Model file not found` | Check `stuctured_pipeline/MyModels/` exists with trained files |
| `LOF: Expected n_neighbors` error | Normal for single-sample inference — LOF result is excluded |

---

**Version**: 3.0 — 15-Agent Multi-Layer Detection System  
**Last Updated**: March 2026
