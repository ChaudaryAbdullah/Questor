# Unified Fraud Detection Pipeline

A comprehensive fraud detection system that combines **structured** (tabular data) and **unstructured** (text/document) analysis pipelines with unified risk scoring for multi-agent integration.

## 🚀 Quick Start

### 1. Setup Environment

```bash
cd /home/cypher/Questor/Pipelines
./setup.sh
```

### 2. Activate Virtual Environment

```bash
source venv/bin/activate
```

### 3. Run Both Pipelines

```bash
python unified_runner.py --pipeline both --limit 5
```

## 📁 Structure

```
Pipelines/
├── unified_runner.py          # 🎯 Main entry point
├── score_combiner.py           # 🔗 Score combination logic
├── config.yaml                 # ⚙️ Configuration
│
├── shared/                     # 📦 Shared infrastructure
│   ├── output_schema.py        # Standardized output format
│   └── utils.py                # Common utilities
│
├── output/                     # 📊 All outputs
│   ├── structured/             # Structured pipeline results
│   ├── unstructured/           # Unstructured pipeline results
│   ├── combined/               # Combined risk scores
│   └── multiagent_ready/       # Multi-agent formatted output
│
├── stuctured_pipeline/         # 🔷 Structured (tabular) pipeline
│   └── run_inference.py        # Entry point
│
└── unstructured_pipeline/      # 🔶 Unstructured (text) pipeline
    └── main.py                 # Entry point
```

## 📖 Key Features

### ✅ Unified Execution
- Run both pipelines from one command
- Automatic output collection and organization
- Configurable pipeline selection

### ✅ Smart Score Combination
```
final_score = (structured × 0.6) + (unstructured × 0.4)
```
- Weighted average of both risk scores
- Conflict detection (flags if scores differ >30 points)
- Confidence penalties for single-source data

### ✅ Multi-Agent Ready
- Standardized JSON output format
- Automatic agent routing recommendations
- Priority flagging (critical/high/normal/low)
- Investigation flags for high-risk cases

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [SETUP.md](SETUP.md) | **Installation & environment setup** |
| [UNIFIED_PIPELINE_GUIDE.md](UNIFIED_PIPELINE_GUIDE.md) | **Complete usage guide & examples** |
| [config.yaml](config.yaml) | Configuration reference |

## 🎯 Usage Examples

### Run Both Pipelines

```bash
python unified_runner.py --pipeline both --limit 10
```

### Run Structured Only

```bash
python unified_runner.py --pipeline structured --input stuctured_pipeline/Input/
```

### Run Unstructured Only

```bash
python unified_runner.py --pipeline unstructured --limit 5 --batch-size 2
```

### Combine Existing Outputs

```bash
python score_combiner.py \
  --structured output/structured/latest.json \
  --unstructured output/unstructured/latest.json \
  --batch-name my_analysis
```

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
score_combination:
  structured_weight: 0.6      # Tabular data weight
  unstructured_weight: 0.4    # Text analysis weight
  conflict_threshold: 30       # Flag if scores differ by this amount
  missing_penalty: 0.8         # Reduce score if only one source
```

## 📊 Output Format

Each record includes:

```json
{
  "record_id": "doc_123",
  "combined_risk": {
    "overall_risk_score": 72.5,
    "risk_level": "HIGH",
    "confidence": 0.95,
    "risk_factors": [
      "[Structured] High fraud probability: 85%",
      "[Unstructured] Keywords detected: manipulation, concealment"
    ]
  },
  "recommended_agents": ["fraud_detection_agent", "compliance_agent"],
  "priority": "high",
  "requires_investigation": false
}
```

## 🔧 Requirements

- Python 3.8+
- 4GB+ RAM
- 10GB disk space
- See [requirements.txt](requirements.txt) for packages

## 🏃 Running Individual Pipelines

Both pipelines remain fully functional independently:

```bash
# Structured pipeline (original)
cd stuctured_pipeline
python run_inference.py Input/

# Unstructured pipeline (original)
cd unstructured_pipeline
python main.py --export-output test_batch
```

## 🤖 Multi-Agent Integration

The unified output is designed for multi-agent systems:

```python
import json

# Load combined output
with open('output/combined/combined_*.json', 'r') as f:
    data = json.load(f)

# Route based on recommendations
for record in data['records']:
    agents = record['recommended_agents']
    priority = record['priority']
    
    # Route to appropriate agents
    if 'fraud_investigation_agent' in agents:
        # High priority investigation
        pass
```

## 📈 Score Combination Logic

### Risk Levels
- **CRITICAL** (≥80): Fraud investigation + alerts
- **HIGH** (60-79): Detailed risk assessment
- **MEDIUM** (40-59): Pattern analysis
- **LOW** (20-39): General monitoring
- **MINIMAL** (<20): Routine processing

### Agent Routing
Automatically recommends agents based on risk:
- Critical → `fraud_investigation_agent`, `alert_agent`, `compliance_agent`
- High → `fraud_detection_agent`, `risk_assessment_agent`
- Medium → `risk_assessment_agent`, `pattern_analysis_agent`
- Low → `general_analysis_agent`, `statistical_analysis_agent`

## 🔍 Troubleshooting

See [SETUP.md#Troubleshooting](SETUP.md#troubleshooting) for common issues.

## 📝 License

[Your License]

## 👥 Contributing

[Your Contributing Guidelines]
