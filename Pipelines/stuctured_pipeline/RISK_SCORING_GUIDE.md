# Structured Pipeline - Risk Scoring Quick Start

## What's New

Your structured pipeline now includes:
- ✅ **Risk Scoring** - Calculates fraud risk scores (0-100) based on model predictions
- ✅ **Multiagent Output** - Formats data for downstream multiagent system
- ✅ **Enhanced Reports** - Includes risk assessments and agent routing

## Quick Start

### Option 1: Run with Risk Scoring (Recommended)

```bash
cd "d:\FAST NUCES\FYP\Questor\Pipelines\stuctured_pipeline"

# Run with risk scoring and export for multiagent system
python structured_pipeline_with_risk.py --export --batch-name fraud_analysis_001
```

### Option 2: Run Original Pipeline (No Risk Scoring)

```bash
# Original pipeline still works
python structured_pipeline.py
```

## Command-Line Options

| Option | Description |
|--------|-------------|
| `--input [path]` | Input CSV file (default: ../Dataset/enhanced_financial_data.csv) |
| `--export` | Export formatted output for multiagent system |
| `--batch-name [name]` | Name for output files (default: structured_risk_analysis) |
| `--disable-risk-scoring` | Disable risk scoring for faster processing |

## Example Commands

```bash
# Basic run with risk scoring
python structured_pipeline_with_risk.py

# Full run with export
python structured_pipeline_with_risk.py --export --batch-name production_batch_001

# Fast run without risk scoring
python structured_pipeline_with_risk.py --disable-risk-scoring

# Custom input file
python structured_pipeline_with_risk.py --input my_data.csv --export
```

## Output Files

### 1. Multiagent JSON Output
**File**: `output/[batch_name]_[timestamp].json`

Contains:
- Record-level predictions
- Risk assessments
- Agent routing recommendations
- Model probabilities

### 2. Predictions CSV
**File**: `[batch_name]_predictions.csv`

Contains:
- All input features
- Fraud predictions
- Fraud probabilities
- Risk scores and levels
- Investigation flags

### Example Output Structure

```json
{
  "batch_metadata": {
    "batch_name": "fraud_analysis_001",
    "timestamp": "2026-02-06T14:00:00",
    "total_records": 1000,
    "pipeline": "structured"
  },
  "records": [
    {
      "record_id": "structured_0",
      "prediction": {
        "ensemble_prediction": 1,
        "ensemble_probability": 0.8542,
        "individual_predictions": {
          "xgboost": 1,
          "random_forest": 1,
          "dnn": 1
        }
      },
      "risk_assessment": {
        "overall_risk_score": 78.45,
        "risk_level": "HIGH",
        "component_scores": {
          "model_confidence": 85.42,
          "model_agreement": 100.0,
          "prediction_strength": 82.15,
          "anomaly_indicators": 15.30
        },
        "requires_investigation": false
      },
      "agent_routing": {
        "recommended_agents": [
          "fraud_detection_agent",
          "risk_assessment_agent",
          "statistical_analysis_agent"
        ],
        "priority": "high"
      }
    }
  ]
}
```

## Risk Scoring Details

### Risk Components (Weighted)

1. **Model Confidence (40%)** - Ensemble fraud probability
2. **Model Agreement (30%)** - Consensus among models
3. **Prediction Strength (20%)** - Average probability with low variance
4. **Anomaly Indicators (10%)** - Statistical anomalies in features

### Risk Levels

- **CRITICAL (80-100)**: Immediate investigation required
- **HIGH (60-79)**: Priority review needed
- **MEDIUM (40-59)**: Standard review
- **LOW (20-39)**: Monitoring
- **MINIMAL (0-19)**: Low concern

### Agent Routing

Based on risk level, records are automatically routed to appropriate agents:

| Risk Level | Agents | Priority |
|------------|--------|----------|
| CRITICAL (≥80) | fraud_investigation_agent, alert_agent | critical |
| HIGH (≥60) | fraud_detection_agent, risk_assessment_agent | high |
| MEDIUM (≥40) | risk_assessment_agent | normal |
| LOW (<40) | general_analysis_agent | low |

## Integration with Multiagent System

### Step 1: Run Pipeline

```bash
python structured_pipeline_with_risk.py --export --batch-name production_001
```

### Step 2: Load in Multiagent System

```python
import json

# Load the output
with open('output/production_001_20260206_140000.json', 'r') as f:
    data = json.load(f)

# Process each record
for record in data['records']:
    risk_score = record['risk_assessment']['overall_risk_score']
    risk_level = record['risk_assessment']['risk_level']
    agents = record['agent_routing']['recommended_agents']
    priority = record['agent_routing']['priority']
    
    # Route to appropriate agents
    if priority == 'critical':
        # Send to fraud investigation immediately
        pass
    elif priority == 'high':
        # Queue for priority review
        pass
```

## Performance

| Mode | Processing Time (1000 records) | Notes |
|------|-------------------------------|-------|
| With risk scoring | ~30-45 seconds | Full analysis |
| Without risk scoring | ~20-30 seconds | Predictions only |

## Comparison: Unstructured vs Structured Pipelines

| Feature | Unstructured Pipeline | Structured Pipeline |
|---------|----------------------|---------------------|
| **Input** | Text documents (PDF, TXT) | Tabular data (CSV) |
| **Processing** | NER, embeddings, graph | Ensemble ML models |
| **Risk Factors** | Keywords, entities, relationships | Model agreement, probabilities |
| **Output** | Document-level analysis | Record-level predictions |
| **Use Case** | Disclosure analysis | Financial metrics analysis |

## Next Steps

1. **Test the Pipeline**
   ```bash
   python structured_pipeline_with_risk.py --export --batch-name test_run
   ```

2. **Review Outputs**
   - Check `test_run_predictions.csv` for predictions
   - Check `output/test_run_*.json` for multiagent data

3. **Integrate with Multiagent System**
   - Use the JSON output in your multiagent system
   - Route records based on risk levels and agent recommendations

4. **Monitor Performance**
   - Track risk score distributions
   - Analyze model agreement patterns
   - Review high-risk cases

## Troubleshooting

### Issue: Models not loading

**Solution**: Ensure models are in `MyModels/` directory with correct structure:
```
MyModels/
├── xgboost_YYYY-MM-DD_HH-MM-SS/
│   ├── model.pkl
│   ├── scaler.pkl
│   └── metrics.txt
```

### Issue: Input features mismatch

**Solution**: Ensure input CSV has same features as training data

### Issue: Risk scores all zero

**Solution**: Check that `--disable-risk-scoring` is not set

## Summary

**You now have:**
- ✅ Risk scoring for structured pipeline
- ✅ Multiagent output formatting
- ✅ Unified output format with unstructured pipeline
- ✅ Agent routing recommendations
- ✅ Comprehensive risk assessments

**Both pipelines are ready to feed into your multiagent system!** 🎉
