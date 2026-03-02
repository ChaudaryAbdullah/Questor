# Unified Fraud Detection Pipeline

## Overview

This unified architecture allows you to run both **structured** and **unstructured** fraud detection pipelines from a single entry point, combine their risk scores, and prepare outputs for multi-agent system integration.

## Quick Start

### Run Both Pipelines

```bash
cd /home/cypher/Questor/Pipelines
python unified_runner.py --pipeline both --limit 5
```

### Run Individual Pipelines

```bash
# Structured only (processes JSON files from Input directory)
python unified_runner.py --pipeline structured --input stuctured_pipeline/Input/

# Unstructured only (processes documents from data)
python unified_runner.py --pipeline unstructured --limit 10
```

### Combine Existing Outputs

```bash
python score_combiner.py \
  --structured output/structured/latest.json \
  --unstructured output/unstructured/latest.json \
  --output output/combined/my_analysis.json
```

## Architecture

```
Pipelines/
├── unified_runner.py          # Main orchestrator
├── score_combiner.py           # Score combination logic
├── config.yaml                 # Configuration
│
├── shared/                     # Shared infrastructure
│   ├── __init__.py
│   ├── output_schema.py        # Standardized output format
│   └── utils.py                # Common utilities
│
├── output/                     # All outputs
│   ├── structured/             # Structured pipeline outputs
│   ├── unstructured/           # Unstructured pipeline outputs
│   ├── combined/               # Combined risk scores
│   └── multiagent_ready/       # Formatted for multi-agent system
│
├── stuctured_pipeline/         # Structured pipeline (unchanged)
│   ├── run_inference.py        # Entry point
│   └── ...
│
└── unstructured_pipeline/      # Unstructured pipeline (unchanged)
    ├── main.py                 # Entry point
    └── ...
```

## How It Works

### 1. Pipeline Execution

**Unified Runner** (`unified_runner.py`) orchestrates:
- Runs structured pipeline via `stuctured_pipeline/run_inference.py`
- Runs unstructured pipeline via `unstructured_pipeline/main.py`
- Collects outputs from both

### 2. Score Combination

**Score Combiner** (`score_combiner.py`) combines outputs:

```python
final_score = (structured_score × 0.6) + (unstructured_score × 0.4)
```

**Default Weights:**
- Structured: 60% (tabular data typically more reliable)
- Unstructured: 40% (text analysis provides context)

**Conflict Handling:**
- If scores differ by >30 points → Flag for review
- If only one pipeline has data → Apply 0.8 penalty
- If risk levels conflict → Take the higher level

### 3. Multi-Agent Preparation

Combined outputs include:

```json
{
  "record_id": "doc_123",
  "combined_risk": {
    "overall_risk_score": 72.5,
    "risk_level": "HIGH",
    "confidence": 0.95
  },
  "recommended_agents": [
    "fraud_detection_agent",
    "risk_assessment_agent"
  ],
  "priority": "high",
  "requires_investigation": false
}
```

## Configuration

Edit `config.yaml` to customize:

```yaml
score_combination:
  structured_weight: 0.6      # Adjust weights
  unstructured_weight: 0.4
  conflict_threshold: 30       # Flag threshold
  missing_penalty: 0.8         # Single-source penalty
```

## Output Format

### Unified Output Schema

Each record includes:

| Field | Description |
|-------|-------------|
| `record_id` | Unique identifier |
| `structured_risk` | Risk from structured pipeline |
| `unstructured_risk` | Risk from unstructured pipeline |
| `combined_risk` | **Final combined risk score** |
| `fraud_prediction` | Binary prediction (0/1) |
| `fraud_probability` | Probability (0-1) |
| `entities` | Extracted entities (unstructured) |
| `relationships` | Entity relationships (unstructured) |
| `recommended_agents` | **Suggested agents for processing** |
| `priority` | `critical`, `high`, `normal`, or `low` |
| `requires_investigation` | Boolean flag for score ≥80 |

### Combined Risk Assessment

```json
{
  "overall_risk_score": 68.4,
  "risk_level": "HIGH",
  "component_scores": {
    "structured_score": 75.0,
    "unstructured_score": 58.0,
    "weighted_combined": 68.4,
    "score_difference": 17.0
  },
  "risk_factors": [
    "[Structured] High fraud probability: 85%",
    "[Structured] Strong model consensus: 80% agreement",
    "[Unstructured] High-risk keywords detected: fictitious, manipulation",
    "[Unstructured] Large financial amounts: 3 transactions"
  ],
  "confidence": 0.83
}
```

## Agent Routing Logic

Automatic agent recommendations based on risk score:

| Score | Risk Level | Agents | Priority |
|-------|-----------|--------|----------|
| ≥80 | CRITICAL | fraud_investigation_agent, alert_agent, compliance_agent | critical |
| 60-79 | HIGH | fraud_detection_agent, risk_assessment_agent, compliance_agent | high |
| 40-59 | MEDIUM | risk_assessment_agent, pattern_analysis_agent | normal |
| 20-39 | LOW | general_analysis_agent, statistical_analysis_agent | low |
| <20 | MINIMAL | general_analysis_agent, statistical_analysis_agent | low |

## Usage Examples

### Example 1: Full Analysis with Limited Data

```bash
python unified_runner.py \
  --pipeline both \
  --limit 10 \
  --batch-name "test_analysis_001"
```

### Example 2: Structured Only with Custom Input

```bash
python unified_runner.py \
  --pipeline structured \
  --input stuctured_pipeline/Input/ \
  --batch-name "tabular_analysis"
```

### Example 3: Unstructured Only with Batch Processing

```bash
python unified_runner.py \
  --pipeline unstructured \
  --limit 20 \
  --batch-size 5 \
  --batch-name "document_analysis"
```

### Example 4: Post-Processing Score Combination

```bash
# Run pipelines separately first
python stuctured_pipeline/run_inference.py
python unstructured_pipeline/main.py --export-output batch_1

# Then combine
python score_combiner.py \
  --structured output/structured/latest.json \
  --unstructured unstructured_pipeline/output/batch_1*.json \
  --batch-name "manual_combination"
```

## Integration with Multi-Agent System

The unified output is designed for seamless multi-agent integration:

### 1. Read Combined Output

```python
import json

with open('output/combined/combined_analysis_*.json', 'r') as f:
    data = json.load(f)

for record in data['records']:
    # Get recommended agents
    agents = record['recommended_agents']
    priority = record['priority']
    
    # Route to appropriate agent
    if 'fraud_investigation_agent' in agents:
        # Send to investigation
        pass
```

### 2. Filter by Priority

```python
critical_cases = [
    r for r in data['records']
    if r['priority'] == 'critical'
]
```

### 3. Access Full Context

```python
record = data['records'][0]

# Structured data
prediction = record['fraud_prediction']
probability = record['fraud_probability']

# Unstructured data
entities = record['entities']
relationships = record['relationships']

# Combined assessment
risk_score = record['combined_risk']['overall_risk_score']
risk_factors = record['combined_risk']['risk_factors']
```

## Troubleshooting

### Issue: No output generated

**Solution:** Check that input data exists:
- Structured: `stuctured_pipeline/Input/*.json`
- Unstructured: Document sources configured in `unstructured_pipeline/`

### Issue: Import errors

**Solution:** Ensure you're running from Pipelines directory:
```bash
cd /home/cypher/Questor/Pipelines
python unified_runner.py --help
```

### Issue: Score combination fails

**Solution:** Verify outputs match expected schema:
```bash
python -c "import json; print(json.load(open('output/structured/latest.json')))"
```

## Original Pipeline Entry Points

Both pipelines remain fully functional independently:

```bash
# Structured pipeline (original)
cd stuctured_pipeline
python run_inference.py Input/

# Unstructured pipeline (original)
cd unstructured_pipeline
python main.py --export-output test
```

## Future Multi-Agent Implementation

When implementing your multi-agent system:

1. **Read from** `output/combined/` or `output/multiagent_ready/`
2. **Use** `recommended_agents` field for routing
3. **Prioritize** based on `priority` field
4. **Investigate** records where `requires_investigation == true`
5. **Access context** from both `structured_risk` and `unstructured_risk`

## Next Steps

1. ✅ Run a test with limited data
2. ✅ Verify combined outputs
3. ⏭️ Implement multi-agent system (your next phase)
4. ⏭️ Fine-tune weights in `config.yaml` based on results
