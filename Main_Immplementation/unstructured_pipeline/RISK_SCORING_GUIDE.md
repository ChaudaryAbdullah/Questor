# Risk Scoring and Multiagent Output Integration

## Overview

This document describes the new **Risk Scoring** and **Multiagent Output Formatting** features added to the unstructured pipeline. These features enable the pipeline to:

1. **Calculate fraud risk scores** for each document based on multiple factors
2. **Format structured output** for consumption by downstream multiagent systems
3. **Generate comprehensive reports** with risk assessments and actionable insights

## New Components

### 1. Risk Scorer (`pipelines/risk_scorer.py`)

The `RiskScorer` class calculates comprehensive fraud risk scores (0-100) for each document based on:

#### Risk Components (Weighted):

- **Fraud Indicators (35%)**: Detection of high-risk keywords and patterns
  - Severe: fictitious, fabricated, concealment, material weakness (0.8-1.0)
  - High: overstatement, round-trip, special purpose entity (0.6-0.8)
  - Medium: derivative, restructuring, write-off (0.4-0.6)
  - Low: contingency, subsidiary, revenue recognition (0.2-0.4)

- **Entity Risk (25%)**: Analysis of extracted entities
  - Financial term density
  - Fraud indicator count
  - Organization complexity
  - Monetary transaction volume

- **Financial Anomalies (25%)**: Pattern detection in financial data
  - Large monetary amounts (>$100M, >$10M, >$1M)
  - Complex transaction patterns
  - Temporal anomalies

- **Relationship Risk (15%)**: Suspicious entity relationships
  - Concealment relationships (0.9)
  - Transfer relationships (0.7)
  - Ownership/control patterns (0.4-0.5)

#### Risk Levels:

- **CRITICAL**: Score ≥ 80 (Immediate investigation required)
- **HIGH**: Score ≥ 60 (Priority review needed)
- **MEDIUM**: Score ≥ 40 (Standard review)
- **LOW**: Score ≥ 20 (Monitoring)
- **MINIMAL**: Score < 20 (Low concern)

### 2. Output Formatter (`pipelines/output_formatter.py`)

The `OutputFormatter` class structures pipeline data for multiagent system consumption.

#### Formatted Output Structure:

```json
{
  "document_id": "string",
  "timestamp": "ISO-8601",
  "source_file": "path/to/file",
  
  "metadata": {
    "label": "Fraud|NonFraud|Unknown",
    "company_id": "string",
    "date": "string",
    "content_length": 0,
    "chunk_count": 0
  },
  
  "risk_assessment": {
    "overall_score": 0.0,
    "risk_level": "CRITICAL|HIGH|MEDIUM|LOW|MINIMAL",
    "component_scores": {
      "fraud_indicators": 0.0,
      "entity_risk": 0.0,
      "financial_anomalies": 0.0,
      "relationship_risk": 0.0
    },
    "risk_factors": ["list of identified risk factors"],
    "requires_investigation": true|false
  },
  
  "extracted_data": {
    "entities": {
      "ORG": [...],
      "PERSON": [...],
      "MONEY": [...],
      "FINANCIAL_TERM": [...],
      "FRAUD_INDICATOR": [...]
    },
    "relationships": [...],
    "entity_summary": {...}
  },
  
  "retrieval_references": {
    "vector_db_chunks": ["chunk_id_1", "chunk_id_2"],
    "knowledge_graph_nodes": ["node_id_1", "node_id_2"],
    "embedding_model": "model_name"
  },
  
  "agent_routing": {
    "recommended_agents": ["agent_1", "agent_2"],
    "priority": "critical|high|normal",
    "processing_hints": ["hint_1", "hint_2"]
  }
}
```

#### Agent Routing Logic:

The formatter automatically determines which agents should process each document:

- **fraud_investigation_agent**: Risk score ≥ 80
- **risk_assessment_agent**: Risk score ≥ 60
- **nlp_disclosure_agent**: Complex financial language (>5 financial terms)
- **graph_linkage_agent**: Complex relationships (>3 relationships)
- **fraud_detection_agent**: Fraud indicators present
- **general_analysis_agent**: Default fallback

## Usage

### Basic Usage with Risk Scoring

```bash
# Process documents with risk scoring enabled (default)
python main.py --limit 100

# Process and show risk summary
python main.py --limit 100 --risk-summary

# Process and export formatted output for multiagent system
python main.py --limit 100 --export-output my_batch_name

# Process with both summary and export
python main.py --limit 100 --risk-summary --export-output fraud_analysis_batch1
```

### Advanced Options

```bash
# Disable risk scoring for faster processing
python main.py --limit 100 --disable-risk-scoring

# Process with custom batch size and risk analysis
python main.py --batch-size 20 --limit 500 --risk-summary --export-output large_batch

# Skip embeddings but keep risk scoring
python main.py --skip-embeddings --limit 100 --risk-summary

# Full pipeline with all features
python main.py --limit 1000 --batch-size 50 --risk-summary --export-output full_analysis --memory-monitor
```

### Command-Line Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--limit` | int | Limit number of documents to process |
| `--batch-size` | int | Documents per batch (default: 10) |
| `--skip-embeddings` | flag | Skip vector embedding generation |
| `--skip-graph` | flag | Skip knowledge graph construction |
| `--disable-risk-scoring` | flag | Disable risk scoring (faster) |
| `--risk-summary` | flag | Show risk score summary after processing |
| `--export-output` | str | Export formatted output (provide batch name) |
| `--memory-monitor` | flag | Enable memory monitoring |
| `--reset` | flag | Reset pipeline (delete all data) |
| `--status` | flag | Show pipeline status |
| `--query` | str | Query the vector database |

## Output Files

### 1. Formatted JSON Output

**Location**: `output/[batch_name]_[timestamp].json`

Contains complete structured data for all processed documents, including:
- Document metadata
- Risk assessments
- Extracted entities and relationships
- Retrieval references
- Agent routing recommendations

### 2. Summary Report

**Location**: `output/[batch_name]_summary.txt`

Human-readable text report containing:
- Overall statistics
- Risk distribution
- Top high-risk documents
- Risk factors summary

### Example Summary Report:

```
========================================
FRAUD DETECTION PIPELINE - SUMMARY REPORT
========================================
Generated: 2026-02-06 12:30:00

OVERALL STATISTICS
------------------
Total Documents Processed: 100
Average Risk Score: 45.32
Maximum Risk Score: 92.15
Minimum Risk Score: 8.50

RISK DISTRIBUTION
-----------------
CRITICAL    :    5 documents
HIGH        :   15 documents
MEDIUM      :   35 documents
LOW         :   30 documents
MINIMAL     :   15 documents

HIGH-RISK DOCUMENTS: 20
========================================

TOP HIGH-RISK DOCUMENTS:
----------------------------------------

1. Document: Fraud_2020_Company_XYZ
   Risk Score: 92.15 (CRITICAL)
   Risk Factors:
   - High-risk keywords detected: concealment, fabricated, manipulation
   - Fraud indicators found: 12 instances
   - Large financial amounts: 8 transactions

2. Document: Unknown_2019_ABC_Corp
   Risk Score: 87.30 (CRITICAL)
   Risk Factors:
   - High-risk keywords detected: special purpose entity, round-trip
   - Complex financial structures: 15 terms
   - Suspicious relationships: 6 connections
...
```

## Integration with Multiagent System

### Step 1: Run Pipeline with Export

```bash
python main.py --limit 1000 --export-output fraud_batch_001 --risk-summary
```

### Step 2: Load Formatted Output in Multiagent System

```python
import json

# Load the formatted output
with open('output/fraud_batch_001_20260206_123000.json', 'r') as f:
    batch_data = json.load(f)

# Access documents
documents = batch_data['documents']

# Get high-risk documents
high_risk = batch_data['high_risk_documents']

# Process each document
for doc in documents:
    # Get risk assessment
    risk = doc['risk_assessment']
    
    # Route to appropriate agents
    agents = doc['agent_routing']['recommended_agents']
    priority = doc['agent_routing']['priority']
    
    # Access extracted data
    entities = doc['extracted_data']['entities']
    relationships = doc['extracted_data']['relationships']
    
    # Get retrieval references for RAG
    vector_chunks = doc['retrieval_references']['vector_db_chunks']
    graph_nodes = doc['retrieval_references']['knowledge_graph_nodes']
```

### Step 3: Use Risk Scores for Prioritization

```python
# Filter critical documents
critical_docs = [
    doc for doc in documents
    if doc['risk_assessment']['overall_score'] >= 80
]

# Sort by risk score
sorted_docs = sorted(
    documents,
    key=lambda x: x['risk_assessment']['overall_score'],
    reverse=True
)

# Get documents requiring investigation
investigation_needed = [
    doc for doc in documents
    if doc['risk_assessment']['requires_investigation']
]
```

## Programmatic Usage

### Using Risk Scorer Directly

```python
from pipelines.risk_scorer import RiskScorer

# Initialize scorer
scorer = RiskScorer()

# Calculate risk for a document
risk_data = scorer.calculate_document_risk(
    document={'doc_id': 'test_doc', 'content': 'document text...'},
    entities={'FRAUD_INDICATOR': [...], 'ORG': [...]},
    relationships=[...]
)

print(f"Risk Score: {risk_data['overall_risk_score']}")
print(f"Risk Level: {risk_data['risk_level']}")
print(f"Risk Factors: {risk_data['risk_factors']}")
```

### Using Output Formatter Directly

```python
from pipelines.output_formatter import OutputFormatter

# Initialize formatter
formatter = OutputFormatter()

# Format single document
formatted = formatter.format_for_multiagent(
    document=doc,
    risk_data=risk_data,
    entities=entities,
    relationships=relationships,
    chunks=chunks
)

# Save to file
formatter.save_to_json(formatted, 'single_doc_output.json')
```

## Performance Considerations

### Risk Scoring Impact:

- **Processing Time**: Adds ~10-15% to overall pipeline time
- **Memory Usage**: Minimal additional memory (<100MB)
- **Accuracy**: Based on pattern matching and statistical analysis

### Optimization Tips:

1. **Disable for Speed**: Use `--disable-risk-scoring` for faster processing when risk scores aren't needed
2. **Batch Size**: Larger batches (20-50) are more efficient for risk scoring
3. **Selective Export**: Only export when needed to save disk space

## Future Enhancements

Potential improvements for future versions:

1. **Machine Learning Models**: Replace rule-based scoring with ML models
2. **Historical Analysis**: Compare risk scores across time periods
3. **Custom Risk Weights**: Allow users to configure component weights
4. **Real-time Scoring**: API endpoint for on-demand risk assessment
5. **Visualization**: Generate risk score charts and graphs
6. **Alert System**: Automatic notifications for critical risk documents

## Troubleshooting

### Issue: No formatted outputs generated

**Solution**: Ensure risk scoring is enabled (don't use `--disable-risk-scoring`)

### Issue: Export fails with "No formatted outputs"

**Solution**: Run the pipeline first, then export in the same execution

### Issue: Risk scores seem too high/low

**Solution**: Risk weights can be adjusted in `pipelines/risk_scorer.py` - modify the `WEIGHTS` dictionary

### Issue: Out of memory during risk scoring

**Solution**: Reduce batch size with `--batch-size 5`

## Contact & Support

For questions or issues related to risk scoring and output formatting:
- Check the logs in `logs/` directory
- Review the pipeline status with `python main.py --status`
- Examine sample outputs in `output/` directory
