# Calculate Risk Scores from Existing Data

## Overview

**Good news!** You don't need to reprocess your entire dataset. This script calculates risk scores from your **existing data in Neo4j and ChromaDB**.

## Quick Start

### Basic Usage (Risk Scores Only)

```bash
# Calculate risk scores for all documents in Neo4j
python calculate_risk_from_existing_data.py

# Calculate for limited number of documents
python calculate_risk_from_existing_data.py --limit 100
```

### Export for Multiagent System

```bash
# Calculate risk scores AND export formatted output
python calculate_risk_from_existing_data.py --export --batch-name my_risk_analysis

# Full export with chunks from ChromaDB (slower but complete)
python calculate_risk_from_existing_data.py --export --batch-name full_analysis --include-chunks
```

## Command-Line Options

| Option | Description |
|--------|-------------|
| `--export` | Export formatted output for multiagent system |
| `--batch-name [name]` | Name for the batch export (default: existing_data_risk_analysis) |
| `--include-chunks` | Include text chunks from ChromaDB (slower) |
| `--limit [N]` | Process only first N documents |

## What This Script Does

1. **Retrieves documents from Neo4j** - Gets all documents with their entities and relationships
2. **Calculates risk scores** - Uses the same risk scoring algorithm as the main pipeline
3. **Optionally retrieves chunks** - Gets text chunks from ChromaDB if needed
4. **Formats output** - Creates structured JSON for your multiagent system
5. **Generates reports** - Creates summary reports with risk statistics

## Example Workflow

### Step 1: Calculate Risk Scores

```bash
python calculate_risk_from_existing_data.py --limit 100
```

**Output:**
```
================================================================================
CALCULATING RISK SCORES FROM EXISTING DATA
================================================================================
Retrieving documents from Neo4j...
Retrieved 100 documents from Neo4j
Calculating risk scores for 100 documents...
Successfully calculated risk for 100 documents

================================================================================
RISK SCORE SUMMARY
================================================================================
Total documents: 100
Average risk score: 45.32
Max risk score: 92.15
Min risk score: 8.50

Risk Level Distribution:
  CRITICAL: 5
  HIGH: 15
  MEDIUM: 35
  LOW: 30
  MINIMAL: 15

High-risk documents: 20

Top 10 High-Risk Documents:
  1. Fraud_2020_XYZ: 92.15 (CRITICAL)
  2. Unknown_2019_ABC: 87.30 (CRITICAL)
  ...
================================================================================
```

### Step 2: Export for Multiagent System

```bash
python calculate_risk_from_existing_data.py --export --batch-name fraud_analysis_2026
```

**Output Files:**
- `output/fraud_analysis_2026_[timestamp].json` - Complete formatted data
- `output/fraud_analysis_2026_summary.txt` - Human-readable summary

## Performance

### Speed Comparison

| Mode | Processing Time (1000 docs) | Notes |
|------|----------------------------|-------|
| Without chunks | ~2-5 minutes | Fast, uses only Neo4j data |
| With chunks | ~5-10 minutes | Slower, retrieves from ChromaDB |
| Full reprocessing | ~60-90 minutes | Processes raw files again |

**Recommendation:** Use without chunks unless you specifically need the chunk data.

## Data Retrieved

### From Neo4j:
- Document metadata (doc_id, label, company_id, date)
- Document content (if stored)
- Extracted entities (ORG, PERSON, MONEY, FINANCIAL_TERM, etc.)
- Relationships between entities

### From ChromaDB (optional):
- Text chunks
- Chunk metadata
- Embeddings (not used for risk scoring)

## Output Format

Same as the main pipeline - fully compatible with your multiagent system:

```json
{
  "batch_metadata": {...},
  "summary_statistics": {...},
  "documents": [
    {
      "document_id": "...",
      "risk_assessment": {
        "overall_score": 75.5,
        "risk_level": "HIGH",
        "component_scores": {...},
        "risk_factors": [...],
        "requires_investigation": true
      },
      "extracted_data": {
        "entities": {...},
        "relationships": [...],
        "entity_summary": {...}
      },
      "retrieval_references": {...},
      "agent_routing": {...}
    }
  ],
  "high_risk_documents": [...]
}
```

## Use Cases

### 1. Initial Risk Assessment
You've already processed your data, now you want risk scores:
```bash
python calculate_risk_from_existing_data.py --export --batch-name initial_assessment
```

### 2. Re-calculate After Tuning
You've adjusted risk weights, recalculate without reprocessing:
```bash
# Edit weights in pipelines/risk_scorer.py
python calculate_risk_from_existing_data.py --export --batch-name tuned_analysis
```

### 3. Subset Analysis
Analyze a subset for testing:
```bash
python calculate_risk_from_existing_data.py --limit 50 --export --batch-name test_batch
```

### 4. Quick Summary
Just want to see risk distribution:
```bash
python calculate_risk_from_existing_data.py --limit 1000
```

## Programmatic Usage

You can also use this in your own Python code:

```python
from calculate_risk_from_existing_data import ExistingDataRiskCalculator

# Initialize
calculator = ExistingDataRiskCalculator()

# Run analysis
result = calculator.run(
    export_output=True,
    batch_name='my_analysis',
    include_chunks=False,
    limit=100
)

# Access results
print(f"Processed: {result['documents_processed']}")
print(f"Output: {result['output_path']}")
print(f"Summary: {result['summary']}")

# Cleanup
calculator.close()
```

## Advantages Over Reprocessing

✅ **Much faster** - 2-10 minutes vs 60-90 minutes  
✅ **No file I/O** - Works directly with databases  
✅ **No model loading** - Doesn't need NER/embedding models  
✅ **Same results** - Uses identical risk scoring algorithm  
✅ **Flexible** - Can run multiple times with different settings  

## Limitations

⚠️ **Requires existing data** - Neo4j must have documents with entities  
⚠️ **No new entity extraction** - Uses entities already in Neo4j  
⚠️ **Content dependency** - If document content isn't in Neo4j, risk scoring may be limited  

## Troubleshooting

### "No documents found in Neo4j"

**Cause:** Neo4j database is empty or documents aren't labeled correctly

**Solution:** 
1. Check Neo4j connection in `.env`
2. Verify data was loaded: `python main.py --status`
3. Run the main pipeline first if needed

### "Failed to retrieve chunks"

**Cause:** ChromaDB collection doesn't exist or is empty

**Solution:** 
- Don't use `--include-chunks` flag
- Or run main pipeline with embeddings first

### Risk scores seem different

**Cause:** Document content might not be fully stored in Neo4j

**Solution:** 
- This is expected if content wasn't stored in graph
- Risk scores are calculated from available entities/relationships
- For most accurate scores, ensure content is in Neo4j

## Integration with Multiagent System

After running this script, use the output exactly like the main pipeline output:

```python
import json

# Load the risk analysis
with open('output/fraud_analysis_2026_20260206_123000.json', 'r') as f:
    data = json.load(f)

# Use in your multiagent system
for doc in data['documents']:
    if doc['risk_assessment']['overall_score'] >= 80:
        # Route to fraud investigation agent
        pass
```

## Summary

**You do NOT need to reprocess your entire dataset!**

Simply run:
```bash
python calculate_risk_from_existing_data.py --export --batch-name my_analysis
```

This will:
- ✅ Calculate risk scores from existing Neo4j data
- ✅ Export formatted output for your multiagent system
- ✅ Generate summary reports
- ✅ Complete in minutes instead of hours

Your existing data in Neo4j and ChromaDB is already sufficient for risk scoring!
