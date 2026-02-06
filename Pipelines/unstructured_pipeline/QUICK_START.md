# 🎯 QUICK START: Risk Scoring from Your Existing Data

## You Asked: "Do I need to run the whole dataset again?"

## Answer: **NO! You can calculate risk scores from your existing Neo4j and ChromaDB data in minutes!**

---

## 🚀 Immediate Next Steps

### Option 1: Quick Risk Assessment (Recommended)

```bash
cd "d:\FAST NUCES\FYP\Questor\Pipelines\unstructured_pipeline"

# Calculate risk scores for all your existing data
python calculate_risk_from_existing_data.py --export --batch-name my_first_analysis
```

**Time:** 2-10 minutes (vs 60-90 minutes for full reprocessing)

### Option 2: Test with Limited Data First

```bash
# Test with 100 documents first
python calculate_risk_from_existing_data.py --limit 100 --export --batch-name test_run
```

**Time:** ~30 seconds

---

## 📊 What You'll Get

After running the script, you'll have:

### 1. JSON Output for Multiagent System
**File:** `output/my_first_analysis_[timestamp].json`

Contains:
- Risk scores (0-100) for each document
- Risk levels (CRITICAL, HIGH, MEDIUM, LOW, MINIMAL)
- Extracted entities and relationships
- Agent routing recommendations
- Retrieval references for RAG

### 2. Summary Report
**File:** `output/my_first_analysis_summary.txt`

Contains:
- Overall statistics
- Risk distribution
- Top high-risk documents
- Risk factors

### 3. Console Output
Immediate feedback showing:
- Number of documents processed
- Average/max/min risk scores
- Risk level distribution
- Top 10 high-risk documents

---

## 📁 Files Created for You

| File | Purpose |
|------|---------|
| `calculate_risk_from_existing_data.py` | **Main script** - Run this to get risk scores |
| `pipelines/risk_scorer.py` | Risk scoring algorithm |
| `pipelines/output_formatter.py` | Output formatting for multiagent system |
| `EXISTING_DATA_RISK_GUIDE.md` | **Detailed guide** - Read this for more info |
| `RISK_SCORING_GUIDE.md` | Complete risk scoring documentation |
| `INTEGRATION_SUMMARY.md` | Integration overview |
| `examples/risk_scoring_example.py` | Example usage scripts |

---

## 🎬 Complete Workflow

### Step 1: Calculate Risk Scores (2-10 minutes)

```bash
python calculate_risk_from_existing_data.py --export --batch-name fraud_analysis_2026
```

### Step 2: Review Results

Check the console output for immediate summary, then review:
- `output/fraud_analysis_2026_[timestamp].json` - Full data
- `output/fraud_analysis_2026_summary.txt` - Summary report

### Step 3: Use in Your Multiagent System

```python
import json

# Load the analysis
with open('output/fraud_analysis_2026_20260206_123000.json', 'r') as f:
    data = json.load(f)

# Get high-risk documents
high_risk = data['high_risk_documents']

# Process each document
for doc in data['documents']:
    risk_score = doc['risk_assessment']['overall_score']
    risk_level = doc['risk_assessment']['risk_level']
    agents = doc['agent_routing']['recommended_agents']
    
    # Route to appropriate agents based on risk
    if risk_score >= 80:
        # Send to fraud investigation agent
        pass
    elif risk_score >= 60:
        # Send to risk assessment agent
        pass
```

---

## 🔍 How It Works

The script:

1. **Connects to Neo4j** → Retrieves your documents with entities and relationships
2. **Calculates Risk Scores** → Uses 4 components:
   - Fraud Indicators (35%)
   - Entity Risk (25%)
   - Financial Anomalies (25%)
   - Relationship Risk (15%)
3. **Formats Output** → Structures data for your multiagent system
4. **Exports Results** → Saves JSON and summary report

**No reprocessing needed!** Uses your existing data.

---

## 💡 Key Advantages

| Feature | This Script | Full Reprocessing |
|---------|-------------|-------------------|
| **Time** | 2-10 minutes | 60-90 minutes |
| **Data Source** | Neo4j + ChromaDB | Raw text files |
| **Model Loading** | None needed | NER + Embeddings |
| **Flexibility** | Run anytime | Heavy operation |
| **Output** | Same format | Same format |

---

## 🎯 Risk Scoring Details

### Risk Levels

- **CRITICAL (80-100)**: Immediate investigation required
- **HIGH (60-79)**: Priority review needed
- **MEDIUM (40-59)**: Standard review
- **LOW (20-39)**: Monitoring
- **MINIMAL (0-19)**: Low concern

### What's Analyzed

✅ Fraud keywords (fictitious, concealment, manipulation)  
✅ Financial complexity (SPEs, derivatives, off-balance sheet)  
✅ Large monetary amounts (>$100M, >$10M, >$1M)  
✅ Suspicious relationships (concealed, transferred)  
✅ Entity patterns (multiple orgs, complex structures)  

---

## 📚 Documentation

- **Quick Start**: This file (you're reading it!)
- **Detailed Guide**: `EXISTING_DATA_RISK_GUIDE.md`
- **Full Documentation**: `RISK_SCORING_GUIDE.md`
- **Examples**: `examples/risk_scoring_example.py`

---

## ❓ Common Questions

### Q: Will this modify my existing data?
**A:** No! It only reads from Neo4j and ChromaDB. Your data is safe.

### Q: Can I run this multiple times?
**A:** Yes! Run it as many times as you want. Each run creates a new timestamped output.

### Q: What if I want to adjust risk weights?
**A:** Edit `pipelines/risk_scorer.py` → Modify the `WEIGHTS` dictionary → Re-run the script.

### Q: Do I need to include chunks from ChromaDB?
**A:** No, not necessary for risk scoring. Only use `--include-chunks` if you need chunk data for other purposes.

### Q: Can I filter by document label (Fraud/NonFraud)?
**A:** The script processes all documents. You can filter the output JSON by label in your multiagent system.

---

## 🚨 Troubleshooting

### "No documents found in Neo4j"
→ Ensure you've run the main pipeline at least once  
→ Check Neo4j connection in `.env`

### "Connection refused"
→ Ensure Neo4j is running: `docker ps | grep neo4j`  
→ Check ChromaDB path in config

### Script runs but no high-risk documents
→ This is normal if your data is mostly clean  
→ Check the summary for overall distribution

---

## ✅ Ready to Start?

Run this command now:

```bash
python calculate_risk_from_existing_data.py --export --batch-name my_analysis
```

Then check:
- Console for immediate summary
- `output/my_analysis_[timestamp].json` for full data
- `output/my_analysis_summary.txt` for report

**Your multiagent system data is ready in minutes!** 🎉

---

## 📞 Need Help?

1. Check `EXISTING_DATA_RISK_GUIDE.md` for detailed usage
2. Review `RISK_SCORING_GUIDE.md` for risk scoring details
3. Run `python examples/risk_scoring_example.py` for examples
4. Check logs in `logs/` directory for errors

---

**Bottom Line:** You have everything you need. Just run the script and you'll have risk scores and formatted output for your multiagent system in minutes! 🚀
