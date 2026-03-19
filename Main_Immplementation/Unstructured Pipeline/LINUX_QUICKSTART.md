# 🚀 Quick Start Guide - Linux

## Setup (One-time)

Run the automated setup script:

```bash
./setup_linux.sh
```

This will:
- ✓ Check Python version (3.11+ required)
- ✓ Create virtual environment
- ✓ Install all dependencies
- ✓ Download spaCy language model
- ✓ Set up environment files
- ✓ Create data directories

**After setup completes:**
1. Edit `.env` file and add your Groq API key:
   ```bash
   nano .env  # or use your preferred editor
   ```
2. Get a free API key from: https://console.groq.com/keys

## Running the Pipeline

Simply run:

```bash
./run.sh
```

This will:
- Activate the virtual environment
- Validate configuration
- Process the first 10-K filing in `Input/` directory
- Save results to `data/reports/`

**Expected runtime:** 6-8 minutes for a 200-page 10-K

## Manual Commands

If you prefer running manually:

```bash
# Activate virtual environment
source venv/bin/activate

# Run pipeline on default input
python run_pipeline.py

# Run with specific file
python -m src.utils.pipeline_orchestrator --input "Input/your_file.txt" --output "data/reports"
```

## Project Structure

```
Unstructured Pipeline/
├── Input/                    # Place 10-K .txt files here
├── data/reports/             # Output reports (JSON)
├── config/                   # Configuration files
├── src/                      # Source code
├── run_pipeline.py           # Main entry point
├── setup_linux.sh            # Setup script (run once)
└── run.sh                    # Run script (run anytime)
```

## What the Pipeline Does

1. **Preprocessing**: Cleans and chunks SEC 10-K filings
2. **Entity Extraction**: Extracts companies, people, transactions using Groq LLM
3. **Embeddings**: Creates semantic embeddings using sentence-transformers
4. **Knowledge Graph**: Builds relationship graphs (NetworkX/Neo4j)
5. **RAG Analysis**: Runs fraud detection queries
6. **Pattern Detection**: Identifies 8 types of fraud patterns

## Output Example

Results saved in `data/reports/`:

```json
{
  "document_id": "1504008_20170228",
  "entities_extracted": 245,
  "fraud_findings_count": 3,
  "fraud_findings": [
    {
      "pattern_name": "UNDISCLOSED_RELATED_PARTY",
      "severity": "HIGH",
      "confidence": 0.85,
      "description": "..."
    }
  ]
}
```

## Fraud Detection Capabilities

The pipeline detects:
1. Undisclosed Related Party Transactions
2. Hidden Subsidiaries (off-balance sheet)
3. Circular Transaction Patterns
4. Revenue Recognition Irregularities
5. Executive Self-Dealing
6. Auditor Red Flags
7. Shell Company Structures
8. MD&A vs Footnote Contradictions

## Requirements

- **OS**: Linux (Ubuntu, Debian, Fedora, etc.)
- **Python**: 3.11+
- **RAM**: 8GB+ recommended
- **Disk**: 5GB+ free space
- **API**: Groq API key (free tier available)
- **Neo4j**: Optional (NetworkX fallback built-in)

## Troubleshooting

### Permission Denied
```bash
chmod +x setup_linux.sh run.sh
```

### Python Version Issues
```bash
# Check version
python3 --version

# Install Python 3.11 on Ubuntu/Debian
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev
```

### API Key Errors
- Verify `.env` file has valid `GROQ_API_KEY`
- Check API rate limits at console.groq.com

### Memory Issues
- Reduce batch size in `config/settings.yaml`
- Close other applications

### No Input Files
```bash
# Check Input directory
ls -la Input/

# Add .txt files to Input/ directory
```

## Docker Alternative

```bash
# Build image
docker build -t fraud-detection-pipeline .

# Run container
docker run -v $(pwd)/data:/app/data \
  -e GROQ_API_KEY=your-key \
  fraud-detection-pipeline \
  --input Input/filing.txt
```

## Getting Help

1. Check logs in console output
2. Review `config/settings.yaml` for customization
3. See full documentation in `README.md`

---

**Created:** 2026-01-23  
**Platform:** Linux  
**Project:** Unstructured Data Pipeline for Financial Fraud Detection
