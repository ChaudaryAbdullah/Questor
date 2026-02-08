# Quick Setup - Unified Fraud Detection Pipeline

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 4GB+ RAM recommended
- 10GB+ disk space for dependencies

## Automated Setup (Recommended)

### 1. One-Command Setup

```bash
cd /home/cypher/Questor/Pipelines
chmod +x setup.sh
./setup.sh
```

This script will:
- ✅ Create a virtual environment (`venv/`)
- ✅ Upgrade pip, setuptools, wheel
- ✅ Install all requirements from both pipelines
- ✅ Download spaCy language model
- ✅ Verify installation

**Time:** ~5-10 minutes depending on your internet speed

### 2. Activate Environment

```bash
source venv/bin/activate
```

### 3. Run Test

```bash
python unified_runner.py --help
```

## Manual Setup

If the automated script doesn't work:

### 1. Create Virtual Environment

```bash
cd /home/cypher/Questor/Pipelines
python3 -m venv venv
source venv/bin/activate
```

### 2. Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Install spaCy Model

```bash
python -m spacy download en_core_web_sm
```

### 5. Verify Installation

```bash
python -c "import numpy, pandas, sklearn, tensorflow, torch, chromadb, neo4j, transformers, yaml; print('✅ All imports successful')"
```

## Post-Installation

### Test Unified Pipeline

```bash
# Activate environment
source venv/bin/activate

# Test with limited data
python unified_runner.py --pipeline both --limit 5

# View help
python unified_runner.py --help
```

### Deactivate Environment

```bash
deactivate
```

## Troubleshooting

### Issue: Virtual environment creation fails

**Solution:** Ensure python3-venv is installed
```bash
sudo apt-get install python3-venv  # Ubuntu/Debian
```

### Issue: PyTorch installation is slow

**Solution:** Use CPU-only version (smaller download)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Issue: TensorFlow compatibility errors

**Solution:** Install specific version
```bash
pip install tensorflow==2.13.0
```

### Issue: Out of memory during installation

**Solution:** Install packages one at a time
```bash
pip install numpy pandas scikit-learn
pip install tensorflow
pip install torch
pip install chromadb neo4j
pip install sentence-transformers spacy transformers
```

### Issue: spaCy model download fails

**Solution:** Download manually
```bash
python -m spacy download en_core_web_sm --no-cache-dir
```

## What Gets Installed?

### Core ML Libraries
- NumPy, Pandas, Scikit-learn
- XGBoost, LightGBM, CatBoost
- TensorFlow, PyTorch

### NLP & Knowledge Graphs
- ChromaDB (vector database)
- Neo4j (graph database)
- Sentence Transformers
- spaCy, Transformers

### Utilities
- PyYAML (for config)
- tqdm (progress bars)
- psutil (memory monitoring)
- joblib (model serialization)

**Total size:** ~5-8GB including models

## Directory Structure After Setup

```
Pipelines/
├── venv/                    # Virtual environment (created by setup)
├── requirements.txt         # All dependencies
├── setup.sh                 # Automated setup script
├── unified_runner.py        # Main entry point
├── score_combiner.py        # Score combination
├── config.yaml              # Configuration
├── shared/                  # Shared infrastructure
├── output/                  # All outputs (created automatically)
├── stuctured_pipeline/      # Structured pipeline
└── unstructured_pipeline/   # Unstructured pipeline
```

## Next Steps

After successful installation:

1. ✅ Read [UNIFIED_PIPELINE_GUIDE.md](UNIFIED_PIPELINE_GUIDE.md) for usage
2. ✅ Configure Neo4j and ChromaDB (for unstructured pipeline)
3. ✅ Test with sample data
4. ✅ Adjust weights in `config.yaml` as needed

## Keeping Environment Active

### Option 1: Always activate before use
```bash
cd /home/cypher/Questor/Pipelines
source venv/bin/activate
python unified_runner.py ...
deactivate
```

### Option 2: Add alias to .bashrc
```bash
echo 'alias fraud-pipeline="cd /home/cypher/Questor/Pipelines && source venv/bin/activate"' >> ~/.bashrc
source ~/.bashrc

# Then just run:
fraud-pipeline
python unified_runner.py ...
```

## Updating Dependencies

```bash
source venv/bin/activate
pip install --upgrade -r requirements.txt
```

## Clean Reinstall

```bash
rm -rf venv
./setup.sh
```
