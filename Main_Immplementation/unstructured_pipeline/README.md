# 🔶 Unstructured Pipeline

Document-level fraud risk analysis using **ChromaDB vector search**, **entity extraction**, and **keyword-based risk scoring**. Operates in **retrieval mode** — searches pre-processed documents stored in ChromaDB by CIK number, avoiding full reprocessing.

---

## Quick Start

```bash
cd Main_Immplementation
source venv/bin/activate

# Run retrieval for a specific CIK (fastest)
python3 unstructured_pipeline/pipelines/risk_retriever.py --cik 1040719

# Via the unified runner (recommended)
python3 run_unified.py
```

---

## Folder Structure

```
unstructured_pipeline/
│
├── ── Entry Points ──
├── main.py                          # Standalone pipeline runner
├── calculate_risk_from_existing_data.py  # Pure retrieval mode
│
├── pipelines/
│   ├── unstructured_pipeline.py     # 🎯 Main optimized pipeline class
│   ├── risk_retriever.py            # ChromaDB/Neo4j retrieval by CIK
│   ├── risk_scorer.py               # Keyword-based risk scoring engine
│   ├── data_loader.py               # Document ingestion & chunking
│   ├── entity_extractor.py          # Named entity recognition
│   ├── output_formatter.py          # Standardized output format
│   └── document_processor.py       # Text preprocessing
│
├── databases/
│   ├── vector_db.py                 # ChromaDB client (stores embeddings)
│   └── graph_db.py                  # Neo4j knowledge graph (optional)
│
├── utils/
│   ├── cik_extractor.py             # CIK normalization & file mapping
│   ├── config.py                    # Pipeline configuration
│   ├── logger.py                    # Logging setup
│   └── helpers.py                   # Shared utility functions
│
├── data/                            # Raw document storage
├── examples/                        # Usage examples
├── tests/                           # Unit tests
└── docker-compose.yml               # Neo4j docker setup
```

---

## How It Works

### Retrieval Mode (Used by Unified Pipeline)

Instead of reprocessing documents from scratch, the pipeline queries **pre-ingested data** from ChromaDB:

```
CIK from input filename (e.g., "0001040719" → "1040719")
        │
        ▼
ChromaDB query: WHERE company_id = "1040719"
        │
        ▼
Retrieved chunks (1000+ document snippets per company)
        │
        ▼
risk_scorer.py: keyword scan + entity extraction + anomaly detection
        │
        ▼
Formatted risk assessment output
```

**Speed**: ~2–4 seconds vs. 600+ seconds for full document processing.

### Full Processing Mode

For new documents not yet in ChromaDB:

```
Input: full-submission.txt (SEC EDGAR full submission)
        │
        ▼
data_loader.py        → Split into chunks (≈512 tokens each)
        │
        ▼
entity_extractor.py   → Extract companies, people, amounts
        │
        ▼
vector_db.py          → Embed chunks with sentence-transformers
                        → Store in ChromaDB with CIK metadata
        │
        ▼
risk_scorer.py        → Score all chunks
```

---

## Risk Scoring Engine (`pipelines/risk_scorer.py`)

Scores documents across **4 weighted components**:

| Component | Weight | What It Detects |
|-----------|--------|----------------|
| `fraud_indicators` | 35% | High-risk keywords: `material weakness`, `restatement`, `going concern`, `misstatement`, `regulatory action` |
| `entity_risk` | 25% | Suspicious entities, shell companies, offshore relationships |
| `financial_anomalies` | 25% | Unusual financial patterns in text |
| `relationship_risk` | 15% | Suspicious related-party relationships |

### Risk Keywords (Sample)

```python
FRAUD_INDICATORS = {
    'critical': ['fraud', 'restatement', 'going concern', 'material weakness'],
    'high':     ['misstatement', 'regulatory action', 'SEC investigation'],
    'medium':   ['significant doubt', 'internal control deficiency'],
    'low':      ['risk factor', 'uncertainty', 'litigation']
}
```

### Output Score Mapping

| Score | Risk Level | Action |
|-------|-----------|--------|
| 0–19 | LOW | Routine monitoring |
| 20–39 | MEDIUM | Flag for review |
| 40–59 | HIGH | Detailed investigation |
| 60–100 | CRITICAL | Immediate escalation |

---

## Databases

### ChromaDB (Vector Database)

- **Collection**: `fraud_documents`
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Metadata Stored**: `company_id` (CIK), `label`, `date`, `chunk_index`
- **Location**: `unstructured_pipeline/databases/chroma_db/` (auto-created)

Query by CIK:
```python
from databases.vector_db import VectorDatabase
db = VectorDatabase()
results = db.query_by_cik("1040719", n_results=100)
```

### Neo4j (Knowledge Graph, Optional)

- **Purpose**: Stores entity relationships (person → company, transaction → entity)
- **Connection**: `bolt://localhost:7687`
- **Fallback**: System automatically uses ChromaDB if Neo4j is empty or unavailable
- **Setup**: `docker-compose up -d` (uses `docker-compose.yml`)

---

## CIK Extractor (`utils/cik_extractor.py`)

Handles CIK normalization across different filename formats:

```python
from utils.cik_extractor import CIKExtractor
extractor = CIKExtractor()

# Extract from various formats
extractor.extract_cik("0001040719.json")  # → "1040719"
extractor.extract_cik("CIK0001040719.txt") # → "1040719"

# Get all CIK→filename mappings in a directory
mapping = extractor.get_cik_file_mapping(Path("Input/"), "*.json")
# → {"1040719": "0001040719.json"}
```

---

## Output Format

```json
{
  "document_id": "NonFraud_1040719_20140227_711",
  "cik": "1040719",
  "risk_assessment": {
    "overall_score": 24.91,
    "risk_level": "LOW",
    "component_scores": {
      "fraud_indicators": 60.45,
      "entity_risk": 0.0,
      "financial_anomalies": 15.0,
      "relationship_risk": 0.0
    },
    "risk_factors": [
      "High-risk keywords detected: material weakness, restatement, misstatement"
    ],
    "requires_investigation": false
  },
  "extracted_data": {
    "entities": {},
    "relationships": []
  },
  "retrieval_references": {
    "vector_db_chunks": [],
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
  }
}
```

---

## Configuration

Key settings in `utils/config.py`:

```python
CHROMA_COLLECTION    = "fraud_documents"
EMBEDDING_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE           = 512       # tokens per chunk
CHUNK_OVERLAP        = 50
NEO4J_URI            = "bolt://localhost:7687"
NEO4J_USER           = "neo4j"
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `0 documents retrieved` | CIK not in ChromaDB | Ingest documents first using `data_loader.py` |
| `Neo4j connection failed` | Neo4j not running | Expected — auto-falls back to ChromaDB |
| `overall_score: 0.0` | No relevant chunks found for CIK | Verify CIK format (no leading zeros in DB) |
| Slow processing | Full processing mode triggered | Use retrieval mode (default in unified runner) |
