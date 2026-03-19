# Unstructured Data Pipeline for Financial Fraud Detection

A comprehensive AI-powered pipeline for processing SEC 10-K filings to detect potential fraud indicators using NLP, knowledge graphs, and retrieval-augmented generation (RAG).

## Features

- **Document Processing**: Automated extraction and cleaning of SEC 10-K filings
- **Entity Extraction**: AI-powered extraction using Groq LLM with spaCy fallback
- **Knowledge Graphs**: Neo4j-based relationship mapping with NetworkX fallback
- **Vector Search**: ChromaDB for semantic similarity search
- **RAG Analysis**: Groq-powered fraud detection queries
- **Fraud Detection**: Pattern-based detection of common fraud schemes

## Architecture

```
Input (10-K Filing)
    ↓
Document Preprocessing
    ↓
Entity Extraction (Groq + spaCy)
    ↓
Embeddings (sentence-transformers)
    ↓
Vector Storage (ChromaDB) + Knowledge Graph (Neo4j)
    ↓
RAG Engine (Groq)
    ↓
Fraud Pattern Detection
    ↓
Reports & Visualizations
```

## Installation

### Prerequisites

- Python 3.11+
- Neo4j (optional, NetworkX fallback available)
- Groq API key

### Setup

1. Clone the repository:
```bash
cd "d:\FAST\FYP\Unstructured Pipeline"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download spaCy model:
```bash
python -m spacy download en_core_web_lg
```

4. Set environment variables:
```bash
# Windows PowerShell
$env:GROQ_API_KEY="your-groq-api-key"
$env:NEO4J_PASSWORD="your-neo4j-password"  # Optional
```

## Configuration

Edit `config/settings.yaml` to customize:

- LLM model selection (Groq)
- Embedding models
- Vector database settings
- Graph database settings
- Fraud detection parameters

## Usage

### Command Line

Process a single 10-K filing:

```bash
python -m src.utils.pipeline_orchestrator --input "Input/1504008_20170228.txt" --output "data/reports"
```

### Python API

```python
from src.utils.pipeline_orchestrator import PipelineOrchestrator

# Initialize pipeline
orchestrator = PipelineOrchestrator()

# Process document
result = orchestrator.process_document("Input/1504008_20170228.txt")

# Access results
print(f"Entities extracted: {result.entities_extracted}")
print(f"Fraud findings: {result.fraud_findings_count}")

# Cleanup
orchestrator.cleanup()
```

### Programmatic Usage

```python
# Individual components
from src.preprocessing import DocumentPreprocessor
from src.nlp import EntityExtractor
from src.retrieval import RAGEngine

# Process document
preprocessor = DocumentPreprocessor()
doc = preprocessor.process("filing.txt")

# Extract entities
extractor = EntityExtractor()
entities = extractor.extract_from_chunks(doc.chunks)

# Run fraud queries
rag = RAGEngine()
response = rag.query("Find undisclosed related party transactions")
```

## Project Structure

```
unstructured_pipeline/
├── config/
│   ├── settings.yaml           # Main configuration
│   ├── constants.py            # Entity types, fraud patterns
│   ├── entity_types.yaml       # Entity definitions
│   └── relationship_rules.yaml # Fraud pattern rules
├── src/
│   ├── preprocessing/          # Document processing
│   ├── nlp/                    # Entity extraction & embeddings
│   ├── graph/                  # Knowledge graph construction
│   ├── retrieval/              # Vector search & RAG
│   ├── validators/             # Fraud pattern detection
│   └── utils/                  # Logging, config, orchestration
├── data/
│   ├── raw/                    # Input documents
│   ├── processed/              # Processed chunks
│   ├── vectors/                # Embeddings & ChromaDB
│   ├── graphs/                 # Graph exports
│   └── reports/                # Output reports
├── Input/                      # 10-K filings
├── requirements.txt
├── Dockerfile
└── README.md
```

## Fraud Detection Capabilities

The pipeline detects:

1. **Undisclosed Related Party Transactions**
2. **Hidden Subsidiaries** (off-balance sheet)
3. **Circular Transaction Patterns**
4. **Revenue Recognition Irregularities**
5. **Executive Self-Dealing**
6. **Auditor Red Flags**
7. **Shell Company Structures**
8. **MD&A vs Footnote Contradictions**

## Output

Results are saved as JSON in `data/reports/`:

```json
{
  "document_id": "1504008_20170228",
  "company_name": "Example Corp",
  "entities_extracted": 245,
  "fraud_findings_count": 3,
  "fraud_findings": [
    {
      "pattern_name": "UNDISCLOSED_RELATED_PARTY",
      "severity": "HIGH",
      "description": "...",
      "confidence": 0.85
    }
  ]
}
```

## Docker Deployment

```bash
# Build image
docker build -t fraud-detection-pipeline .

# Run container
docker run -v $(pwd)/data:/app/data \
  -e GROQ_API_KEY=your-key \
  fraud-detection-pipeline \
  --input Input/filing.txt
```

## API Integration

The pipeline is designed to integrate with other systems:

- **Modular Components**: Each module can be used independently
- **JSON I/O**: Standardized input/output formats
- **REST API Ready**: Can be wrapped in FastAPI (included in requirements)

## Performance

Typical processing times for a 200-page 10-K:

- Preprocessing: ~30s
- Entity Extraction: ~2-3 min (with Groq)
- Embeddings: ~1 min
- Graph Construction: ~30s
- Fraud Analysis: ~2 min

**Total: ~6-8 minutes**

## Troubleshooting

### Groq API Issues
- Verify `GROQ_API_KEY` is set
- Check API rate limits
- Fallback to spaCy if API unavailable

### Neo4j Connection
- Pipeline uses NetworkX fallback automatically
- Set `NEO4J_PASSWORD` if using Neo4j

### Memory Issues
- Reduce `batch_size` in `config/settings.yaml`
- Process documents in smaller chunks

## Contributing

This is a research project for financial fraud detection. Contributions welcome!

## License

MIT License

## Contact

For questions or issues, please open a GitHub issue.

## Acknowledgments

- Groq for LLM API
- ChromaDB for vector storage
- Neo4j for graph database
- FinBERT for financial NLP
