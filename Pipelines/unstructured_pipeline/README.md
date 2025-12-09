# File: README.md

"""

# Financial Fraud Detection System - Unstructured Pipeline

A comprehensive AI-powered system for detecting financial fraud in unstructured documents using hybrid retrieval (Vector DB + Knowledge Graph).

## 🏗️ Architecture

```
fraud-detection-system/
├── data/
│   └── unstructured_data/          # Your 1800 .txt files
├── pipelines/
│   ├── data_loader.py              # Document loading
│   ├── chunking.py                 # Text chunking
│   ├── embedding.py                # Vector embeddings
│   ├── ner_extraction.py           # Entity extraction
│   ├── graph_builder.py            # Knowledge graph
│   └── unstructured_pipeline.py    # Main pipeline
├── databases/
│   ├── vector_db.py                # ChromaDB interface
│   └── graph_db.py                 # Neo4j interface
├── agents/
│   ├── nlp_disclosure_agent.py     # NLP analysis agent
│   └── graph_linkage_agent.py      # Graph query agent
├── utils/
│   ├── config.py                   # Configuration
│   ├── logger.py                   # Logging
│   └── exceptions.py               # Custom exceptions
├── tests/
│   └── test_pipeline.py
├── requirements.txt
├── setup.py
├── .env.example
└── main.py
```

## 📋 Prerequisites

- Python 3.9+
- Neo4j Database (version 5.x)
- 8GB RAM minimum (16GB recommended)
- 10GB disk space

## 🚀 Installation

### Step 1: Clone & Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Step 2: Install Neo4j

**Option A: Docker (Recommended)**

```bash
docker run \
    --name neo4j-fraud \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/your_password \
    -v $PWD/neo4j/data:/data \
    neo4j:5.15.0
```

**Option B: Local Installation**

- Download from: https://neo4j.com/download/
- Follow installation instructions
- Set password and note the bolt URI

### Step 3: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and set your Neo4j credentials
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here
```

### Step 4: Verify Data Directory

```bash
# Ensure your data is in the correct location
ls data/unstructured_data/

# Should show your 1800 .txt files
# NonFraud_1961_20200330_1376.txt
# Unknown_full-submission_163.txt
# etc.
```

## 🎯 Usage

### Run Complete Pipeline

Process all 1800 documents:

```bash
python main.py
```

Process limited number of documents (for testing):

```bash
python main.py --limit 100
```

### Run Specific Steps

Skip embedding generation (if already done):

```bash
python main.py --skip-embeddings
```

Skip knowledge graph (faster testing):

```bash
python main.py --skip-graph
```

### Query Operations

Query the vector database:

```bash
python main.py --query "special purpose entity fraud"
```

Check pipeline status:

```bash
python main.py --status
```

Reset pipeline (delete all data):

```bash
python main.py --reset
```

## 📊 Pipeline Steps

### Step 1: Data Ingestion

- Loads 1800 text files from `data/unstructured_data`
- Extracts metadata from filenames
- Parses fraud labels (NonFraud, Fraud, Unknown)

### Step 2: Text Chunking

- Splits documents into 512-token chunks
- 50-token overlap for context preservation
- Preserves document metadata in each chunk

### Step 3: Vector Embeddings

- Generates embeddings using `all-MiniLM-L6-v2`
- Stores in ChromaDB for similarity search
- Creates ~5-10 chunks per document

### Step 4: Entity Extraction

- Extracts entities: Companies, People, Money, Dates
- Identifies financial terms and fraud indicators
- Extracts relationships between entities

### Step 5: Knowledge Graph

- Builds graph in Neo4j
- Creates nodes for entities
- Creates relationships between entities
- Links entities to source documents

## 🔍 Example Queries

### Vector Database Query (Semantic Search)

```python
from pipelines.unstructured_pipeline import UnstructuredPipeline

pipeline = UnstructuredPipeline()
results = pipeline.query_vector_db(
    "revenue recognition fraud round-trip transactions",
    n_results=5
)
```

### Knowledge Graph Query (Cypher)

```python
# Find all companies with hidden subsidiaries
query = """
MATCH (c:Company)-[:OWNS]->(s:Subsidiary)
WHERE s.disclosed = false
RETURN c.name, s.name
"""
results = pipeline.query_knowledge_graph(query)
```

## 🧪 Testing

Run tests:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest --cov=pipelines --cov=databases tests/
```

## 📈 Performance Metrics

Expected processing times (on standard hardware):

- 100 documents: ~5-10 minutes
- 500 documents: ~20-30 minutes
- 1800 documents: ~60-90 minutes

Resource usage:

- RAM: 4-8 GB during processing
- Disk: ~2-3 GB for databases
- CPU: Multi-threaded embedding generation

## 🔧 Configuration

Key settings in `utils/config.py`:

- `CHUNK_SIZE`: 512 (tokens per chunk)
- `CHUNK_OVERLAP`: 50 (overlap tokens)
- `BATCH_SIZE`: 32 (embedding batch size)
- `EMBEDDING_MODEL`: sentence-transformers/all-MiniLM-L6-v2

## 📝 Output

The pipeline creates:

1. **Vector Database** (ChromaDB)
   - Location: `databases/vector_store/`
   - Contains: Document chunks + embeddings
2. **Knowledge Graph** (Neo4j)

   - Accessible at: http://localhost:7474
   - Contains: Entities + relationships

3. **Logs**
   - Location: `logs/`
   - Files: `UnstructuredPipeline_YYYYMMDD.log`

## 🚨 Troubleshooting

### Issue: Neo4j Connection Failed

```bash
# Check if Neo4j is running
docker ps | grep neo4j

# Restart Neo4j
docker restart neo4j-fraud
```

### Issue: Out of Memory

```bash
# Process in batches
python main.py --limit 100
```

### Issue: ChromaDB Lock Error

```bash
# Reset the database
python main.py --reset
```

## 📚 Next Steps

After running the unstructured pipeline:

1. Build the structured data pipeline (Excel files)
2. Implement the agent system (NLP Disclosure Agent, Graph Linkage Agent)
3. Create the hybrid retrieval system
4. Build the final scoring engine

## 📄 License

MIT License

## 🤝 Contributing

Contributions welcome! Please read CONTRIBUTING.md first.

## 📧 Contact

For questions or support, please open an issue on GitHub.
"""
