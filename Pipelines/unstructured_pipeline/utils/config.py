# File: utils/config.py
"""
Configuration management for the fraud detection system
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Central configuration class"""
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data" / "unstructured_data"
    VECTOR_DB_DIR = BASE_DIR / "databases" / "vector_store"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Vector Database (ChromaDB)
    VECTOR_DB_COLLECTION = "fraud_documents"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    
    # Neo4j Graph Database
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    
    # NLP Models
    SPACY_MODEL = "en_core_web_sm"
    
    # OpenAI (Optional - for advanced extraction)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    # Processing
    BATCH_SIZE = 32
    MAX_WORKERS = 4
    
    # Logging
    LOG_LEVEL = "INFO"
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        cls.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)

