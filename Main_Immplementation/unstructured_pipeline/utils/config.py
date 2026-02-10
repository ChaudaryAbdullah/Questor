# File: utils/config_optimized.py
"""
Memory-Optimized Configuration for the fraud detection system
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class ConfigOptimized:
    """Memory-optimized configuration class"""
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR.parent / "Input"  # Use centralized Input folder from Main_Immplementation
    VECTOR_DB_DIR = BASE_DIR / "databases" / "vector_store"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Vector Database (ChromaDB)
    VECTOR_DB_COLLECTION = "fraud_documents"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Chunking Settings - Optimized for memory
    CHUNK_SIZE = 512  # tokens per chunk
    CHUNK_OVERLAP = 50  # overlap tokens
    
    # Neo4j Graph Database - Optimized connection pool
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    NEO4J_MAX_CONNECTION_LIFETIME = 3600  # 1 hour
    NEO4J_MAX_CONNECTION_POOL_SIZE = 10  # Reduced from default 100
    NEO4J_CONNECTION_TIMEOUT = 30  # seconds
    
    # NLP Models
    SPACY_MODEL = "en_core_web_sm"  # Use small model for memory efficiency
    
    # Processing Settings - Memory Optimized
    BATCH_SIZE = 16  # Reduced from 32
    DOCUMENT_BATCH_SIZE = 10  # Process 10 documents at a time
    EMBEDDING_BATCH_SIZE = 8  # Smaller batches for embeddings
    NER_CHUNK_SIZE = 500000  # 500K characters per NER chunk
    MAX_WORKERS = 2  # Reduced parallelism to save memory
    
    # Memory Management
    ENABLE_GARBAGE_COLLECTION = True
    GC_THRESHOLD = 100  # Run GC every N documents
    MAX_MEMORY_PERCENT = 80  # Alert if memory usage exceeds this
    
    # Logging
    LOG_LEVEL = "INFO"
    
    # Performance Settings
    USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"
    TORCH_NUM_THREADS = int(os.getenv("TORCH_NUM_THREADS", "2"))
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        cls.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_memory_optimized_settings(cls):
        """Get current memory-optimized settings"""
        return {
            'batch_size': cls.BATCH_SIZE,
            'document_batch_size': cls.DOCUMENT_BATCH_SIZE,
            'embedding_batch_size': cls.EMBEDDING_BATCH_SIZE,
            'ner_chunk_size': cls.NER_CHUNK_SIZE,
            'max_workers': cls.MAX_WORKERS,
            'gc_enabled': cls.ENABLE_GARBAGE_COLLECTION,
            'gc_threshold': cls.GC_THRESHOLD,
        }