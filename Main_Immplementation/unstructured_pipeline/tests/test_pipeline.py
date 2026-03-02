
# File: tests/test_pipeline.py
"""
Unit tests for the pipeline
"""
import pytest
from pathlib import Path
import tempfile
import shutil
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.unstructured_pipeline import UnstructuredPipeline
from pipelines import DataLoader, TextChunker, EmbeddingGenerator
from utils import Config


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory for testing"""
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create sample files
    for i in range(5):
        file_path = temp_dir / f"NonFraud_{1000+i}_20230101_{i}.txt"
        file_path.write_text(f"This is test document {i} about financial fraud detection.")
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


def test_data_loader(temp_data_dir):
    """Test document loading"""
    loader = DataLoader(data_dir=temp_data_dir)
    documents = loader.load_documents()
    
    assert len(documents) == 5
    assert all('doc_id' in doc for doc in documents)
    assert all('content' in doc for doc in documents)


def test_text_chunker():
    """Test text chunking"""
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    
    text = "This is a test sentence. " * 100
    chunks = chunker.chunk_text(text, "test_doc")
    
    assert len(chunks) > 1
    assert all('chunk_id' in chunk for chunk in chunks)
    assert all('text' in chunk for chunk in chunks)


def test_embedding_generator():
    """Test embedding generation"""
    generator = EmbeddingGenerator()
    
    texts = ["Test document about fraud", "Another test document"]
    embeddings = generator.generate_embeddings(texts, show_progress=False)
    
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] == generator.get_embedding_dimension()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

