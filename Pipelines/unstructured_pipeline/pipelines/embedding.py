
# File: pipelines/embedding.py
"""
Embedding generation using sentence transformers
"""
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm

from utils import Config, Logger
from utils.exceptions import EmbeddingError


class EmbeddingGenerator:
    """Generates embeddings for text chunks"""
    
    def __init__(self, model_name: str = Config.EMBEDDING_MODEL):
        self.logger = Logger.get_logger(self.__class__.__name__)
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model"""
        try:
            self.logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.logger.info("Embedding model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {str(e)}")
            raise EmbeddingError(f"Model loading failed: {str(e)}")
    
    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = Config.BATCH_SIZE,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of embeddings
        """
        try:
            self.logger.info(f"Generating embeddings for {len(texts)} texts...")
            
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            
            self.logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {str(e)}")
            raise EmbeddingError(f"Embedding generation failed: {str(e)}")
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        try:
            embedding = self.model.encode([text], convert_to_numpy=True)
            return embedding[0]
        except Exception as e:
            self.logger.error(f"Single embedding generation failed: {str(e)}")
            raise EmbeddingError(f"Single embedding generation failed: {str(e)}")
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors"""
        return self.model.get_sentence_embedding_dimension()

