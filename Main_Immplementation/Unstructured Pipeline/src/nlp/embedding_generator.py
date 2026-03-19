"""
Embedding Generator Module
Generates embeddings using sentence-transformers with FinBERT support.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np

from ..utils.logger import get_nlp_logger
from ..utils.config_manager import get_config


class EmbeddingGenerator:
    """
    Generates text embeddings using sentence-transformers.
    Supports caching and batch processing.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_enabled: bool = True,
        cache_dir: Optional[str] = None,
        batch_size: int = 32
    ):
        """
        Initialize embedding generator.
        
        Args:
            model_name: HuggingFace model name
            cache_enabled: Whether to cache embeddings
            cache_dir: Directory for embedding cache
            batch_size: Batch size for processing
        """
        self.logger = get_nlp_logger()
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache_enabled = cache_enabled
        
        # Load config
        try:
            config = get_config()
            embed_config = config.embedding_config
            self.model_name = embed_config.get("primary_model", model_name)
            self.batch_size = embed_config.get("batch_size", batch_size)
            self.cache_dir = Path(embed_config.get("cache_dir", cache_dir or "data/vectors/cache"))
        except Exception:
            self.cache_dir = Path(cache_dir) if cache_dir else Path("data/vectors/cache")
        
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._model = None
        self._initialized = False
        self.embedding_dim = 384  # Default for MiniLM
        
        self.logger.info(f"EmbeddingGenerator initialized with model: {self.model_name}")
    
    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        if self._initialized:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            from pathlib import Path
            import os
            
            self.logger.info(f"Loading embedding model: {self.model_name}")
            
            # Priority 1: Check project's local_models directory
            project_local_path = Path(__file__).parent.parent.parent / "local_models"
            
            # Priority 2: Check user's home cache
            home_cache_path = Path.home() / ".cache/torch/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2"
            
            # Try local_models first (user's manual download)
            if project_local_path.exists() and (project_local_path / "model.safetensors").exists():
                self.logger.info(f"✓ Loading from project local_models: {project_local_path}")
                self._model = SentenceTransformer(str(project_local_path))
                
            # Try home cache second
            elif home_cache_path.exists() and (home_cache_path / "model.safetensors").exists():
                self.logger.info(f"Loading from home cache: {home_cache_path}")
                self._model = SentenceTransformer(str(home_cache_path))
                
            # Fall back to online download
            else:
                self.logger.info("Local models not found, downloading from HuggingFace...")
                # Set cache directory
                cache_folder = str(Path.home() / ".cache/torch/sentence_transformers")
                self._model = SentenceTransformer(self.model_name, cache_folder=cache_folder)
            
            self.embedding_dim = self._model.get_sentence_embedding_dimension()
            self._initialized = True
            self.logger.info(f"✓ Model loaded successfully! Embedding dimension: {self.embedding_dim}")
            
        except ImportError:
            self.logger.error("sentence-transformers not installed")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(f"{self.model_name}:{text}".encode()).hexdigest()
    
    def _get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache if available."""
        if not self.cache_enabled:
            return None
        
        cache_key = self._get_cache_key(text)
        cache_file = self.cache_dir / f"{cache_key}.npy"
        
        if cache_file.exists():
            try:
                return np.load(cache_file)
            except Exception:
                pass
        return None
    
    def _cache_embedding(self, text: str, embedding: np.ndarray) -> None:
        """Cache embedding to disk."""
        if not self.cache_enabled:
            return
        
        cache_key = self._get_cache_key(text)
        cache_file = self.cache_dir / f"{cache_key}.npy"
        
        try:
            np.save(cache_file, embedding)
        except Exception as e:
            self.logger.warning(f"Failed to cache embedding: {e}")
    
    def generate(self, text: str) -> np.ndarray:
        """
        Generate embedding for single text.
        
        Args:
            text: Text to embed
        
        Returns:
            Numpy array of embedding
        """
        # Check cache
        cached = self._get_cached_embedding(text)
        if cached is not None:
            return cached
        
        self._load_model()
        
        embedding = self._model.encode(text, convert_to_numpy=True)
        self._cache_embedding(text, embedding)
        
        return embedding
    
    def generate_batch(
        self,
        texts: List[str],
        show_progress: bool = False
    ) -> List[np.ndarray]:
        """
        Generate embeddings for batch of texts.
        
        Args:
            texts: List of texts to embed
            show_progress: Show progress bar
        
        Returns:
            List of embedding arrays
        """
        self._load_model()
        
        # Check cache for each text
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []
        
        for i, text in enumerate(texts):
            cached = self._get_cached_embedding(text)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            embeddings = self._model.encode(
                uncached_texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            
            for idx, (i, embedding) in enumerate(zip(uncached_indices, embeddings)):
                results[i] = embedding
                self._cache_embedding(uncached_texts[idx], embedding)
        
        return results
    
    def generate_for_chunks(
        self,
        chunks: List[Dict],
        text_key: str = "text"
    ) -> List[Dict]:
        """
        Generate embeddings for document chunks.
        
        Args:
            chunks: List of chunk dictionaries
            text_key: Key for text in chunk dict
        
        Returns:
            Chunks with embeddings added
        """
        texts = [chunk.get(text_key, "") for chunk in chunks]
        embeddings = self.generate_batch(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding.tolist()
        
        return chunks
    
    def similarity(
        self,
        embedding1: Union[np.ndarray, List[float]],
        embedding2: Union[np.ndarray, List[float]]
    ) -> float:
        """
        Calculate cosine similarity between embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
        
        Returns:
            Cosine similarity score
        """
        e1 = np.array(embedding1) if isinstance(embedding1, list) else embedding1
        e2 = np.array(embedding2) if isinstance(embedding2, list) else embedding2
        
        dot_product = np.dot(e1, e2)
        norm1 = np.linalg.norm(e1)
        norm2 = np.linalg.norm(e2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def find_similar(
        self,
        query_embedding: np.ndarray,
        embeddings: List[np.ndarray],
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[tuple]:
        """
        Find most similar embeddings.
        
        Args:
            query_embedding: Query embedding
            embeddings: List of embeddings to search
            top_k: Number of results
            threshold: Minimum similarity
        
        Returns:
            List of (index, similarity) tuples
        """
        similarities = []
        for i, emb in enumerate(embeddings):
            sim = self.similarity(query_embedding, emb)
            if sim >= threshold:
                similarities.append((i, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        if self.cache_dir.exists():
            for f in self.cache_dir.glob("*.npy"):
                f.unlink()
            self.logger.info("Embedding cache cleared")
