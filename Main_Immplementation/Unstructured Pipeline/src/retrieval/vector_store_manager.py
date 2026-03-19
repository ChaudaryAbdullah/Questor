"""
Vector Store Manager Module
Manages vector storage using ChromaDB for similarity search.
"""

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

from ..utils.logger import get_retrieval_logger
from ..utils.config_manager import get_config


@dataclass
class SearchResult:
    """Result from vector similarity search."""
    id: str
    text: str
    score: float
    metadata: Dict


class VectorStoreManager:
    """
    Manages vector storage and retrieval using ChromaDB.
    """
    
    def __init__(
        self,
        collection_name: str = "sec_filings",
        persist_directory: Optional[str] = None,
        distance_metric: str = "cosine"
    ):
        """
        Initialize vector store manager.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory for persistent storage
            distance_metric: Distance metric (cosine, l2, ip)
        """
        self.logger = get_retrieval_logger()
        self.collection_name = collection_name
        
        # Load config
        try:
            config = get_config()
            vdb_config = config.vector_db_config
            self.persist_directory = persist_directory or vdb_config.get(
                "persist_directory", "data/vectors/chromadb"
            )
            self.distance_metric = vdb_config.get("distance_metric", distance_metric)
        except Exception:
            self.persist_directory = persist_directory or "data/vectors/chromadb"
            self.distance_metric = distance_metric
        
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        self._client = None
        self._collection = None
        self._initialized = False
        
        self.logger.info(f"VectorStoreManager initialized for collection: {collection_name}")
    
    def _init_client(self) -> None:
        """Initialize ChromaDB client."""
        if self._initialized:
            return
        
        try:
            import chromadb
            from chromadb.config import Settings
            
            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance_metric}
            )
            
            self._initialized = True
            self.logger.info(f"ChromaDB initialized. Collection: {self.collection_name}")
            
        except ImportError:
            self.logger.error("chromadb not installed")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def add(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            texts: List of text content
            embeddings: List of embedding vectors
            metadatas: Optional list of metadata dicts
            ids: Optional list of IDs
        
        Returns:
            List of document IDs
        """
        self._init_client()
        
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Ensure metadatas are serializable
        clean_metadatas = []
        for meta in metadatas:
            clean_meta = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                elif isinstance(v, list):
                    clean_meta[k] = str(v)
                else:
                    clean_meta[k] = str(v)
            clean_metadatas.append(clean_meta)
        
        self._collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=clean_metadatas,
            ids=ids
        )
        
        self.logger.debug(f"Added {len(texts)} documents to collection")
        return ids
    
    def add_chunks(
        self,
        chunks: List[Dict],
        text_key: str = "text",
        embedding_key: str = "embedding",
        id_key: str = "id"
    ) -> List[str]:
        """
        Add document chunks to vector store.
        
        Args:
            chunks: List of chunk dictionaries
            text_key: Key for text in chunk
            embedding_key: Key for embedding in chunk
            id_key: Key for ID in chunk
        
        Returns:
            List of document IDs
        """
        texts = []
        embeddings = []
        metadatas = []
        ids = []
        
        for chunk in chunks:
            texts.append(chunk.get(text_key, ""))
            embeddings.append(chunk.get(embedding_key, []))
            
            # Build metadata excluding text and embedding
            metadata = {k: v for k, v in chunk.items() 
                       if k not in [text_key, embedding_key]}
            metadatas.append(metadata)
            
            ids.append(chunk.get(id_key, str(uuid.uuid4())))
        
        return self.add(texts, embeddings, metadatas, ids)
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_metadata: Optional[Dict] = None
    ) -> List[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
        
        Returns:
            List of SearchResult objects
        """
        self._init_client()
        
        where_filter = filter_metadata if filter_metadata else None
        
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        search_results = []
        if results and results["ids"]:
            for i, doc_id in enumerate(results["ids"][0]):
                # Convert distance to similarity score
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1 - distance if self.distance_metric == "cosine" else 1 / (1 + distance)
                
                search_results.append(SearchResult(
                    id=doc_id,
                    text=results["documents"][0][i] if results["documents"] else "",
                    score=score,
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {}
                ))
        
        return search_results
    
    def search_by_text(
        self,
        query_text: str,
        embedding_generator,
        top_k: int = 10,
        filter_metadata: Optional[Dict] = None
    ) -> List[SearchResult]:
        """
        Search using text query (generates embedding automatically).
        
        Args:
            query_text: Text query
            embedding_generator: EmbeddingGenerator instance
            top_k: Number of results
            filter_metadata: Optional metadata filter
        
        Returns:
            List of SearchResult objects
        """
        query_embedding = embedding_generator.generate(query_text).tolist()
        return self.search(query_embedding, top_k, filter_metadata)
    
    def get(self, ids: List[str]) -> List[Dict]:
        """
        Get documents by IDs.
        
        Args:
            ids: List of document IDs
        
        Returns:
            List of document dictionaries
        """
        self._init_client()
        
        results = self._collection.get(
            ids=ids,
            include=["documents", "metadatas", "embeddings"]
        )
        
        documents = []
        for i, doc_id in enumerate(results["ids"]):
            documents.append({
                "id": doc_id,
                "text": results["documents"][i] if results["documents"] else "",
                "metadata": results["metadatas"][i] if results["metadatas"] else {},
                "embedding": results["embeddings"][i] if results.get("embeddings") else None
            })
        
        return documents
    
    def delete(self, ids: List[str]) -> None:
        """
        Delete documents by IDs.
        
        Args:
            ids: List of document IDs to delete
        """
        self._init_client()
        self._collection.delete(ids=ids)
        self.logger.debug(f"Deleted {len(ids)} documents")
    
    def count(self) -> int:
        """Get total number of documents in collection."""
        self._init_client()
        return self._collection.count()
    
    def clear(self) -> None:
        """Clear all documents from collection."""
        self._init_client()
        
        # Get all IDs and delete
        results = self._collection.get()
        if results["ids"]:
            self._collection.delete(ids=results["ids"])
        
        self.logger.info("Collection cleared")
    
    def create_collection(self, name: str) -> None:
        """Create a new collection."""
        self._init_client()
        
        self._collection = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": self.distance_metric}
        )
        self.collection_name = name
        self.logger.info(f"Collection created: {name}")
    
    def list_collections(self) -> List[str]:
        """List all collections."""
        self._init_client()
        return [c.name for c in self._client.list_collections()]
    
    def get_collection_stats(self) -> Dict:
        """Get collection statistics."""
        self._init_client()
        
        return {
            "collection_name": self.collection_name,
            "document_count": self.count(),
            "persist_directory": self.persist_directory,
            "distance_metric": self.distance_metric
        }
