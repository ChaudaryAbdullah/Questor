# File: databases/vector_db.py
"""
Vector Database Interface using ChromaDB
"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Any
from pathlib import Path
import numpy as np

from utils import Config, Logger
from utils.exceptions import VectorDBError


class VectorDatabase:
    """Interface for ChromaDB vector database operations"""
    
    def __init__(self):
        self.logger = Logger.get_logger(self.__class__.__name__)
        self.client = None
        self.collection = None
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize ChromaDB client and collection"""
        try:
            self.logger.info("Initializing ChromaDB client...")
            
            # Create persistent client
            self.client = chromadb.PersistentClient(
                path=str(Config.VECTOR_DB_DIR),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=Config.VECTOR_DB_COLLECTION,
                metadata={"description": "Financial fraud detection documents"}
            )
            
            self.logger.info(f"Vector database initialized. Collection: {Config.VECTOR_DB_COLLECTION}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vector database: {str(e)}")
            raise VectorDBError(f"Database initialization failed: {str(e)}")
    
    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ) -> bool:
        """
        Add documents with embeddings to the vector database
        
        Args:
            documents: List of text chunks
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
            ids: List of unique document IDs
            
        Returns:
            bool: Success status
        """
        try:
            self.logger.info(f"Adding {len(documents)} documents to vector database...")
            
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            self.logger.info(f"Successfully added {len(documents)} documents")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add documents: {str(e)}")
            raise VectorDBError(f"Failed to add documents: {str(e)}")
    
    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 10,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Query the vector database for similar documents
        
        Args:
            query_embeddings: List of query embedding vectors
            n_results: Number of results to return
            where: Metadata filter
            where_document: Document content filter
            
        Returns:
            Dict containing query results
        """
        try:
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                where_document=where_document
            )
            
            self.logger.debug(f"Query returned {len(results['ids'][0])} results")
            return results
            
        except Exception as e:
            self.logger.error(f"Query failed: {str(e)}")
            raise VectorDBError(f"Query failed: {str(e)}")
    
    def get_collection_count(self) -> int:
        """Get total number of documents in collection"""
        try:
            return self.collection.count()
        except Exception as e:
            self.logger.error(f"Failed to get collection count: {str(e)}")
            return 0
    
    def delete_collection(self):
        """Delete the entire collection"""
        try:
            self.client.delete_collection(name=Config.VECTOR_DB_COLLECTION)
            self.logger.info(f"Collection {Config.VECTOR_DB_COLLECTION} deleted")
        except Exception as e:
            self.logger.error(f"Failed to delete collection: {str(e)}")
            raise VectorDBError(f"Failed to delete collection: {str(e)}")
    
    def reset_database(self):
        """Reset the entire database"""
        try:
            self.client.reset()
            self.logger.warning("Vector database has been reset")
            self._initialize_db()
        except Exception as e:
            self.logger.error(f"Failed to reset database: {str(e)}")
            raise VectorDBError(f"Failed to reset database: {str(e)}")

