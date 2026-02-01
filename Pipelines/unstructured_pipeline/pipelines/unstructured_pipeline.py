# File: pipelines/unstructured_pipeline_optimized.py
"""
Memory-Optimized Unstructured Data Processing Pipeline
Implements streaming, batching, and proper resource management
"""
from typing import Optional, Dict, Any, Iterator
from pathlib import Path
from tqdm import tqdm
import time
import gc
import sys
import os

# Ensure proper imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Direct imports
from databases.vector_db import VectorDatabase
from databases.graph_db import GraphDatabase
from pipelines.data_loader import DataLoader
from pipelines.chunking import TextChunker
from pipelines.embedding import EmbeddingGenerator
from pipelines.ner_extraction import NERExtractorOptimized
from pipelines.graph_builder import GraphBuilder

try:
    from utils.config_optimized import ConfigOptimized as Config
except ImportError:
    from utils.config import ConfigOptimized as Config

try:
    from utils.logger import Logger
except ImportError:
    from utils import Logger


class UnstructuredPipelineOptimized:
    """
    Memory-optimized unstructured data processing pipeline
    
    Key optimizations:
    1. Streaming document processing (one at a time)
    2. Batch processing with configurable batch sizes
    3. Explicit memory cleanup with gc.collect()
    4. Progressive database writes instead of bulk loading
    5. Lazy loading of models
    """
    
    def __init__(self, batch_size: int = 10):
        """
        Initialize pipeline with memory-conscious settings
        
        Args:
            batch_size: Number of documents to process before writing to DB
        """
        self.logger = Logger.get_logger(self.__class__.__name__)
        self.batch_size = batch_size
        
        # Initialize components (lazy loading where possible)
        self.data_loader = DataLoader()
        self.text_chunker = TextChunker()
        
        # These will be initialized when needed
        self.embedding_generator = None
        self.ner_extractor = None  # Will use NERExtractorOptimized
        self.graph_builder = None
        
        # Initialize databases
        self.vector_db = VectorDatabase()
        self.graph_db = GraphDatabase()
        
        self.logger.info(f"Optimized pipeline initialized with batch_size={batch_size}")
    
    def run(
        self,
        limit: Optional[int] = None,
        skip_embeddings: bool = False,
        skip_graph: bool = False,
        process_batch_size: int = None
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline with memory optimization
        
        Args:
            limit: Maximum number of documents to process (None for all)
            skip_embeddings: Skip vector embedding generation
            skip_graph: Skip knowledge graph construction
            process_batch_size: Override default batch size
            
        Returns:
            Dictionary containing pipeline statistics
        """
        start_time = time.time()
        batch_size = process_batch_size or self.batch_size
        
        self.logger.info("=" * 80)
        self.logger.info("STARTING OPTIMIZED UNSTRUCTURED DATA PIPELINE")
        self.logger.info(f"Batch size: {batch_size}")
        self.logger.info("=" * 80)
        
        stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'entities_extracted': 0,
            'graph_nodes_added': 0,
            'graph_relationships_added': 0,
            'peak_memory_mb': 0
        }
        
        try:
            # Get file list
            file_paths = self._get_file_list(limit)
            total_files = len(file_paths)
            
            self.logger.info(f"Processing {total_files} documents in batches of {batch_size}")
            
            # Process in batches
            for batch_start in range(0, total_files, batch_size):
                batch_end = min(batch_start + batch_size, total_files)
                batch_files = file_paths[batch_start:batch_end]
                
                self.logger.info(f"\nProcessing batch {batch_start//batch_size + 1}: "
                               f"documents {batch_start+1} to {batch_end}")
                
                # Process this batch
                batch_stats = self._process_batch(
                    batch_files,
                    skip_embeddings=skip_embeddings,
                    skip_graph=skip_graph
                )
                
                # Update overall stats
                stats['documents_processed'] += batch_stats['documents_processed']
                stats['chunks_created'] += batch_stats['chunks_created']
                stats['entities_extracted'] += batch_stats.get('entities_extracted', 0)
                stats['graph_nodes_added'] += batch_stats.get('graph_nodes_added', 0)
                stats['graph_relationships_added'] += batch_stats.get('graph_relationships_added', 0)
                
                # Force garbage collection after each batch
                gc.collect()
                
                self.logger.info(f"Batch complete. Total progress: {stats['documents_processed']}/{total_files}")
            
            # Calculate final statistics
            elapsed_time = time.time() - start_time
            
            final_stats = {
                'success': True,
                'elapsed_time': elapsed_time,
                **stats,
                'embeddings_skipped': skip_embeddings,
                'graph_skipped': skip_graph
            }
            
            self.logger.info("=" * 80)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info(f"Total time: {elapsed_time:.2f} seconds")
            self.logger.info(f"Documents processed: {stats['documents_processed']}")
            self.logger.info(f"Chunks created: {stats['chunks_created']}")
            self.logger.info("=" * 80)
            
            return final_stats
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'elapsed_time': time.time() - start_time,
                **stats
            }
        finally:
            # Cleanup
            self._cleanup_models()
    
    def _get_file_list(self, limit: Optional[int]) -> list:
        """Get list of file paths to process"""
        files = list(Config.DATA_DIR.glob("*.txt"))
        if limit:
            files = files[:limit]
        return files
    
    def _process_batch(
        self,
        file_paths: list,
        skip_embeddings: bool = False,
        skip_graph: bool = False
    ) -> Dict[str, Any]:
        """
        Process a batch of documents
        
        Args:
            file_paths: List of file paths in this batch
            skip_embeddings: Skip embedding generation
            skip_graph: Skip graph construction
            
        Returns:
            Batch statistics
        """
        batch_stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'entities_extracted': 0,
            'graph_nodes_added': 0,
            'graph_relationships_added': 0
        }
        
        # Step 1: Load documents (one batch at a time)
        documents = []
        for file_path in file_paths:
            doc = self.data_loader._load_single_document(file_path)
            if doc:
                documents.append(doc)
        
        batch_stats['documents_processed'] = len(documents)
        
        if not documents:
            return batch_stats
        
        # Step 2: Chunk documents
        all_chunks = []
        for doc in documents:
            chunks = self.text_chunker.chunk_text(
                text=doc['content'],
                doc_id=doc['doc_id']
            )
            
            # Add document metadata to each chunk
            for chunk in chunks:
                chunk['label'] = doc.get('label', 'unknown')
                chunk['company_id'] = doc.get('company_id')
                chunk['date'] = doc.get('date')
            
            all_chunks.extend(chunks)
        
        batch_stats['chunks_created'] = len(all_chunks)
        
        # Step 3: Generate embeddings (if not skipped)
        if not skip_embeddings and all_chunks:
            self._process_embeddings_batch(all_chunks)
        
        # Clear chunks from memory if not needed for graph
        if skip_graph:
            del all_chunks
            gc.collect()
        
        # Step 4: Extract entities (if graph is needed)
        if not skip_graph:
            # Initialize NER if needed - use optimized version
            if self.ner_extractor is None:
                self.ner_extractor = NERExtractorOptimized()
            
            # Process documents one by one for NER (memory intensive)
            documents_with_entities = []
            for doc in tqdm(documents, desc="Extracting entities", leave=False):
                try:
                    content = doc.get('content', '')
                    entities = self.ner_extractor.extract_entities(content)
                    relationships = self.ner_extractor.extract_relationships(content, entities)
                    
                    doc['entities'] = entities
                    doc['relationships'] = relationships
                    documents_with_entities.append(doc)
                    
                    # Count entities
                    batch_stats['entities_extracted'] += sum(
                        len(ent_list) for ent_list in entities.values()
                    )
                    
                except Exception as e:
                    self.logger.warning(f"Failed to extract from doc {doc.get('doc_id')}: {str(e)}")
                    continue
            
            # Step 5: Build knowledge graph
            if documents_with_entities:
                if self.graph_builder is None:
                    self.graph_builder = GraphBuilder()
                
                graph_stats = self.graph_builder.build_graph_from_documents(documents_with_entities)
                batch_stats['graph_nodes_added'] = graph_stats.get('entities_added', 0)
                batch_stats['graph_relationships_added'] = graph_stats.get('relationships_added', 0)
            
            # Clear documents with entities
            del documents_with_entities
        
        # Clear documents from memory
        del documents
        gc.collect()
        
        return batch_stats
    
    def _process_embeddings_batch(self, chunks: list):
        """
        Process embeddings in sub-batches to manage memory
        
        Args:
            chunks: List of text chunks
        """
        # Initialize embedding generator if needed
        if self.embedding_generator is None:
            self.embedding_generator = EmbeddingGenerator()
        
        # Process in smaller sub-batches
        embedding_batch_size = Config.BATCH_SIZE
        
        for i in range(0, len(chunks), embedding_batch_size):
            batch_chunks = chunks[i:i + embedding_batch_size]
            
            # Extract texts
            texts = [chunk['text'] for chunk in batch_chunks]
            
            # Generate embeddings
            embeddings = self.embedding_generator.generate_embeddings(
                texts=texts,
                batch_size=min(embedding_batch_size, 16),  # Smaller internal batch
                show_progress=False
            )
            
            # Prepare metadata
            metadatas = []
            ids = []
            
            for chunk in batch_chunks:
                metadata = {
                    'doc_id': chunk['doc_id'],
                    'chunk_index': chunk['chunk_index'],
                    'label': chunk.get('label', 'unknown'),
                    'company_id': str(chunk.get('company_id', '')),
                    'date': str(chunk.get('date', '')),
                    'length': chunk['length']
                }
                metadatas.append(metadata)
                ids.append(chunk['chunk_id'])
            
            # Store in vector database immediately
            self.vector_db.add_documents(
                documents=texts,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
                ids=ids
            )
            
            # Clear embeddings from memory
            del embeddings
            del texts
            gc.collect()
    
    def _cleanup_models(self):
        """Release model resources"""
        if self.embedding_generator is not None:
            del self.embedding_generator
            self.embedding_generator = None
        
        if self.ner_extractor is not None:
            del self.ner_extractor
            self.ner_extractor = None
        
        gc.collect()
        self.logger.info("Model resources released")
    
    def query_vector_db(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """Query the vector database for similar documents"""
        if self.embedding_generator is None:
            self.embedding_generator = EmbeddingGenerator()
        
        query_embedding = self.embedding_generator.generate_single_embedding(query_text)
        results = self.vector_db.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        
        return results
    
    def query_knowledge_graph(self, cypher_query: str) -> list:
        """Query the knowledge graph"""
        return self.graph_db.query_graph(cypher_query)
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current status of the pipeline"""
        return {
            'vector_db_count': self.vector_db.get_collection_count(),
            'data_directory': str(Config.DATA_DIR),
            'batch_size': self.batch_size,
            'vector_db_collection': Config.VECTOR_DB_COLLECTION,
            'embedding_model': Config.EMBEDDING_MODEL,
            'nlp_model': Config.SPACY_MODEL
        }
    
    def reset_pipeline(self, confirm: bool = False):
        """Reset the entire pipeline"""
        if not confirm:
            self.logger.warning("Reset not confirmed. Pass confirm=True to reset.")
            return
        
        self.logger.warning("RESETTING PIPELINE - ALL DATA WILL BE DELETED")
        self.vector_db.reset_database()
        self.graph_db.clear_database()
        self.logger.warning("Pipeline reset complete")
    
    def close(self):
        """Close all database connections and cleanup"""
        self._cleanup_models()
        if self.graph_builder:
            self.graph_builder.close()
        self.logger.info("Pipeline connections closed")