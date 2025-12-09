# File: pipelines/unstructured_pipeline.py
"""
Main Unstructured Data Processing Pipeline
This is the complete Pipeline 1 from your dry run document
"""
from typing import Optional, Dict, Any
from pathlib import Path
from tqdm import tqdm
import time

from databases import VectorDatabase, GraphDatabase
from pipelines import (
    DataLoader,
    TextChunker,
    EmbeddingGenerator,
    NERExtractor,
    GraphBuilder
)
from utils import Config, Logger


class UnstructuredPipeline:
    """
    Complete unstructured data processing pipeline
    
    This pipeline:
    1. Loads documents from data/unstructured_data
    2. Chunks text into manageable pieces
    3. Generates embeddings for each chunk
    4. Stores embeddings in vector database (ChromaDB)
    5. Extracts entities and relationships using NER
    6. Builds knowledge graph (Neo4j)
    """
    
    def __init__(self):
        self.logger = Logger.get_logger(self.__class__.__name__)
        
        # Initialize components
        self.data_loader = DataLoader()
        self.text_chunker = TextChunker()
        self.embedding_generator = EmbeddingGenerator()
        self.ner_extractor = NERExtractor()
        self.graph_builder = GraphBuilder()
        
        # Initialize databases
        self.vector_db = VectorDatabase()
        self.graph_db = GraphDatabase()
        
        self.logger.info("Unstructured pipeline initialized")
    
    def run(
        self,
        limit: Optional[int] = None,
        skip_embeddings: bool = False,
        skip_graph: bool = False
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline
        
        Args:
            limit: Maximum number of documents to process (None for all)
            skip_embeddings: Skip vector embedding generation
            skip_graph: Skip knowledge graph construction
            
        Returns:
            Dictionary containing pipeline statistics
        """
        start_time = time.time()
        
        self.logger.info("=" * 80)
        self.logger.info("STARTING UNSTRUCTURED DATA PIPELINE")
        self.logger.info("=" * 80)
        
        try:
            # Step 1: Load documents
            documents = self._step1_load_documents(limit)
            
            # Step 2: Chunk documents
            chunks = self._step2_chunk_documents(documents)
            
            # Step 3: Generate embeddings and store in vector DB
            if not skip_embeddings:
                embedding_stats = self._step3_generate_embeddings(chunks)
            else:
                embedding_stats = {'status': 'skipped'}
            
            # Step 4: Extract entities and relationships
            documents_with_entities = self._step4_extract_entities(documents)
            
            # Step 5: Build knowledge graph
            if not skip_graph:
                graph_stats = self._step5_build_knowledge_graph(documents_with_entities)
            else:
                graph_stats = {'status': 'skipped'}
            
            # Calculate statistics
            elapsed_time = time.time() - start_time
            
            stats = {
                'success': True,
                'elapsed_time': elapsed_time,
                'documents_processed': len(documents),
                'chunks_created': len(chunks),
                'embedding_stats': embedding_stats,
                'graph_stats': graph_stats
            }
            
            self.logger.info("=" * 80)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info(f"Total time: {elapsed_time:.2f} seconds")
            self.logger.info("=" * 80)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'elapsed_time': time.time() - start_time
            }
    
    def _step1_load_documents(self, limit: Optional[int]) -> list:
        """Step 1: Load documents from data directory"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STEP 1: LOADING DOCUMENTS")
        self.logger.info("=" * 80)
        
        # Get statistics first
        stats = self.data_loader.get_document_statistics()
        self.logger.info(f"Data directory: {stats['data_directory']}")
        self.logger.info(f"Total documents found: {stats['total_documents']}")
        self.logger.info(f"Fraud labeled: {stats['fraud_labeled']}")
        self.logger.info(f"Unknown: {stats['unknown']}")
        
        if limit:
            self.logger.info(f"Processing limit: {limit} documents")
        
        documents = self.data_loader.load_documents(limit=limit)
        
        self.logger.info(f"✓ Loaded {len(documents)} documents")
        return documents
    
    def _step2_chunk_documents(self, documents: list) -> list:
        """Step 2: Chunk documents into smaller pieces"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STEP 2: CHUNKING DOCUMENTS")
        self.logger.info("=" * 80)
        self.logger.info(f"Chunk size: {Config.CHUNK_SIZE} tokens")
        self.logger.info(f"Chunk overlap: {Config.CHUNK_OVERLAP} tokens")
        
        all_chunks = []
        
        for doc in tqdm(documents, desc="Chunking documents"):
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
        
        self.logger.info(f"✓ Created {len(all_chunks)} chunks from {len(documents)} documents")
        self.logger.info(f"  Average chunks per document: {len(all_chunks) / len(documents):.2f}")
        
        return all_chunks
    
    def _step3_generate_embeddings(self, chunks: list) -> Dict[str, Any]:
        """Step 3: Generate embeddings and store in vector database"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STEP 3: GENERATING EMBEDDINGS & POPULATING VECTOR DB")
        self.logger.info("=" * 80)
        self.logger.info(f"Embedding model: {Config.EMBEDDING_MODEL}")
        self.logger.info(f"Batch size: {Config.BATCH_SIZE}")
        
        # Extract texts
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_generator.generate_embeddings(
            texts=texts,
            batch_size=Config.BATCH_SIZE,
            show_progress=True
        )
        
        # Prepare metadata for vector DB
        metadatas = []
        ids = []
        
        for chunk in chunks:
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
        
        # Store in vector database
        self.logger.info("Storing embeddings in ChromaDB...")
        self.vector_db.add_documents(
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )
        
        # Verify storage
        count = self.vector_db.get_collection_count()
        
        stats = {
            'chunks_embedded': len(chunks),
            'embedding_dimension': embeddings.shape[1],
            'vector_db_count': count
        }
        
        self.logger.info(f"✓ Generated and stored {len(chunks)} embeddings")
        self.logger.info(f"  Embedding dimension: {embeddings.shape[1]}")
        self.logger.info(f"  Vector DB total count: {count}")
        
        return stats
    
    def _step4_extract_entities(self, documents: list) -> list:
        """Step 4: Extract entities and relationships using NER"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STEP 4: EXTRACTING ENTITIES & RELATIONSHIPS")
        self.logger.info("=" * 80)
        self.logger.info(f"NLP model: {Config.SPACY_MODEL}")
        
        # Extract entities from all documents
        documents_with_entities = self.ner_extractor.extract_document_entities(documents)
        
        # Calculate statistics
        total_entities = sum(
            len(entities)
            for doc in documents_with_entities
            for entities in doc.get('entities', {}).values()
        )
        
        total_relationships = sum(
            len(doc.get('relationships', []))
            for doc in documents_with_entities
        )
        
        self.logger.info(f"✓ Extracted entities from {len(documents_with_entities)} documents")
        self.logger.info(f"  Total entities: {total_entities}")
        self.logger.info(f"  Total relationships: {total_relationships}")
        
        return documents_with_entities
    
    def _step5_build_knowledge_graph(self, documents: list) -> Dict[str, Any]:
        """Step 5: Build knowledge graph in Neo4j"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STEP 5: BUILDING KNOWLEDGE GRAPH")
        self.logger.info("=" * 80)
        self.logger.info(f"Graph database: {Config.NEO4J_URI}")
        
        # Build graph
        stats = self.graph_builder.build_graph_from_documents(documents)
        
        self.logger.info(f"✓ Knowledge graph constructed")
        self.logger.info(f"  Documents processed: {stats['documents_processed']}")
        self.logger.info(f"  Entities added: {stats['entities_added']}")
        self.logger.info(f"  Relationships added: {stats['relationships_added']}")
        
        return stats
    
    def query_vector_db(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Query the vector database for similar documents
        
        Args:
            query_text: Query text
            n_results: Number of results to return
            
        Returns:
            Query results
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_single_embedding(query_text)
        
        # Query vector DB
        results = self.vector_db.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        
        return results
    
    def query_knowledge_graph(self, cypher_query: str) -> list:
        """
        Query the knowledge graph
        
        Args:
            cypher_query: Cypher query string
            
        Returns:
            Query results
        """
        return self.graph_db.query_graph(cypher_query)
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current status of the pipeline"""
        return {
            'vector_db_count': self.vector_db.get_collection_count(),
            'data_directory': str(Config.DATA_DIR),
            'vector_db_collection': Config.VECTOR_DB_COLLECTION,
            'embedding_model': Config.EMBEDDING_MODEL,
            'nlp_model': Config.SPACY_MODEL
        }
    
    def reset_pipeline(self, confirm: bool = False):
        """
        Reset the entire pipeline (WARNING: deletes all data)
        
        Args:
            confirm: Must be True to actually reset
        """
        if not confirm:
            self.logger.warning("Reset not confirmed. Pass confirm=True to reset.")
            return
        
        self.logger.warning("RESETTING PIPELINE - ALL DATA WILL BE DELETED")
        
        # Reset vector database
        self.vector_db.reset_database()
        
        # Clear graph database
        self.graph_db.clear_database()
        
        self.logger.warning("Pipeline reset complete")
    
    def close(self):
        """Close all database connections"""
        self.graph_builder.close()
        self.logger.info("Pipeline connections closed")

