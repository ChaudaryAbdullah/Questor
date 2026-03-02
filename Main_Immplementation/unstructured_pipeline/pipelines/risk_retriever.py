#!/usr/bin/env python3
"""
Risk Retriever Module
Retrieves existing data from ChromaDB/Neo4j and calculates risk scores by CIK
"""
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from databases.vector_db import VectorDatabase
from databases.graph_db import GraphDatabase
from pipelines.risk_scorer import RiskScorer
from pipelines.output_formatter import OutputFormatter
from utils import Config, Logger


class RiskRetriever:
    """
    Retrieve existing data and calculate risk scores by CIK number
    """
    
    def __init__(self):
        """Initialize retriever with database connections"""
        self.logger = Logger.get_logger(self.__class__.__name__)
        
        # Initialize databases
        self.vector_db = VectorDatabase()
        self.graph_db = GraphDatabase()
        
        # Initialize risk scorer and output formatter
        self.risk_scorer = RiskScorer()
        self.output_formatter = OutputFormatter()
        
        self.logger.info("Risk Retriever initialized")
    
    def get_documents_by_cik(self, cik: str) -> List[Dict[str, Any]]:
        """
        Retrieve all documents for a specific CIK from Neo4j and ChromaDB
        Falls back to ChromaDB-only if Neo4j is empty
        
        Args:
            cik: Normalized CIK number (without leading zeros)
            
        Returns:
            List of document dictionaries with entities and content
        """
        self.logger.info(f"Retrieving documents for CIK: {cik}")
        
        # Try Neo4j first
        query = """
        MATCH (d:Document)
        WHERE d.company_id = $cik
        RETURN d.doc_id as doc_id,
               d.label as label,
               d.company_id as company_id,
               d.date as date,
               d.file_name as file_name
        """
        
        try:
            results = self.graph_db.query_graph(query, {'cik': cik})
            
            if not results:
                self.logger.info(f"No documents in Neo4j for CIK: {cik}, falling back to ChromaDB")
                return self._get_documents_from_chromadb_only(cik)
            
            # Neo4j has data - use it
            documents = []
            for record in results:
                doc_id = record.get('doc_id', 'unknown')
                
                # Reconstruct document structure
                doc = {
                    'doc_id': doc_id,
                    'label': record.get('label', 'unknown'),
                    'company_id': record.get('company_id'),
                    'date': record.get('date'),
                    'file_name': record.get('file_name', ''),
                    'content': '',
                    'entities': {},
                    'relationships': []
                }
                
                # Get entities from Neo4j
                doc['entities'] = self._get_entities_for_document(doc_id)
                
                # Get content from ChromaDB
                doc['content'] = self._get_content_from_chromadb(doc_id)
                
                # Get chunks for this document
                doc['chunks'] = self._get_chunks_from_chromadb(doc_id)
                
                documents.append(doc)
            
            self.logger.info(f"Retrieved {len(documents)} documents for CIK: {cik}")
            return documents
            
        except Exception as e:
            self.logger.error(f"Failed to query Neo4j for CIK {cik}: {str(e)}")
            self.logger.info("Falling back to ChromaDB-only mode")
            return self._get_documents_from_chromadb_only(cik)
    
    def _get_documents_from_chromadb_only(self, cik: str) -> List[Dict[str, Any]]:
        """
        Get documents directly from ChromaDB when Neo4j is unavailable
        
        Args:
            cik: Normalized CIK number
            
        Returns:
            List of document dictionaries
        """
        try:
            # Query ChromaDB for all chunks with this company_id
            results = self.vector_db.collection.get(
                where={"company_id": cik},
                include=["metadatas", "documents"]
            )
            
            if not results or not results.get('ids'):
                self.logger.warning(f"No documents found in ChromaDB for CIK: {cik}")
                return []
            
            self.logger.info(f"Found {len(results['ids'])} chunks in ChromaDB for CIK: {cik}")
            
            # Group chunks by doc_id
            doc_chunks = {}
            for i, chunk_id in enumerate(results['ids']):
                metadata = results['metadatas'][i] if results.get('metadatas') else {}
                doc_id = metadata.get('doc_id', 'unknown')
                
                if doc_id not in doc_chunks:
                    doc_chunks[doc_id] = {
                        'doc_id': doc_id,
                        'company_id': cik,
                        'label': metadata.get('label', 'unknown'),
                        'date': metadata.get('date'),
                        'file_name': metadata.get('file_name', ''),
                        'content': '',
                        'chunks': [],
                        'entities': {},
                        'relationships': []
                    }
                
                # Add chunk
                doc_chunks[doc_id]['chunks'].append({
                    'chunk_id': chunk_id,
                    'doc_id': doc_id,
                    'text': results['documents'][i] if results.get('documents') else '',
                    'metadata': metadata
                })
            
            # Combine chunks into content
            documents = []
            for doc_id, doc_data in doc_chunks.items():
                doc_data['content'] = ' '.join(chunk['text'] for chunk in doc_data['chunks'])
                documents.append(doc_data)
            
            self.logger.info(f"Retrieved {len(documents)} documents from ChromaDB for CIK: {cik}")
            return documents
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve from ChromaDB for CIK {cik}: {str(e)}")
            return []
    
    def _get_entities_for_document(self, doc_id: str) -> Dict[str, List[Dict]]:
        """Get entities for a document from Neo4j"""
        entity_query = """
        MATCH (d:Document {doc_id: $doc_id})-[:MENTIONS]->(e)
        RETURN e.name as name, e.type as entity_type, labels(e)[0] as label
        """
        
        try:
            entity_results = self.graph_db.query_graph(entity_query, {'doc_id': doc_id})
            
            # Group entities by type
            entities_dict = {}
            for ent_record in entity_results:
                entity_name = ent_record.get('name', '')
                entity_type = ent_record.get('entity_type') or ent_record.get('label', 'UNKNOWN')
                
                if entity_name:  # Filter out empty entities
                    if entity_type not in entities_dict:
                        entities_dict[entity_type] = []
                    entities_dict[entity_type].append({
                        'text': entity_name,
                        'label': entity_type
                    })
            
            return entities_dict
            
        except Exception as e:
            self.logger.debug(f"No entities found for {doc_id}: {str(e)}")
            return {}
    
    def _get_content_from_chromadb(self, doc_id: str) -> str:
        """Get document content from ChromaDB chunks"""
        try:
            results = self.vector_db.collection.get(
                where={"doc_id": doc_id},
                include=["documents"]
            )
            
            if results and results.get('documents'):
                # Combine all chunks
                content = ' '.join(results['documents'])
                return content
            else:
                return 'No content available'
                
        except Exception as e:
            self.logger.debug(f"No content found in ChromaDB for {doc_id}: {str(e)}")
            return 'No content available'
    
    def _get_chunks_from_chromadb(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get chunks for a document from ChromaDB"""
        try:
            results = self.vector_db.collection.get(
                where={"doc_id": doc_id},
                include=["metadatas", "documents"]
            )
            
            chunks = []
            if results and results.get('ids'):
                for i, chunk_id in enumerate(results['ids']):
                    chunk = {
                        'chunk_id': chunk_id,
                        'doc_id': doc_id,
                        'text': results['documents'][i] if results.get('documents') else '',
                        'metadata': results['metadatas'][i] if results.get('metadatas') else {}
                    }
                    chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            self.logger.warning(f"Failed to retrieve chunks for {doc_id}: {str(e)}")
            return []
    
    def calculate_risk_for_cik(
        self,
        cik: str,
        include_chunks: bool = False
    ) -> Dict[str, Any]:
        """
        Calculate risk scores for a specific CIK
        
        Args:
            cik: Normalized CIK number
            include_chunks: Whether to include chunk details
            
        Returns:
            Dictionary with risk data and documents
        """
        self.logger.info(f"Calculating risk for CIK: {cik}")
        
        # Get documents for this CIK
        documents = self.get_documents_by_cik(cik)
        
        if not documents:
            return {
                'success': False,
                'cik': cik,
                'error': 'No documents found for this CIK',
                'documents': []
            }
        
        # Calculate risk for each document
        documents_with_risk = []
        
        for doc in tqdm(documents, desc=f"Calculating risk for CIK {cik}", leave=False):
            try:
                risk_data = self.risk_scorer.calculate_document_risk(
                    document=doc,
                    entities=doc.get('entities'),
                    relationships=doc.get('relationships')
                )
                
                doc['risk_data'] = risk_data
                
                if not include_chunks:
                    doc['chunks'] = []
                
                documents_with_risk.append(doc)
                
            except Exception as e:
                self.logger.error(f"Failed to calculate risk for {doc.get('doc_id')}: {str(e)}")
                continue
        
        # Calculate aggregate risk for this CIK
        risk_scores = [
            doc.get('risk_data', {}).get('overall_risk_score', 0)
            for doc in documents_with_risk
        ]
        
        avg_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 0
        max_risk = max(risk_scores) if risk_scores else 0
        
        return {
            'success': True,
            'cik': cik,
            'documents_count': len(documents_with_risk),
            'average_risk_score': round(avg_risk, 2),
            'max_risk_score': round(max_risk, 2),
            'documents': documents_with_risk
        }
    
    def process_multiple_ciks(
        self,
        ciks: List[str],
        include_chunks: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Process multiple CIKs
        
        Args:
            ciks: List of normalized CIK numbers
            include_chunks: Whether to include chunk details
            
        Returns:
            List of results for each CIK
        """
        self.logger.info(f"Processing {len(ciks)} CIKs")
        
        results = []
        for cik in ciks:
            result = self.calculate_risk_for_cik(cik, include_chunks)
            results.append(result)
        
        return results
    
    def format_for_multiagent(
        self,
        cik_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Format CIK result for multiagent system
        
        Args:
            cik_result: Result from calculate_risk_for_cik
            
        Returns:
            Formatted output for multiagent system
        """
        if not cik_result.get('success'):
            return {
                'cik': cik_result.get('cik'),
                'error': cik_result.get('error'),
                'risk_assessment': {
                    'overall_risk_score': 0.0,
                    'risk_level': 'UNKNOWN',
                    'risk_factors': ['No data available']
                }
            }
        
        documents = cik_result.get('documents', [])
        
        # Use the document with highest risk as primary
        if documents:
            # Sort by risk score
            sorted_docs = sorted(
                documents,
                key=lambda d: d.get('risk_data', {}).get('overall_risk_score', 0),
                reverse=True
            )
            primary_doc = sorted_docs[0]
            
            # Format using output formatter
            formatted = self.output_formatter.format_for_multiagent(
                document=primary_doc,
                risk_data=primary_doc.get('risk_data', {}),
                entities=primary_doc.get('entities'),
                relationships=primary_doc.get('relationships'),
                chunks=primary_doc.get('chunks', [])
            )
            
            # Add CIK-level metadata
            formatted['cik'] = cik_result.get('cik')
            formatted['documents_analyzed'] = cik_result.get('documents_count', 0)
            formatted['average_risk_score'] = cik_result.get('average_risk_score', 0)
            
            return formatted
        
        return {
            'cik': cik_result.get('cik'),
            'error': 'No documents processed',
            'risk_assessment': {
                'overall_risk_score': 0.0,
                'risk_level': 'UNKNOWN',
                'risk_factors': ['No documents processed']
            }
        }
    
    def close(self):
        """Close database connections"""
        self.logger.info("Connections closed")


def main():
    """Test the risk retriever"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Retrieve risk scores by CIK')
    parser.add_argument('--cik', type=str, required=True, help='CIK number (without leading zeros)')
    parser.add_argument('--include-chunks', action='store_true', help='Include chunk details')
    
    args = parser.parse_args()
    
    retriever = RiskRetriever()
    
    try:
        result = retriever.calculate_risk_for_cik(args.cik, args.include_chunks)
        
        if result['success']:
            print(f"\n✓ CIK: {result['cik']}")
            print(f"  Documents: {result['documents_count']}")
            print(f"  Average Risk: {result['average_risk_score']:.2f}")
            print(f"  Max Risk: {result['max_risk_score']:.2f}")
            
            # Format for multiagent
            formatted = retriever.format_for_multiagent(result)
            print(f"\n  Risk Level: {formatted.get('risk_assessment', {}).get('risk_level')}")
            print(f"  Overall Score: {formatted.get('risk_assessment', {}).get('overall_score', 0):.2f}")
        else:
            print(f"\n✗ Failed: {result.get('error')}")
    
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        retriever.close()


if __name__ == "__main__":
    main()
