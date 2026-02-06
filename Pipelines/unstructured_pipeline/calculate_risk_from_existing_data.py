#!/usr/bin/env python
# File: calculate_risk_from_existing_data.py
"""
Calculate risk scores from existing data in Neo4j and ChromaDB
WITHOUT reprocessing the entire dataset

This script:
1. Retrieves documents and entities from Neo4j
2. Retrieves metadata from ChromaDB
3. Calculates risk scores for each document
4. Exports formatted output for multiagent system
"""
import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm
import json

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from databases.vector_db import VectorDatabase
from databases.graph_db import GraphDatabase
from pipelines.risk_scorer import RiskScorer
from pipelines.output_formatter import OutputFormatter
from utils import Config, Logger


class ExistingDataRiskCalculator:
    """
    Calculate risk scores from existing Neo4j and ChromaDB data
    """
    
    def __init__(self):
        """Initialize calculator with database connections"""
        self.logger = Logger.get_logger(self.__class__.__name__)
        
        # Initialize databases
        self.vector_db = VectorDatabase()
        self.graph_db = GraphDatabase()
        
        # Initialize risk scorer and output formatter
        self.risk_scorer = RiskScorer()
        self.output_formatter = OutputFormatter()
        
        self.logger.info("Existing Data Risk Calculator initialized")
    
    def get_all_documents_from_graph(self) -> List[Dict[str, Any]]:
        """
        Retrieve all documents from Neo4j knowledge graph
        
        Returns:
            List of document dictionaries with entities and relationships
        """
        self.logger.info("Retrieving documents from Neo4j...")
        
        # Query to get all documents (content is NOT stored in Neo4j, only metadata)
        query = """
        MATCH (d:Document)
        RETURN d.doc_id as doc_id,
               d.label as label,
               d.company_id as company_id,
               d.date as date,
               d.file_name as file_name
        """
        
        try:
            results = self.graph_db.query_graph(query)
            
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
                    'content': '',  # Will be filled from ChromaDB
                    'entities': {},
                    'relationships': []
                }
                
                # Get entities using correct MENTIONS relationship
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
                    
                    doc['entities'] = entities_dict
                except Exception as e:
                    self.logger.debug(f"No entities found for {doc_id}: {str(e)}")
                    doc['entities'] = {}
                
                # Get content from ChromaDB
                try:
                    content_chunks = self.get_document_chunks_from_vector_db(doc_id)
                    if content_chunks:
                        # Combine all chunks into content
                        doc['content'] = ' '.join([chunk.get('text', '') for chunk in content_chunks])
                    else:
                        doc['content'] = 'No content available'
                except Exception as e:
                    self.logger.debug(f"No content found in ChromaDB for {doc_id}: {str(e)}")
                    doc['content'] = 'No content available'
                
                documents.append(doc)
            
            self.logger.info(f"Retrieved {len(documents)} documents from Neo4j")
            return documents
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve documents from Neo4j: {str(e)}")
            return []
    
    def get_document_chunks_from_vector_db(self, doc_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve chunks for a specific document from ChromaDB
        
        Args:
            doc_id: Document ID
            
        Returns:
            List of chunk dictionaries
        """
        try:
            # Query ChromaDB for chunks of this document
            # Note: ChromaDB doesn't have a direct "get by metadata" method,
            # so we'll use a workaround with query
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
    
    def calculate_risk_scores(
        self,
        documents: List[Dict[str, Any]],
        include_chunks: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Calculate risk scores for all documents
        
        Args:
            documents: List of document dictionaries
            include_chunks: Whether to retrieve and include chunks
            
        Returns:
            List of documents with risk scores
        """
        self.logger.info(f"Calculating risk scores for {len(documents)} documents...")
        
        documents_with_risk = []
        
        for doc in tqdm(documents, desc="Calculating risk scores"):
            try:
                # Calculate risk score
                risk_data = self.risk_scorer.calculate_document_risk(
                    document=doc,
                    entities=doc.get('entities'),
                    relationships=doc.get('relationships')
                )
                
                doc['risk_data'] = risk_data
                
                # Optionally retrieve chunks
                if include_chunks:
                    chunks = self.get_document_chunks_from_vector_db(doc['doc_id'])
                    doc['chunks'] = chunks
                else:
                    doc['chunks'] = []
                
                documents_with_risk.append(doc)
                
            except Exception as e:
                self.logger.error(f"Failed to calculate risk for {doc.get('doc_id')}: {str(e)}")
                continue
        
        self.logger.info(f"Successfully calculated risk for {len(documents_with_risk)} documents")
        return documents_with_risk
    
    def format_and_export(
        self,
        documents: List[Dict[str, Any]],
        batch_name: str = "existing_data_risk_analysis"
    ) -> Path:
        """
        Format documents and export for multiagent system
        
        Args:
            documents: List of documents with risk scores
            batch_name: Name for the batch export
            
        Returns:
            Path to exported file
        """
        self.logger.info("Formatting output for multiagent system...")
        
        formatted_outputs = []
        
        for doc in tqdm(documents, desc="Formatting outputs"):
            try:
                formatted_output = self.output_formatter.format_for_multiagent(
                    document=doc,
                    risk_data=doc.get('risk_data', {}),
                    entities=doc.get('entities'),
                    relationships=doc.get('relationships'),
                    chunks=doc.get('chunks', [])
                )
                formatted_outputs.append(formatted_output)
            except Exception as e:
                self.logger.warning(f"Failed to format {doc.get('doc_id')}: {str(e)}")
                continue
        
        # Create batch output
        batch_output = self.output_formatter.format_batch_for_multiagent(
            [{'document': doc, 'risk_data': doc.get('risk_data', {}),
              'entities': doc.get('entities'), 'relationships': doc.get('relationships'),
              'chunks': doc.get('chunks', [])} for doc in documents]
        )
        
        # Override with formatted outputs
        batch_output['documents'] = formatted_outputs
        
        # Save to file
        output_path = self.output_formatter.save_batch_output(batch_output, batch_name)
        
        # Generate and save summary report
        report = self.output_formatter.create_summary_report(batch_output)
        report_path = self.output_formatter.output_dir / f"{batch_name}_summary.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"Exported to: {output_path}")
        self.logger.info(f"Summary report: {report_path}")
        
        return output_path
    
    def get_risk_summary(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate risk summary statistics
        
        Args:
            documents: List of documents with risk scores
            
        Returns:
            Summary statistics dictionary
        """
        risk_scores = [
            doc.get('risk_data', {}).get('overall_risk_score', 0)
            for doc in documents
        ]
        
        risk_levels = [
            doc.get('risk_data', {}).get('risk_level', 'UNKNOWN')
            for doc in documents
        ]
        
        # Count by risk level
        risk_level_counts = {}
        for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL']:
            risk_level_counts[level] = risk_levels.count(level)
        
        # Get high-risk documents
        high_risk_docs = [
            {
                'doc_id': doc.get('doc_id'),
                'risk_score': doc.get('risk_data', {}).get('overall_risk_score', 0),
                'risk_level': doc.get('risk_data', {}).get('risk_level'),
                'risk_factors': doc.get('risk_data', {}).get('risk_factors', [])
            }
            for doc in documents
            if doc.get('risk_data', {}).get('overall_risk_score', 0) >= 60
        ]
        
        return {
            'total_documents': len(documents),
            'average_risk_score': round(sum(risk_scores) / len(risk_scores), 2) if risk_scores else 0,
            'max_risk_score': max(risk_scores) if risk_scores else 0,
            'min_risk_score': min(risk_scores) if risk_scores else 0,
            'risk_level_distribution': risk_level_counts,
            'high_risk_count': len(high_risk_docs),
            'high_risk_documents': sorted(high_risk_docs, key=lambda x: x['risk_score'], reverse=True)
        }
    
    def run(
        self,
        export_output: bool = True,
        batch_name: str = "existing_data_risk_analysis",
        include_chunks: bool = False,
        limit: int = None
    ) -> Dict[str, Any]:
        """
        Main execution method
        
        Args:
            export_output: Whether to export formatted output
            batch_name: Name for batch export
            include_chunks: Whether to retrieve chunks from ChromaDB
            limit: Limit number of documents to process
            
        Returns:
            Summary statistics
        """
        self.logger.info("=" * 80)
        self.logger.info("CALCULATING RISK SCORES FROM EXISTING DATA")
        self.logger.info("=" * 80)
        
        # Step 1: Retrieve documents from Neo4j
        documents = self.get_all_documents_from_graph()
        
        if not documents:
            self.logger.error("No documents found in Neo4j")
            return {'success': False, 'error': 'No documents found'}
        
        # Apply limit if specified
        if limit:
            documents = documents[:limit]
            self.logger.info(f"Limited to {limit} documents")
        
        # Step 2: Calculate risk scores
        documents_with_risk = self.calculate_risk_scores(
            documents,
            include_chunks=include_chunks
        )
        
        # Step 3: Generate summary
        summary = self.get_risk_summary(documents_with_risk)
        
        # Step 4: Export if requested
        output_path = None
        if export_output:
            output_path = self.format_and_export(documents_with_risk, batch_name)
        
        # Display summary
        self.logger.info("\n" + "=" * 80)
        self.logger.info("RISK SCORE SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"Total documents: {summary['total_documents']}")
        self.logger.info(f"Average risk score: {summary['average_risk_score']:.2f}")
        self.logger.info(f"Max risk score: {summary['max_risk_score']:.2f}")
        self.logger.info(f"Min risk score: {summary['min_risk_score']:.2f}")
        self.logger.info("\nRisk Level Distribution:")
        for level, count in summary['risk_level_distribution'].items():
            self.logger.info(f"  {level}: {count}")
        self.logger.info(f"\nHigh-risk documents: {summary['high_risk_count']}")
        
        if summary['high_risk_documents']:
            self.logger.info("\nTop 10 High-Risk Documents:")
            for i, doc in enumerate(summary['high_risk_documents'][:10], 1):
                self.logger.info(f"  {i}. {doc['doc_id']}: {doc['risk_score']:.2f} ({doc['risk_level']})")
        
        self.logger.info("=" * 80)
        
        return {
            'success': True,
            'summary': summary,
            'output_path': str(output_path) if output_path else None,
            'documents_processed': len(documents_with_risk)
        }
    
    def close(self):
        """Close database connections"""
        # ChromaDB doesn't need explicit closing
        # Neo4j connection is managed by GraphDatabase
        self.logger.info("Connections closed")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Calculate risk scores from existing Neo4j and ChromaDB data'
    )
    parser.add_argument(
        '--export',
        action='store_true',
        help='Export formatted output for multiagent system'
    )
    parser.add_argument(
        '--batch-name',
        type=str,
        default='existing_data_risk_analysis',
        help='Name for the batch export (default: existing_data_risk_analysis)'
    )
    parser.add_argument(
        '--include-chunks',
        action='store_true',
        help='Include chunks from ChromaDB (slower but more complete)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of documents to process'
    )
    
    args = parser.parse_args()
    
    # Create necessary directories
    Config.create_directories()
    
    # Initialize calculator
    calculator = ExistingDataRiskCalculator()
    
    try:
        # Run risk calculation
        result = calculator.run(
            export_output=args.export,
            batch_name=args.batch_name,
            include_chunks=args.include_chunks,
            limit=args.limit
        )
        
        if result['success']:
            print(f"\n✓ Successfully processed {result['documents_processed']} documents")
            if result.get('output_path'):
                print(f"✓ Output exported to: {result['output_path']}")
        else:
            print(f"\n✗ Failed: {result.get('error')}")
    
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        calculator.close()


if __name__ == "__main__":
    main()
