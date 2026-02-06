#!/usr/bin/env python
"""
Fixed version - works with YOUR actual Neo4j schema
Retrieves content from ChromaDB and entities from Neo4j correctly
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Starting risk calculation (fixed version)...")
print("Importing modules...")

from databases.vector_db import VectorDatabase
from databases.graph_db import GraphDatabase
from pipelines.risk_scorer import RiskScorer
from pipelines.output_formatter import OutputFormatter
from utils import Config
from tqdm import tqdm
import argparse

print("✓ Imports successful\n")


def get_document_content_from_chromadb(vector_db, doc_id):
    """Retrieve document content from ChromaDB chunks"""
    try:
        results = vector_db.collection.get(
            where={"doc_id": doc_id},
            include=["documents", "metadatas"]
        )
        
        if results and results.get('documents'):
            # Combine all chunks for this document
            content = " ".join(results['documents'])
            return content
        return ""
    except Exception as e:
        return ""


def main():
    parser = argparse.ArgumentParser(description='Calculate risk scores (FIXED)')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--batch-name', type=str, default='risk_analysis')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("CALCULATING RISK SCORES (FIXED VERSION)")
    print("=" * 80)
    print(f"Limit: {args.limit or 'All'}\n")
    
    # Connect to databases
    print("[1/5] Connecting to databases...")
    vector_db = VectorDatabase()
    graph_db = GraphDatabase()
    print("  ✓ Connected\n")
    
    # Initialize components
    print("[2/5] Initializing components...")
    risk_scorer = RiskScorer()
    if args.export:
        output_formatter = OutputFormatter()
    print("  ✓ Ready\n")
    
    # Query Neo4j for documents (using actual schema)
    print("[3/5] Querying Neo4j...")
    query = """
    MATCH (d:Document)
    RETURN d.doc_id as doc_id,
           d.label as label,
           d.company_id as company_id,
           d.date as date
    LIMIT $limit
    """
    
    limit_val = args.limit if args.limit else 10000
    results = graph_db.driver.execute_query(
        query,
        limit=limit_val,
        database_="neo4j"
    )
    
    documents = []
    for record in results.records:
        doc = {
            'doc_id': record.get('doc_id', 'unknown'),
            'label': record.get('label', 'unknown'),
            'company_id': record.get('company_id'),
            'date': record.get('date'),
            'content': '',  # Will be filled from ChromaDB
            'entities': {},
            'relationships': []
        }
        documents.append(doc)
    
    print(f"  ✓ Found {len(documents)} documents\n")
    
    if len(documents) == 0:
        print("No documents found!")
        return
    
    # Get content and entities for each document
    print("[4/5] Retrieving content and entities...")
    for doc in tqdm(documents, desc="  Processing"):
        doc_id = doc['doc_id']
        
        # Get content from ChromaDB
        content = get_document_content_from_chromadb(vector_db, doc_id)
        doc['content'] = content if content else "No content available"
        
        # Get entities from Neo4j (using correct MENTIONS relationship)
        entity_query = """
        MATCH (d:Document {doc_id: $doc_id})-[:MENTIONS]->(e)
        RETURN e.name as name, e.type as entity_type, labels(e)[0] as label
        """
        
        try:
            entity_results = graph_db.driver.execute_query(
                entity_query,
                doc_id=doc_id,
                database_="neo4j"
            )
            
            # Group entities by type
            entities_dict = {}
            for record in entity_results.records:
                entity_type = record.get('entity_type') or record.get('label', 'UNKNOWN')
                entity_name = record.get('name', '')
                
                if entity_type not in entities_dict:
                    entities_dict[entity_type] = []
                
                entities_dict[entity_type].append({
                    'text': entity_name,
                    'label': entity_type
                })
            
            doc['entities'] = entities_dict
            
        except Exception as e:
            doc['entities'] = {}
    
    print()
    
    # Calculate risk scores
    print("[5/5] Calculating risk scores...")
    documents_with_risk = []
    
    for doc in tqdm(documents, desc="  Computing"):
        try:
            risk_data = risk_scorer.calculate_document_risk(
                document=doc,
                entities=doc.get('entities'),
                relationships=doc.get('relationships')
            )
            doc['risk_data'] = risk_data
            documents_with_risk.append(doc)
        except Exception as e:
            print(f"\n  Warning: {doc['doc_id']}: {e}")
            continue
    
    print()
    
    # Display summary
    print("=" * 80)
    print("RISK SCORE SUMMARY")
    print("=" * 80)
    
    if documents_with_risk:
        risk_scores = [d['risk_data']['overall_risk_score'] for d in documents_with_risk]
        risk_levels = [d['risk_data']['risk_level'] for d in documents_with_risk]
        
        print(f"\nTotal documents: {len(documents_with_risk)}")
        print(f"Average risk score: {sum(risk_scores)/len(risk_scores):.2f}")
        print(f"Max risk score: {max(risk_scores):.2f}")
        print(f"Min risk score: {min(risk_scores):.2f}")
        
        print("\nRisk Level Distribution:")
        for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL']:
            count = risk_levels.count(level)
            print(f"  {level:12s}: {count:4d}")
        
        # High-risk documents
        high_risk = [d for d in documents_with_risk 
                    if d['risk_data']['overall_risk_score'] >= 60]
        print(f"\nHigh-risk documents (≥60): {len(high_risk)}")
        
        if high_risk:
            high_risk_sorted = sorted(high_risk, 
                                     key=lambda x: x['risk_data']['overall_risk_score'], 
                                     reverse=True)
            print("\nTop High-Risk Documents:")
            for i, doc in enumerate(high_risk_sorted[:10], 1):
                score = doc['risk_data']['overall_risk_score']
                level = doc['risk_data']['risk_level']
                print(f"  {i:2d}. {doc['doc_id']:40s} {score:6.2f} ({level})")
                # Show risk factors
                factors = doc['risk_data'].get('risk_factors', [])
                if factors:
                    print(f"      Risk factors: {factors[0]}")
    else:
        print("\nNo documents with risk scores calculated.")
    
    # Export if requested
    if args.export and output_formatter and documents_with_risk:
        print("\n" + "=" * 80)
        print("EXPORTING OUTPUT")
        print("=" * 80)
        
        try:
            from datetime import datetime
            
            formatted_outputs = []
            for doc in tqdm(documents_with_risk, desc="  Formatting"):
                formatted = output_formatter.format_for_multiagent(
                    document=doc,
                    risk_data=doc['risk_data'],
                    entities=doc.get('entities'),
                    relationships=doc.get('relationships'),
                    chunks=[]
                )
                formatted_outputs.append(formatted)
            
            # Create batch output
            batch_output = {
                'batch_metadata': {
                    'total_documents': len(formatted_outputs),
                    'timestamp': datetime.now().isoformat(),
                    'processing_pipeline': 'unstructured_pipeline',
                    'version': '1.0.0'
                },
                'summary_statistics': {
                    'total_documents': len(formatted_outputs),
                    'average_risk_score': sum(risk_scores)/len(risk_scores) if risk_scores else 0,
                    'max_risk_score': max(risk_scores) if risk_scores else 0,
                    'min_risk_score': min(risk_scores) if risk_scores else 0,
                },
                'documents': formatted_outputs,
                'high_risk_documents': [
                    {
                        'document_id': d['doc_id'],
                        'risk_score': d['risk_data']['overall_risk_score'],
                        'risk_level': d['risk_data']['risk_level'],
                        'risk_factors': d['risk_data'].get('risk_factors', [])
                    }
                    for d in high_risk_sorted[:20]
                ] if high_risk else []
            }
            
            # Save
            output_path = output_formatter.save_batch_output(batch_output, args.batch_name)
            
            # Save summary report
            report = output_formatter.create_summary_report(batch_output)
            report_path = output_formatter.output_dir / f"{args.batch_name}_summary.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            print(f"\n✓ Exported to: {output_path}")
            print(f"✓ Summary: {report_path}")
            
        except Exception as e:
            print(f"\n✗ Export failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\n✓ Successfully processed {len(documents_with_risk)} documents")
    if args.export:
        print(f"✓ Output files saved in: output/")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
