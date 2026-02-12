#!/usr/bin/env python3
"""
Quick test to verify ChromaDB data retrieval by CIK
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unstructured_pipeline.databases.vector_db import VectorDatabase
from unstructured_pipeline.utils.cik_extractor import CIKExtractor

def test_chromadb_by_cik():
    """Test retrieving data from ChromaDB by CIK"""
    
    # Extract CIK from input file
    extractor = CIKExtractor()
    input_file = Path("Input/0001040719.json")
    
    cik = extractor.extract_cik_from_file(input_file)
    print(f"Extracted CIK: {cik}")
    
    # Query ChromaDB
    vector_db = VectorDatabase()
    
    # Get documents for this CIK  
    results = vector_db.collection.get(
        where={"company_id": cik},
        include=["metadatas", "documents"],
        limit=5
    )
    
    print(f"\nChromaDB Results for CIK {cik}:")
    print(f"  Total chunks found: {len(results['ids']) if results.get('ids') else 0}")
    
    if results.get('ids'):
        # Get unique doc_ids
        doc_ids = set()
        for metadata in results.get('metadatas', []):
            doc_id = metadata.get('doc_id')
            if doc_id:
                doc_ids.add(doc_id)
        
        print(f"  Unique documents: {len(doc_ids)}")
        print(f"  Document IDs: {list(doc_ids)[:3]}...")
        
        # Show sample metadata
        print(f"\n  Sample metadata:")
        for i, metadata in enumerate(results.get('metadatas', [])[:2]):
            print(f"    Chunk {i+1}: {metadata}")
    else:
        print("  No data found!")

if __name__ == "__main__":
    test_chromadb_by_cik()
