#!/usr/bin/env python3
"""
Script to check CIK numbers of processed files in ChromaDB
"""
import sys
import os
from pathlib import Path
from collections import Counter

# Add the unstructured_pipeline directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from databases.vector_db import VectorDatabase
from utils.config import ConfigOptimized as Config

def check_cik_numbers():
    """Check all CIK numbers (company_id) in ChromaDB"""
    
    print("=" * 80)
    print("CHECKING CIK NUMBERS IN CHROMADB")
    print("=" * 80)
    
    try:
        # Initialize vector database
        vector_db = VectorDatabase()
        
        # Get collection
        collection = vector_db.collection
        
        # Get total count
        total_count = collection.count()
        print(f"\nTotal documents in ChromaDB: {total_count}")
        
        if total_count == 0:
            print("\n❌ No documents found in ChromaDB!")
            print("   The database appears to be empty.")
            return
        
        # Get all documents with metadata
        print("\nFetching all documents with metadata...")
        result = collection.get(
            include=['metadatas']
        )
        
        # Extract company_ids (CIK numbers)
        company_ids = []
        doc_ids = []
        dates = []
        labels = []
        
        for metadata in result['metadatas']:
            company_id = metadata.get('company_id', 'unknown')
            doc_id = metadata.get('doc_id', 'unknown')
            date = metadata.get('date', 'unknown')
            label = metadata.get('label', 'unknown')
            
            if company_id and company_id != '' and company_id != 'unknown':
                company_ids.append(company_id)
            
            if doc_id and doc_id not in doc_ids:
                doc_ids.append(doc_id)
            
            if date and date != '' and date != 'unknown':
                dates.append(date)
            
            if label and label != 'unknown':
                labels.append(label)
        
        # Count unique values
        unique_company_ids = sorted(set(company_ids))
        unique_doc_ids = sorted(set(doc_ids))
        company_id_counts = Counter(company_ids)
        
        # Display results
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total chunks in database: {total_count}")
        print(f"Unique document IDs: {len(unique_doc_ids)}")
        print(f"Unique CIK numbers (company_id): {len(unique_company_ids)}")
        
        if labels:
            label_counts = Counter(labels)
            print(f"\nLabel distribution:")
            for label, count in label_counts.most_common():
                print(f"  - {label}: {count} chunks")
        
        # Display CIK numbers
        print("\n" + "=" * 80)
        print("CIK NUMBERS (Company IDs)")
        print("=" * 80)
        
        if unique_company_ids:
            print(f"\nFound {len(unique_company_ids)} unique CIK numbers:")
            for cik in unique_company_ids:
                count = company_id_counts[cik]
                print(f"  • CIK {cik}: {count} chunks")
        else:
            print("\n❌ No CIK numbers found in metadata!")
            print("   Documents may not have company_id metadata.")
        
        # Display sample document IDs
        print("\n" + "=" * 80)
        print("DOCUMENT IDs (Sample)")
        print("=" * 80)
        print(f"\nShowing first 20 document IDs:")
        for doc_id in unique_doc_ids[:20]:
            print(f"  • {doc_id}")
        
        if len(unique_doc_ids) > 20:
            print(f"\n  ... and {len(unique_doc_ids) - 20} more documents")
        
        # Display sample metadata
        print("\n" + "=" * 80)
        print("SAMPLE METADATA (First 3 documents)")
        print("=" * 80)
        
        for i, metadata in enumerate(result['metadatas'][:3]):
            print(f"\nDocument {i+1}:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
        
        print("\n" + "=" * 80)
        print("DONE")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_cik_numbers()
