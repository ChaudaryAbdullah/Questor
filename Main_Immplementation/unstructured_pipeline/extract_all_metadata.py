#!/usr/bin/env python3
import sys
import os

# Add the unstructured_pipeline directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from databases.vector_db import VectorDatabase

def extract_all_metadata():
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "all_metadata.txt")
    
    try:
        vector_db = VectorDatabase()
        collection = vector_db.collection
        
        print("Fetching metadatas from ChromaDB...")
        # To avoid massive memory spike if db is huge, we can just get all metadatas
        result = collection.get(include=['metadatas'])
        
        # Deduplicate by doc_id
        unique_docs = {}
        
        for metadata in result.get('metadatas', []):
            if not metadata:
                continue
                
            doc_id = metadata.get('doc_id')
            if doc_id and doc_id not in unique_docs:
                unique_docs[doc_id] = {
                    'company_id': metadata.get('company_id', 'unknown'),
                    'label': metadata.get('label', 'unknown'),
                    'date': metadata.get('date', 'unknown')
                }
        
        print(f"Found {len(unique_docs)} unique documents. Writing to {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"{'Document_ID':<40} | {'CIK':<15} | {'Date':<12} | {'Label':<15}\n")
            f.write("-" * 90 + "\n")
            
            for doc_id, meta in sorted(unique_docs.items()):
                f.write(f"{str(doc_id):<40} | {str(meta['company_id']):<15} | {str(meta['date']):<12} | {str(meta['label']):<15}\n")
                
        print(f"Successfully extracted metadata for {len(unique_docs)} documents.")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    extract_all_metadata()
