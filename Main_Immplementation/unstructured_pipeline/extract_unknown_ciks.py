#!/usr/bin/env python3
import sys
import os
import re

# Add the unstructured_pipeline directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from databases.vector_db import VectorDatabase

def extract_unknown_ciks():
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unknown_ciks.txt")
    
    try:
        vector_db = VectorDatabase()
        collection = vector_db.collection
        
        print("Fetching all documents from ChromaDB to parse text...")
        result = collection.get(include=['metadatas', 'documents'])
        
        ciks_found = {} # doc_id -> found CIK
        
        # We will scan the text of chunks for CIK patterns.
        # The text contains 'Source File: Fraudulant Datasetsec-edgar-filings000000213510-K...'
        cik_patterns = [
            r'sec-edgar-filings(\d{10})',
            r'CENTRAL INDEX KEY:\s*0*(\d+)',
            r'CIK:\s*0*(\d+)'
        ]
        
        documents = result.get('documents', [])
        metadatas = result.get('metadatas', [])
        
        for idx, metadata in enumerate(metadatas):
            doc_id = metadata.get('doc_id', '')
            # We only care about doc_ids containing 'Unknown' where company_id is None or 'unknown'
            if 'Unknown' in doc_id:
                text = documents[idx]
                
                # If we haven't found a CIK for this doc_id yet, try to find it in this chunk
                if doc_id not in ciks_found or ciks_found[doc_id] is None:
                    found = False
                    for pattern in cik_patterns:
                        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                        if match:
                            # Usually CIKs are 10 digits padded with 0s.
                            cik_val = match.group(1).lstrip('0') or '0'
                            ciks_found[doc_id] = cik_val
                            found = True
                            break
                    
                    if not found and doc_id not in ciks_found:
                        ciks_found[doc_id] = None # init as None if no matching pattern yet
                        
        found_count = len([c for c in ciks_found.values() if c is not None])
        print(f"Scanned {len(ciks_found)} 'Unknown' documents. Found CIKs for {found_count} of them.")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"{'Document_ID':<40} | {'Extracted_CIK':<20}\n")
            f.write("-" * 65 + "\n")
            
            for doc_id, cik in sorted(ciks_found.items()):
                f.write(f"{str(doc_id):<40} | {str(cik if cik else 'NOT_FOUND'):<20}\n")
                
        print(f"Results saved to {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    extract_unknown_ciks()
