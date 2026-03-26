#!/usr/bin/env python3
import sys
import os

# Add the unstructured_pipeline directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from databases.vector_db import VectorDatabase

def extract_cik_status():
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CIK_fraud_status.txt")
    
    try:
        vector_db = VectorDatabase()
        collection = vector_db.collection
        
        cik_status = {}
        offset = 0
        limit = 500
        
        while True:
            result = collection.get(include=['metadatas'], limit=limit, offset=offset)
            metadatas = result.get('metadatas', [])
            
            if not metadatas:
                break
                
            for metadata in metadatas:
                # In ChromaDB, metadata might be None for some entries
                if not metadata:
                    continue
                    
                cik = metadata.get('company_id')
                label = metadata.get('label')
                
                if cik and str(cik).lower() != 'unknown':
                    # If label is present, record it
                    if str(label) == "1" or "fraud" in str(label).lower() and "non" not in str(label).lower():
                        parsed_label = "Fraudulent"
                    elif str(label) == "0" or "non" in str(label).lower() or str(label).lower() == "normal":
                        parsed_label = "Non-Fraudulent"
                    else:
                        parsed_label = str(label)
                    
                    if cik not in cik_status or cik_status.get(cik) == "unknown":
                        cik_status[cik] = parsed_label
            
            offset += limit
        
        with open(output_file, 'w') as f:
            f.write("CIK\tStatus\n")
            f.write("-" * 30 + "\n")
            
            for cik, status in sorted(cik_status.items()):
                f.write(f"{cik}\t{status}\n")
                
        print(f"Successfully extracted {len(cik_status)} CIKs and their status to {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    extract_cik_status()
