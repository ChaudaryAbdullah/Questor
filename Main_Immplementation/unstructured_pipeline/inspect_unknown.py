#!/usr/bin/env python3
import sys
import os

# Add the unstructured_pipeline directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from databases.vector_db import VectorDatabase

def inspect():
    db = VectorDatabase()
    res = db.collection.get(
        where={'doc_id': 'Unknown_full-submission_1'},
        include=['documents']
    )
    if res['documents']:
        text = res['documents'][0]
        with open('unknown_text_sample.txt', 'w', encoding='utf-8') as f:
            f.write(text[:3000])
        print("Wrote first 3000 chars to unknown_text_sample.txt")
    else:
        print("Document not found.")

if __name__ == "__main__":
    inspect()
