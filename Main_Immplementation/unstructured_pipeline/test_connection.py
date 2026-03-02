#!/usr/bin/env python
"""
Quick test script to check database connections and data availability
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 80)
print("TESTING DATABASE CONNECTIONS")
print("=" * 80)

# Test 1: Import modules
print("\n[1/4] Testing imports...")
try:
    from databases.vector_db import VectorDatabase
    from databases.graph_db import GraphDatabase
    print("✓ Imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Connect to ChromaDB
print("\n[2/4] Testing ChromaDB connection...")
try:
    vector_db = VectorDatabase()
    count = vector_db.get_collection_count()
    print(f"✓ ChromaDB connected. Documents: {count}")
except Exception as e:
    print(f"✗ ChromaDB connection failed: {e}")

# Test 3: Connect to Neo4j
print("\n[3/4] Testing Neo4j connection...")
try:
    graph_db = GraphDatabase()
    print("✓ Neo4j connected")
except Exception as e:
    print(f"✗ Neo4j connection failed: {e}")
    print("  Make sure Neo4j is running and credentials in .env are correct")
    sys.exit(1)

# Test 4: Query Neo4j for documents
print("\n[4/4] Querying Neo4j for documents...")
try:
    # Simple query to count documents
    query = "MATCH (d:Document) RETURN count(d) as count"
    result = graph_db.query_graph(query)
    
    if result:
        doc_count = result[0].get('count', 0)
        print(f"✓ Found {doc_count} documents in Neo4j")
        
        if doc_count == 0:
            print("\n⚠ WARNING: No documents found in Neo4j!")
            print("  You need to run the main pipeline first to populate Neo4j")
            print("  Run: python main.py --limit 10")
        else:
            # Get sample document
            sample_query = "MATCH (d:Document) RETURN d.doc_id as doc_id LIMIT 1"
            sample = graph_db.query_graph(sample_query)
            if sample:
                print(f"  Sample doc_id: {sample[0].get('doc_id')}")
    else:
        print("✗ Query returned no results")
        
except Exception as e:
    print(f"✗ Neo4j query failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("CONNECTION TEST COMPLETE")
print("=" * 80)
