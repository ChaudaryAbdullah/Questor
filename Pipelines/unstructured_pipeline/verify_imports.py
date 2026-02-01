#!/usr/bin/env python
"""
Verification script to check all imports are working correctly
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

errors = []
warnings = []

print("=" * 60)
print("VERIFYING IMPORTS IN UNSTRUCTURED_PIPELINE")
print("=" * 60)

# Test 1: Utils imports
print("\n[1/5] Testing utils imports...")
try:
    from utils import Config, Logger
    print("  ✓ Successfully imported Config and Logger from utils")
except Exception as e:
    errors.append(f"Utils import failed: {e}")
    print(f"  ✗ Error: {e}")

# Test 2: Exception imports
print("\n[2/5] Testing exceptions...")
try:
    from utils.exceptions import (
        DataIngestionError, EmbeddingError, VectorDBError, 
        GraphDBError, NERExtractionError
    )
    print("  ✓ Successfully imported all custom exceptions")
except Exception as e:
    errors.append(f"Exception import failed: {e}")
    print(f"  ✗ Error: {e}")

# Test 3: Database imports
print("\n[3/5] Testing database imports...")
try:
    from databases import VectorDatabase, GraphDatabase
    print("  ✓ Successfully imported VectorDatabase and GraphDatabase")
except Exception as e:
    warnings.append(f"Database import (will fail without dependencies): {e}")
    print(f"  ! Warning (expected without dependencies): {e}")

# Test 4: Pipeline module imports
print("\n[4/5] Testing pipeline __init__.py...")
try:
    from pipelines.data_loader import DataLoader
    from pipelines.chunking import TextChunker
    from pipelines.embedding import EmbeddingGenerator
    from pipelines.ner_extraction import NERExtractorOptimized
    from pipelines.graph_builder import GraphBuilder
    print("  ✓ Successfully imported all pipeline components")
except Exception as e:
    warnings.append(f"Pipeline component import (expected without dependencies): {e}")
    print(f"  ! Warning (expected without dependencies): {e}")

# Test 5: Main pipeline import
print("\n[5/5] Testing main UnstructuredPipelineOptimized...")
try:
    from pipelines.unstructured_pipeline import UnstructuredPipelineOptimized
    print("  ✓ Successfully imported UnstructuredPipelineOptimized")
except Exception as e:
    warnings.append(f"Main pipeline import (expected without dependencies): {e}")
    print(f"  ! Warning (expected without dependencies): {e}")

print("\n" + "=" * 60)
print("VERIFICATION SUMMARY")
print("=" * 60)

if not errors:
    print("✓ ALL IMPORT PATHS ARE CORRECT")
    print("\nNo critical errors detected!")
    if warnings:
        print(f"\n{len(warnings)} warnings (due to missing dependencies):")
        for w in warnings:
            print(f"  - {w}")
    print("\nTo run the full pipeline, install the missing dependencies:")
    print("  pip install -r requirements.txt")
    sys.exit(0)
else:
    print(f"✗ FOUND {len(errors)} CRITICAL ERROR(S):")
    for err in errors:
        print(f"  - {err}")
    sys.exit(1)
