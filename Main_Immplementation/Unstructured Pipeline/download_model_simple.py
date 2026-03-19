#!/usr/bin/env python3
"""
Download model using HuggingFace Hub's built-in download method.
This ensures compatibility with SentenceTransformer's loading mechanism.
"""

import sys
from pathlib import Path

print("=" * 60)
print("Downloading model using HuggingFace Hub...")
print("=" * 60)
print()

try:
    from sentence_transformers import SentenceTransformer
    
    print("Downloading sentence-transformers/all-MiniLM-L6-v2...")
    print("This will cache the model permanently.")
    print()
    
    # This will download and cache properly
    model = SentenceTransformer(
        'sentence-transformers/all-MiniLM-L6-v2',
        cache_folder=str(Path.home() / ".cache/torch/sentence_transformers")
    )
    
    print()
    print("=" * 60)
    print("✅ SUCCESS!")
    print("=" * 60)
    print()
    print(f"Model loaded and cached successfully!")
    print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")
    print()
    print("The model is now permanently cached and ready to use.")
    print()
    
    # Test it works
    test_embedding = model.encode("This is a test sentence.")
    print(f"✓ Test embedding generated: {len(test_embedding)} dimensions")
    print()
    
    sys.exit(0)
    
except KeyboardInterrupt:
    print("\n\nDownload cancelled by user")
    sys.exit(1)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print()
    print("This might be due to network issues.")
    print("Please ensure you have a stable internet connection and try again.")
    sys.exit(1)
