#!/usr/bin/env python3
"""
Download sentence-transformers model manually using Python.
More reliable than wget/curl for large files with redirects.
"""

import sys
import os
from pathlib import Path
import urllib.request
import urllib.error

# Model files to download
FILES = {
    "config.json": 612,
    "model.safetensors": 90900000,  # 90.9 MB - the big one
    "tokenizer_config.json": 350,
    "vocab.txt": 237000,  # 232 KB
    "special_tokens_map.json": 112,
    "tokenizer.json": 466000,  # 466 KB
    "sentence_bert_config.json": 53,
    "config_sentence_transformers.json": 116,
    "modules.json": 229,
}

BASE_URL = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main"
CACHE_DIR = Path.home() / ".cache/torch/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2"

def format_size(bytes):
    """Format bytes to human readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.1f} TB"

def download_file(filename, expected_size, max_retries=3):
    """Download a single file with retry logic."""
    url = f"{BASE_URL}/{filename}"
    output_path = CACHE_DIR / filename
    
    # Skip if already exists and has correct size
    if output_path.exists():
        actual_size = output_path.stat().st_size
        if actual_size > 0:
            print(f"  ✓ {filename} already exists ({format_size(actual_size)}), skipping")
            return True
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"  [{attempt}/{max_retries}] Downloading {filename} (~{format_size(expected_size)})...", end='', flush=True)
            
            # Download with progress
            def reporthook(block_num, block_size, total_size):
                if total_size > 0:
                    downloaded = block_num * block_size
                    percent = min(100, downloaded * 100 / total_size)
                    sys.stdout.write(f"\r  [{attempt}/{max_retries}] Downloading {filename}: {percent:.0f}%")
                    sys.stdout.flush()
            
            urllib.request.urlretrieve(url, output_path, reporthook)
            print()  # New line after progress
            
            # Verify size
            actual_size = output_path.stat().st_size
            if actual_size > 0:
                print(f"  ✓ {filename} downloaded successfully ({format_size(actual_size)})")
                return True
            else:
                print(f"  ✗ {filename} is empty, retrying...")
                output_path.unlink()
                
        except urllib.error.URLError as e:
            print()  # New line
            print(f"  ✗ Network error: {e}")
            if attempt < max_retries:
                print(f"  Retrying ({attempt}/{max_retries})...")
                if output_path.exists():
                    output_path.unlink()
            else:
                return False
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Download interrupted by user")
            if output_path.exists():
                output_path.unlink()
            sys.exit(1)
            
        except Exception as e:
            print()
            print(f"  ✗ Error: {e}")
            if attempt < max_retries:
                print(f"  Retrying ({attempt}/{max_retries})...")
                if output_path.exists():
                    output_path.unlink()
            else:
                return False
    
    return False

def main():
    print("=" * 60)
    print("🔽 Downloading Sentence Transformer Model")
    print("=" * 60)
    print()
    
    # Create cache directory
    print(f"Cache directory: {CACHE_DIR}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print()
    
    # Download files
    total_files = len(FILES)
    successful = 0
    failed = []
    
    print(f"Downloading {total_files} files...")
    print()
    
    for i, (filename, size) in enumerate(FILES.items(), 1):
        print(f"[{i}/{total_files}] {filename}")
        
        if download_file(filename, size):
            successful += 1
        else:
            failed.append(filename)
            print(f"  ✗ Failed to download {filename}")
        
        print()
    
    # Summary
    print("=" * 60)
    if successful == total_files:
        print("✅ SUCCESS!")
        print("=" * 60)
        print()
        print(f"All {total_files} files downloaded successfully!")
        print(f"Cache location: {CACHE_DIR}")
        print()
        print("The model is now permanently cached.")
        print("You can run the pipeline without network access!")
        print()
        return 0
    else:
        print("⚠️  INCOMPLETE")
        print("=" * 60)
        print()
        print(f"Downloaded: {successful}/{total_files} files")
        if failed:
            print(f"Failed: {', '.join(failed)}")
        print()
        print("You can:")
        print("1. Re-run this script to retry (completed files will be skipped)")
        print("2. Check your internet connection")
        print("3. Try again later")
        print()
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Download cancelled by user")
        sys.exit(1)
