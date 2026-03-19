#!/bin/bash
# Manual download script for sentence-transformers/all-MiniLM-L6-v2
# Downloads model files individually to avoid network timeout issues

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=========================================="
echo "🔽 Downloading Embedding Model"
echo "=========================================="
echo ""

# Create cache directory
CACHE_DIR="$HOME/.cache/torch/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2"
echo -e "${YELLOW}Creating cache directory...${NC}"
mkdir -p "$CACHE_DIR"
cd "$CACHE_DIR"

echo -e "${GREEN}✓ Cache directory: $CACHE_DIR${NC}"
echo ""

# Base URL
BASE_URL="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main"

# List of required files with sizes (approximate)
declare -A FILES=(
    ["config.json"]="612 bytes"
    ["model.safetensors"]="90.9 MB"
    ["tokenizer_config.json"]="350 bytes"
    ["vocab.txt"]="232 KB"
    ["special_tokens_map.json"]="112 bytes"
    ["tokenizer.json"]="466 KB"
    ["sentence_bert_config.json"]="53 bytes"
    ["config_sentence_transformers.json"]="116 bytes"
    ["modules.json"]="229 bytes"
)

TOTAL_FILES=${#FILES[@]}
CURRENT=0

echo -e "${YELLOW}Downloading $TOTAL_FILES files...${NC}"
echo ""

# Function to download with retry
download_file() {
    local filename=$1
    local size=$2
    local url="$BASE_URL/$filename"
    local max_retries=3
    
    for i in $(seq 1 $max_retries); do
        echo -e "${YELLOW}[$CURRENT/$TOTAL_FILES] Downloading $filename ($size)...${NC}"
        
        if wget -q --show-progress "$url" -O "$filename" 2>&1; then
            echo -e "${GREEN}✓ Downloaded $filename${NC}"
            return 0
        else
            if [ $i -lt $max_retries ]; then
                echo -e "${RED}✗ Failed, retrying ($i/$max_retries)...${NC}"
                sleep 2
            else
                echo -e "${RED}✗ Failed to download $filename after $max_retries attempts${NC}"
                return 1
            fi
        fi
    done
}

# Download each file
for file in "${!FILES[@]}"; do
    ((CURRENT++))
    
    # Skip if file already exists and is not empty
    if [ -f "$file" ] && [ -s "$file" ]; then
        echo -e "${GREEN}✓ [$CURRENT/$TOTAL_FILES] $file already exists, skipping${NC}"
        continue
    fi
    
    if ! download_file "$file" "${FILES[$file]}"; then
        echo ""
        echo -e "${RED}=========================================="
        echo -e "❌ Download failed"
        echo -e "==========================================${NC}"
        echo ""
        echo "Some files may have been downloaded. You can:"
        echo "1. Re-run this script to resume (already downloaded files will be skipped)"
        echo "2. Check your internet connection and try again"
        echo "3. Download manually from: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/main"
        exit 1
    fi
    
    echo ""
done

# Verify all files exist
echo -e "${YELLOW}Verifying downloads...${NC}"
ALL_GOOD=true

for file in "${!FILES[@]}"; do
    if [ ! -f "$file" ] || [ ! -s "$file" ]; then
        echo -e "${RED}✗ Missing or empty: $file${NC}"
        ALL_GOOD=false
    fi
done

if [ "$ALL_GOOD" = true ]; then
    echo ""
    echo "=========================================="
    echo -e "${GREEN}✅ SUCCESS!${NC}"
    echo "=========================================="
    echo ""
    echo "Model downloaded successfully to:"
    echo "$CACHE_DIR"
    echo ""
    echo "Total size: ~91 MB"
    echo ""
    echo "The model is now cached permanently."
    echo "You can run the pipeline without re-downloading!"
    echo ""
    exit 0
else
    echo ""
    echo -e "${RED}=========================================="
    echo -e "⚠️  Some files are missing"
    echo -e "==========================================${NC}"
    echo ""
    echo "Re-run this script to retry downloading missing files."
    exit 1
fi
