#!/bin/bash
# Setup script for Unstructured Data Pipeline on Linux
# This script sets up the environment and dependencies

set -e  # Exit on error

echo "=========================================="
echo "🚀 Setting up Fraud Detection Pipeline"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo -e "\n${YELLOW}Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.11"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}❌ Python 3.11+ required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $PYTHON_VERSION installed${NC}"

# Create virtual environment
echo -e "\n${YELLOW}Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "\n${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip
echo -e "\n${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel --quiet
echo -e "${GREEN}✓ pip upgraded${NC}"

# Install dependencies
echo -e "\n${YELLOW}Installing dependencies (this may take a few minutes)...${NC}"
pip install -r requirements.txt --quiet
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Download spaCy model
echo -e "\n${YELLOW}Downloading spaCy language model...${NC}"
python -m spacy download en_core_web_lg --quiet
echo -e "${GREEN}✓ spaCy model downloaded${NC}"

# Setup .env file
echo -e "\n${YELLOW}Setting up environment variables...${NC}"
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "${YELLOW}⚠️  Please edit .env file and add your GROQ_API_KEY${NC}"
    echo -e "${YELLOW}   You can get one from: https://console.groq.com/keys${NC}"
else
    echo -e "${GREEN}✓ .env file already exists${NC}"
fi

# Create data directories
echo -e "\n${YELLOW}Creating data directories...${NC}"
mkdir -p data/{raw,processed,vectors,graphs,reports}
echo -e "${GREEN}✓ Data directories created${NC}"

# Check for input files
echo -e "\n${YELLOW}Checking for input files...${NC}"
if [ -d "Input" ] && [ "$(ls -A Input/*.txt 2>/dev/null)" ]; then
    INPUT_COUNT=$(ls -1 Input/*.txt 2>/dev/null | wc -l)
    echo -e "${GREEN}✓ Found $INPUT_COUNT input file(s) in Input/ directory${NC}"
else
    echo -e "${YELLOW}⚠️  No .txt files found in Input/ directory${NC}"
    echo -e "${YELLOW}   Please add SEC 10-K filings to Input/ before running the pipeline${NC}"
fi

# Final instructions
echo -e "\n=========================================="
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo -e "=========================================="
echo -e "\n${YELLOW}Next steps:${NC}"
echo -e "1. Edit .env and add your GROQ_API_KEY"
echo -e "2. Place SEC 10-K filings (.txt) in the Input/ directory"
echo -e "3. Run the pipeline with: ${GREEN}./run.sh${NC}"
echo -e "\nOr activate the environment manually:"
echo -e "  ${GREEN}source venv/bin/activate${NC}"
echo -e "  ${GREEN}python run_pipeline.py${NC}"
echo -e "\n=========================================="
