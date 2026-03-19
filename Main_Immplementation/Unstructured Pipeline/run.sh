#!/bin/bash
# Run script for Unstructured Data Pipeline on Linux
# This script activates the virtual environment and runs the pipeline

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=========================================="
echo "🚀 Fraud Detection Pipeline Runner"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}❌ Virtual environment not found!${NC}"
    echo -e "${YELLOW}Please run setup_linux.sh first:${NC}"
    echo -e "  ${GREEN}chmod +x setup_linux.sh${NC}"
    echo -e "  ${GREEN}./setup_linux.sh${NC}"
    exit 1
fi

# Activate virtual environment
echo -e "\n${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Check for .env file
if [ ! -f ".env" ]; then
    echo -e "${RED}❌ .env file not found!${NC}"
    echo -e "${YELLOW}Please copy .env.example to .env and configure it${NC}"
    exit 1
fi

# Check for GROQ_API_KEY
source .env
if [ -z "$GROQ_API_KEY" ] || [ "$GROQ_API_KEY" = "your_groq_api_key_here" ]; then
    echo -e "${RED}❌ GROQ_API_KEY not configured in .env file!${NC}"
    echo -e "${YELLOW}Please edit .env and add your Groq API key${NC}"
    echo -e "${YELLOW}Get one from: https://console.groq.com/keys${NC}"
    exit 1
fi

# Check for input files
if [ ! -d "Input" ] || [ -z "$(ls -A Input/*.txt 2>/dev/null)" ]; then
    echo -e "${RED}❌ No .txt files found in Input/ directory!${NC}"
    echo -e "${YELLOW}Please add SEC 10-K filings to Input/ directory${NC}"
    exit 1
fi

# Display configuration
echo -e "\n${GREEN}✓ Configuration verified${NC}"
echo -e "\n${YELLOW}Running pipeline...${NC}"
echo -e "${YELLOW}Expected runtime: 6-8 minutes for a 200-page 10-K${NC}\n"

# Run the pipeline
python run_pipeline.py

# Check exit status
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ Pipeline completed successfully!${NC}"
    echo -e "${YELLOW}Check data/reports/ for results${NC}"
else
    echo -e "\n${RED}❌ Pipeline failed!${NC}"
    echo -e "${YELLOW}Check logs for details${NC}"
    exit 1
fi
