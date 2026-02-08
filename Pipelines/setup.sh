#!/bin/bash
# Setup script for Unified Fraud Detection Pipeline
# Creates virtual environment and installs all dependencies

set -e  # Exit on error

echo "========================================================================"
echo "Unified Fraud Detection Pipeline - Environment Setup"
echo "========================================================================"
echo ""

# Configuration
VENV_NAME="venv"
PYTHON_VERSION="python3"

# Check if Python is available
if ! command -v $PYTHON_VERSION &> /dev/null; then
    echo "❌ Error: Python 3 is not installed or not in PATH"
    exit 1
fi

echo "✓ Found Python: $($PYTHON_VERSION --version)"
echo ""

# Step 1: Create virtual environment
echo "📦 Step 1: Creating virtual environment..."
if [ -d "$VENV_NAME" ]; then
    read -p "Virtual environment '$VENV_NAME' already exists. Recreate? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "   Removing existing venv..."
        rm -rf "$VENV_NAME"
    else
        echo "   Using existing venv..."
    fi
fi

if [ ! -d "$VENV_NAME" ]; then
    $PYTHON_VERSION -m venv "$VENV_NAME"
    echo "   ✓ Virtual environment created: $VENV_NAME"
else
    echo "   ✓ Virtual environment exists: $VENV_NAME"
fi
echo ""

# Step 2: Activate virtual environment
echo "📦 Step 2: Activating virtual environment..."
source "$VENV_NAME/bin/activate"
echo "   ✓ Virtual environment activated"
echo ""

# Step 3: Upgrade pip
echo "📦 Step 3: Upgrading pip..."
pip install --upgrade pip setuptools wheel -q
echo "   ✓ pip upgraded"
echo ""

# Step 4: Install requirements
echo "📦 Step 4: Installing requirements..."
echo "   This may take several minutes..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "   ✓ All requirements installed successfully"
else
    echo "   ❌ Error installing requirements"
    exit 1
fi
echo ""

# Step 5: Install spaCy model
echo "📦 Step 5: Installing spaCy language model..."
python -m spacy download en_core_web_sm
if [ $? -eq 0 ]; then
    echo "   ✓ spaCy model installed"
else
    echo "   ⚠ Warning: spaCy model installation failed (you may need to run manually)"
fi
echo ""

# Step 6: Verify installation
echo "📦 Step 6: Verifying installation..."

# Check critical packages
CRITICAL_PACKAGES=("numpy" "pandas" "scikit-learn" "tensorflow" "torch" "chromadb" "neo4j" "transformers" "yaml")
ALL_OK=true

for package in "${CRITICAL_PACKAGES[@]}"; do
    if python -c "import $package" 2>/dev/null; then
        echo "   ✓ $package"
    else
        echo "   ❌ $package (FAILED)"
        ALL_OK=false
    fi
done

echo ""

# Final status
echo "========================================================================"
if [ "$ALL_OK" = true ]; then
    echo "✅ SETUP COMPLETE!"
    echo "========================================================================"
    echo ""
    echo "To activate the environment in the future, run:"
    echo "   source venv/bin/activate"
    echo ""
    echo "To run the unified pipeline:"
    echo "   python unified_runner.py --pipeline both --limit 5"
    echo ""
    echo "To deactivate the environment:"
    echo "   deactivate"
else
    echo "⚠️ SETUP COMPLETED WITH WARNINGS"
    echo "========================================================================"
    echo "Some packages failed to install. Please check errors above."
    echo "You may need to install them manually."
fi
echo ""
