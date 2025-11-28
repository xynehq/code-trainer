#!/bin/bash
# Quick start script for generating Hyperswitch CPT dataset

set -e

echo "======================================"
echo "Hyperswitch CPT Dataset - Quick Start"
echo "======================================"

# Check if hyperswitch repo exists
if [ ! -d "hyperswitch" ]; then
    echo ""
    echo "Step 1: Cloning Hyperswitch repository..."
    git clone https://github.com/juspay/hyperswitch.git
    echo "✓ Repository cloned"
else
    echo "✓ Hyperswitch repository already exists"
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo ""
    echo "Step 2: Creating Python virtual environment..."
    python3 -m venv .venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

echo ""
echo "Step 3: Activating virtual environment and installing dependencies..."
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "✓ Dependencies installed"

# Check if config.yaml exists
if [ ! -f "config.yaml" ]; then
    echo ""
    echo "⚠ config.yaml not found!"
    echo "Please copy config.template.yaml to config.yaml and add your GitHub token:"
    echo "  cp config.template.yaml config.yaml"
    echo "  # Edit config.yaml and add your GitHub API token"
    exit 1
fi

echo ""
echo "Step 4: Running dataset generator..."
python generate_dataset.py

echo ""
echo "✓ Dataset generation complete!"
echo "Check hyperswitch_cpt_dataset.jsonl for the output"
