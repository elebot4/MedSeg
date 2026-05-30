#!/bin/bash

# Medical Segmentation Training Pipeline
# Complete training setup following project guidelines

set -e  # Exit on error

echo "🔬 Medical Segmentation Training Pipeline"
echo "========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "Creating output directories..."
mkdir -p checkpoints/{2d_axi,2d_cor,2d_sag,3d_fullres}

# Check if processed data exists
if [ ! -d "data/processed/Task01_BrainTumour" ]; then
    echo "❌ Error: Processed data not found!"
    echo "Expected: data/processed/Task01_BrainTumour/"
    echo "Please run data preparation first:"
    echo "  python scripts/prepare.py --raw_dir ./data/raw/Task01_BrainTumour --save_dir ./data/processed"
    exit 1
fi

echo "✅ Found processed data"

# Training options
CONFIG=${1:-"config/Task01_BrainTumour/2d_axi.py"}
echo "Using config: $CONFIG"

# Launch training
echo "🚀 Starting training..."
echo "Config: $CONFIG"
cd src
python train.py "../$CONFIG"

echo "✅ Training completed! Check checkpoints/ directory for results."
# Medical Segmentation Training Pipeline
# Nanochat-inspired: single script to run the complete pipeline

set -e  # Exit on error

# Configuration 
DATA_DIR="./data/processed"
CONFIG=${1:-"config/2d_axi.yaml"}  # Use first arg or default
CHECKPOINT_DIR="./checkpoints"

echo "🔬 Medical Segmentation Training Pipeline"
echo "==========================================="
echo "Config: $CONFIG"
echo "Data dir: $DATA_DIR"
echo ""

# Step 1: Check if data exists, if not suggest preparation
if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A $DATA_DIR 2>/dev/null)" ]; then
    echo "❌ No processed data found in $DATA_DIR"
    echo "💡 To prepare data, run:"
    echo "   python scripts/prepare.py --raw_dir ./BraTS2021 --save_dir $DATA_DIR"
    echo ""
    exit 1
fi

# Step 2: Create checkpoint directory
mkdir -p "$CHECKPOINT_DIR"

# Step 3: Train model (nanochat style: simple command)
echo "🚀 Starting training with $CONFIG"
echo "   Output: $CHECKPOINT_DIR"
echo ""

python src/base_train.py "$CONFIG"

# Step 4: Success message
echo ""
echo "✅ Training completed!"
echo "📁 Checkpoints saved to: $CHECKPOINT_DIR"
echo ""
echo "💡 Next steps:"
echo "   - Evaluate: python src/eval.py --checkpoint $CHECKPOINT_DIR/best_model.pth"
echo "   - Different config: bash train.sh config/3d_vol.yaml" 
echo "   - Mobile export: python scripts/export_mobile.py --checkpoint $CHECKPOINT_DIR/best_model.pth"