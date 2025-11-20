#!/bin/bash
# ============================================================================
# Setup Script for MoE Fine-tuning Environment
# Installs all required dependencies
# ============================================================================

set -e

echo "============================================================================"
echo "📦 MoE Fine-tuning Environment Setup"
echo "============================================================================"
echo ""

# Check if virtual environment exists
if [[ ! -d ".venv" ]]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
echo ""
echo "Installing PyTorch with CUDA 12.1 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Transformers and related libraries
echo ""
echo "Installing Transformers ecosystem..."
pip install transformers>=4.36.0
pip install accelerate>=0.25.0
pip install peft>=0.7.0
pip install bitsandbytes>=0.41.0

# Install dataset libraries
echo ""
echo "Installing dataset libraries..."
pip install datasets>=2.15.0

# Install evaluation libraries
echo ""
echo "Installing evaluation libraries..."
pip install evaluate
pip install scikit-learn

# Install Flash Attention 2 (for H200 GPUs)
echo ""
echo "Installing Flash Attention 2..."
pip install flash-attn --no-build-isolation

# Install utilities
echo ""
echo "Installing utilities..."
pip install pyyaml
pip install tensorboard
pip install tqdm
pip install numpy
pip install scipy

# Install optional: wandb for experiment tracking
echo ""
read -p "Install Weights & Biases for experiment tracking? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install wandb
    echo "✓ W&B installed"
fi

# Install optional: deepspeed
echo ""
read -p "Install DeepSpeed for advanced optimization? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install deepspeed
    echo "✓ DeepSpeed installed"
fi

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p data
mkdir -p outputs
mkdir -p logs

# Clone hyperswitch if not exists
if [[ ! -d "hyperswitch" ]]; then
    echo ""
    echo "Cloning Hyperswitch repository..."
    git clone --depth 1 https://github.com/juspay/hyperswitch.git
    echo "✓ Hyperswitch cloned"
fi

# Test GPU availability
echo ""
echo "============================================================================"
echo "🎮 GPU Information:"
echo "============================================================================"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

# Test imports
echo ""
echo "============================================================================"
echo "🧪 Testing installations..."
echo "============================================================================"
python3 << EOF
import torch
import transformers
import peft
import datasets
import accelerate

print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
print(f"✓ CUDA version: {torch.version.cuda}")
print(f"✓ GPUs available: {torch.cuda.device_count()}")
print(f"✓ Transformers version: {transformers.__version__}")
print(f"✓ PEFT version: {peft.__version__}")
print(f"✓ Datasets version: {datasets.__version__}")
print(f"✓ Accelerate version: {accelerate.__version__}")

# Test flash attention
try:
    import flash_attn
    print(f"✓ Flash Attention 2: Available")
except ImportError:
    print("⚠️  Flash Attention 2: Not available (optional)")
EOF

echo ""
echo "============================================================================"
echo "✅ Setup Complete!"
echo "============================================================================"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source .venv/bin/activate"
echo "  2. Prepare dataset: python prepare_dataset.py"
echo "  3. Start training: bash train.sh"
echo ""
