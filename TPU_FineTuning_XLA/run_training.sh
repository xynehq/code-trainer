#!/bin/bash
# TPU Training Launcher Script
# Optimized for 8 x v6e TPUs

set -e

echo "========================================================================"
echo "🚀 Launching TPU Training for Qwen2.5-Coder-7B-Instruct"
echo "========================================================================"

# Activate virtual environment
source .venv/bin/activate

# Set TPU environment variables
export XLA_USE_BF16=1                          # Use bfloat16 for better performance
export XLA_TENSOR_ALLOCATOR_MAXSIZE=100000000  # Increase tensor allocator size
export PJRT_DEVICE=TPU                         # Use PJRT runtime

# Note: TPU_NUM_DEVICES is auto-detected by the training script
# Uncomment and set if you want to limit devices:
# export TPU_NUM_DEVICES=1

# PyTorch XLA settings
export XLA_IR_DEBUG=0                          # Disable IR debugging for performance
export XLA_HLO_DEBUG=0                         # Disable HLO debugging
export TF_CPP_MIN_LOG_LEVEL=0                  # Show important logs
export GRPC_VERBOSITY=ERROR                    # Reduce gRPC verbosity

# Performance optimizations
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"
export TOKENIZERS_PARALLELISM=false            # Avoid tokenizer warnings

echo "Environment Variables Set:"
echo "  - XLA_USE_BF16: $XLA_USE_BF16"
echo "  - PJRT_DEVICE: $PJRT_DEVICE"
echo "  - Devices will be auto-detected by training script"
echo "========================================================================"

# Check TPU availability
echo "Checking TPU availability..."
python3 -c "import torch_xla.runtime as xr; print(f'TPU devices found: {xr.world_size()}')" || {
    echo "❌ Error: TPUs not available. Please check your TPU setup."
    exit 1
}

echo "✓ TPUs detected successfully"
echo "========================================================================"

# Create checkpoints directory
mkdir -p checkpoints

# Run training
echo "Starting training..."
echo "========================================================================"

python3 train_tpu.py 2>&1 | tee training.log

echo "========================================================================"
echo "✅ Training completed! Check training.log for details."
echo "========================================================================"
