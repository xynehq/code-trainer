#!/bin/bash
# ============================================================================
# MoE Fine-tuning Training Launcher
# Distributed training across 4x H200 GPUs
# ============================================================================

set -e  # Exit on error

echo "============================================================================"
echo "🚀 MoE Fine-tuning Training Launcher"
echo "============================================================================"

# Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export TOKENIZERS_PARALLELISM=false

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/config.yaml"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_moe.py"

# Check if virtual environment is activated
if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo "⚠️  Virtual environment not activated. Activating..."
    source "${SCRIPT_DIR}/.venv/bin/activate"
fi

# Check if config exists
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "❌ Config file not found: $CONFIG_PATH"
    exit 1
fi

# Check if training script exists
if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "❌ Training script not found: $TRAIN_SCRIPT"
    exit 1
fi

# Check GPU availability
echo ""
echo "📊 GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
echo ""

# Number of GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "🎮 Detected $NUM_GPUS GPUs"

# Check if dataset exists
DATA_FILE="${SCRIPT_DIR}/data/all_data.jsonl"
if [[ ! -f "$DATA_FILE" ]]; then
    echo "❌ Dataset not found: $DATA_FILE"
    echo "📝 Please run prepare_dataset.py first:"
    echo "   python prepare_dataset.py"
    exit 1
fi

echo ""
echo "✓ Configuration: $CONFIG_PATH"
echo "✓ Training script: $TRAIN_SCRIPT"
echo "✓ Dataset: $DATA_FILE"
echo ""

# Launch training with accelerate or torchrun
echo "============================================================================"
echo "🚂 Launching distributed training on $NUM_GPUS GPUs..."
echo "============================================================================"
echo ""

# Option 1: Using accelerate (recommended)
if command -v accelerate &> /dev/null; then
    echo "Using Accelerate for distributed training..."
    accelerate launch \
        --num_processes=$NUM_GPUS \
        --multi_gpu \
        --mixed_precision=bf16 \
        "$TRAIN_SCRIPT" \
        --config "$CONFIG_PATH"

# Option 2: Using torchrun (fallback)
elif command -v torchrun &> /dev/null; then
    echo "Using torchrun for distributed training..."
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=29500 \
        "$TRAIN_SCRIPT" \
        --config "$CONFIG_PATH"

# Option 3: Single GPU fallback
else
    echo "⚠️  No distributed launcher found. Running on single GPU..."
    python "$TRAIN_SCRIPT" --config "$CONFIG_PATH"
fi

# Check exit status
if [[ $? -eq 0 ]]; then
    echo ""
    echo "============================================================================"
    echo "✅ Training completed successfully!"
    echo "============================================================================"
    echo ""
    echo "📁 Output directory: outputs/moe-hyperswitch-attn-lora"
    echo "📊 Tensorboard logs: outputs/moe-hyperswitch-attn-lora/runs"
    echo ""
    echo "To view training logs:"
    echo "  tensorboard --logdir outputs/moe-hyperswitch-attn-lora"
    echo ""
else
    echo ""
    echo "============================================================================"
    echo "❌ Training failed!"
    echo "============================================================================"
    exit 1
fi
