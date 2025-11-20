#!/bin/bash
# Optimal training command for 8x H200 with memory constraints

set -e

echo "🚀 Starting MoE Fine-tuning with Optimal Settings"

# Export environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS="ignore::FutureWarning"
export NCCL_TIMEOUT=7200
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0

# Run with accelerate
accelerate launch \
    --config_file accelerate_config.yaml \
    train_moe.py \
    --config config.yaml

echo "✅ Training completed!"
