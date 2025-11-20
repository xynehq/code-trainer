#!/bin/bash
# Optimal training command for 4x H200 with memory constraints

set -e

echo "🚀 Starting MoE Fine-tuning with Optimal Settings"

# Export environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Run with accelerate
accelerate launch \
    --config_file accelerate_config.yaml \
    train_moe.py \
    --config config.yaml

echo "✅ Training completed!"
