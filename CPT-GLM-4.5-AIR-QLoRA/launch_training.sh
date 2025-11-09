#!/bin/bash

# Launch script for GLM-4.5-Air 100B MoE model training on 4xH200 GPUs
# Uses improved training script with validation, ETA tracking, and enhanced logging
# Uses DeepSpeed ZeRO-3 for memory efficiency

# Create workspace-based temp and cache directories
mkdir -p /workspace/tmp
mkdir -p /workspace/.cache/huggingface

# Environment setup
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=3

# H200 optimizations
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_LAUNCH_BLOCKING=0

# Memory optimizations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Use workspace for temp files and HuggingFace cache (avoid /tmp crashes)
export TMPDIR=/workspace/tmp
export TEMP=/workspace/tmp
export TMP=/workspace/tmp
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets

echo "==================================================================="
echo "  GLM-4.5-Air 100B MoE Training - IMPROVED VERSION"
echo "==================================================================="
echo ""
echo "📊 Training Configuration:"
echo "  • Model: GLM-4.5-Air (100B+ parameters, MoE architecture)"
echo "  • GPUs: 4x H200 (80GB each)"
echo "  • Strategy: QLoRA + 4-bit quantization (NF4)"
echo "  • Effective batch: 64 (1 micro × 16 grad accum × 4 GPUs)"
echo "  • Learning rate: 5e-6 (optimized for CPT)"
echo "  • Warmup: 8% (59 updates, maximum stability)"
echo "  • Epochs: 3 (standard for domain adaptation)"
echo ""
echo "✨ Features:"
echo "  • Initial baseline evaluation (track improvement)"
echo "  • Validation every 1000 steps"
echo "  • Checkpointing every 1000 steps (keep top 8)"
echo "  • Best adapter auto-selection"
echo "  • Fixed scheduler (LR increases correctly)"
echo "  • Enhanced logging (loss, grad norm, LR, ETA)"
echo "  • 90/10 train/validation split"
echo ""
echo "==================================================================="
echo ""

# Check if model exists locally
MODEL_PATH="/workspace/Avinash/models/GLM-4.6"
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Model not found at $MODEL_PATH"
    echo "Please run: python download_model.py"
    exit 1
fi

# echo "✅ Model found at $MODEL_PATH"

# Check if dataset exists
DATA_PATH="/workspace/Avinash/dataset/all_data.jsonl"
if [ ! -f "$DATA_PATH" ]; then
    echo "❌ Dataset not found at $DATA_PATH"
    exit 1
fi

echo "✅ Dataset found at $DATA_PATH"

# Check GPU availability
echo ""
echo "🖥️  GPU Status:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

echo ""
echo "==================================================================="
echo "  Starting Training..."
echo "==================================================================="
echo ""

# Check if checkpoints exist for auto-resume
CHECKPOINT_DIR="glm45-air-cpt-qlora"
RESUME_FLAG=""

if [ -d "$CHECKPOINT_DIR" ]; then
    # Count checkpoint directories
    CHECKPOINT_COUNT=$(find "$CHECKPOINT_DIR" -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null | wc -l)
    
    if [ "$CHECKPOINT_COUNT" -gt 0 ]; then
        echo "🔄 Found $CHECKPOINT_COUNT existing checkpoint(s)"
        echo "   Training will automatically resume from latest checkpoint"
        echo ""
        RESUME_FLAG="--resume"
    fi
fi

# Launch with config file
accelerate launch \
    --num_machines 1 \
    --num_processes 4 \
    --machine_rank 0 \
    --main_process_ip localhost \
    --main_process_port 29500 \
    train_qlora.py \
    --config training_config.yaml \
    $RESUME_FLAG

EXIT_CODE=$?

echo ""
echo "==================================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "  ✅ Training Completed Successfully!"
    echo "==================================================================="
    echo ""
    echo "📁 Output Files:"
    echo "  • Checkpoints: glm45-air-cpt-qlora/checkpoint-*/"
    echo "  • Best adapter: best-glm45-adapter/"
    echo "  • Training logs: glm45-air-cpt-qlora/logs/training_log.jsonl"
    echo "  • Validation logs: glm45-air-cpt-qlora/logs/eval_log.jsonl"
    echo "  • Expert usage: glm45-air-cpt-qlora/logs/expert_usage_log.jsonl"
    echo ""
    echo "🎯 Next Steps:"
    echo "  1. Check validation loss in eval_log.jsonl"
    echo "  2. Best adapter is in: best-glm45-adapter/"
    echo "  3. Use best adapter for inference"
else
    echo "  ❌ Training Failed with exit code: $EXIT_CODE"
    echo "==================================================================="
    echo ""
    echo "Check the logs above for error details"
fi
echo ""
