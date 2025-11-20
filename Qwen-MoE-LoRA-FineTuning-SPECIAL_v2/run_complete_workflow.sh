#!/bin/bash
# Complete MoE Fine-tuning Workflow
# Run this script to execute the entire pipeline

set -e

echo "============================================================================"
echo "🚀 Complete MoE Fine-tuning Workflow"
echo "============================================================================"
echo ""

# Step 1: Environment Setup
echo "Step 1/6: Setting up environment..."
if [[ ! -d ".venv" ]]; then
    bash setup.sh
else
    echo "✓ Environment already set up"
    source .venv/bin/activate
fi

# Step 2: Verify GPU availability
echo ""
echo "Step 2/6: Verifying GPU setup..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "✓ Found $NUM_GPUS GPUs"

# Step 3: Prepare dataset
echo ""
echo "Step 3/6: Preparing dataset from Hyperswitch..."
if [[ ! -f "data/all_data.jsonl" ]]; then
    python prepare_dataset.py
else
    echo "✓ Dataset already exists at data/all_data.jsonl"
    read -p "Regenerate dataset? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python prepare_dataset.py
    fi
fi

# Step 4: Verify configuration
echo ""
echo "Step 4/6: Configuration review..."
echo "Current config.yaml settings:"
echo "  Model: $(grep 'name:' config.yaml | head -1 | awk '{print $2}')"
echo "  Learning Rate: $(grep 'learning_rate:' config.yaml | head -1 | awk '{print $2}')"
echo "  Epochs: $(grep 'num_train_epochs:' config.yaml | awk '{print $2}')"
echo "  LoRA Rank: $(grep -A 5 '^lora:' config.yaml | grep 'r:' | awk '{print $2}')"
echo ""
read -p "Proceed with training? (Y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]; then
    echo "Aborted. Edit config.yaml and run again."
    exit 0
fi

# Step 5: Training
echo ""
echo "Step 5/6: Starting training..."
echo "This will take approximately 12-18 hours for 3 epochs"
echo "You can monitor progress with: tensorboard --logdir outputs/moe-hyperswitch-attn-lora"
echo ""
read -p "Start training now? (Y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    bash train.sh
else
    echo "Skipping training. Run 'bash train.sh' when ready."
    exit 0
fi

# Step 6: Evaluation
echo ""
echo "Step 6/6: Evaluating trained model..."
if [[ -d "outputs/moe-hyperswitch-attn-lora" ]]; then
    python evaluate_moe.py --max-samples 100
else
    echo "⚠️  No trained model found. Training may have failed."
    exit 1
fi

# Final summary
echo ""
echo "============================================================================"
echo "✅ Complete Workflow Finished!"
echo "============================================================================"
echo ""
echo "Results:"
echo "  📁 Model: outputs/moe-hyperswitch-attn-lora/"
echo "  📊 Metrics: outputs/moe-hyperswitch-attn-lora/evaluation_results.json"
echo "  📈 Logs: outputs/moe-hyperswitch-attn-lora/runs/"
echo ""
echo "Next steps:"
echo "  • Review evaluation_results.json for metrics"
echo "  • View TensorBoard: tensorboard --logdir outputs/moe-hyperswitch-attn-lora"
echo "  • Test generation: python evaluate_moe.py --max-samples 500"
echo "  • Deploy model or continue training"
echo ""
echo "============================================================================"
