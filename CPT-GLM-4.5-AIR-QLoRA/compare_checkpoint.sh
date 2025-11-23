#!/bin/bash

# Comparison Script Launcher for GLM-4.5-Air Checkpoints
# Easily compare base model with any checkpoint

echo "======================================================================"
echo "  GLM-4.5-Air Checkpoint Comparison Tool"
echo "======================================================================"
echo ""

# Default paths
BASE_MODEL="/workspace/Avinash/models/GLM-4.5-Air"
DEFAULT_CHECKPOINT="glm45-air-cpt-qlora/checkpoint-1000"

# Parse arguments
CHECKPOINT="${1:-$DEFAULT_CHECKPOINT}"
SAVE_FILE="${2:-comparison_results.json}"

echo "📁 Base Model:     $BASE_MODEL"
echo "📁 Checkpoint:     $CHECKPOINT"
echo "💾 Results will be saved to: $SAVE_FILE"
echo ""

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT" ]; then
    echo "❌ Error: Checkpoint directory not found: $CHECKPOINT"
    echo ""
    echo "Available checkpoints:"
    find glm45-air-cpt-qlora -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null | sort
    echo ""
    exit 1
fi

echo "✅ Checkpoint found!"
echo ""
echo "======================================================================"
echo "  Starting comparison..."
echo "======================================================================"
echo ""

# Run the comparison script
python3 compare_models.py \
    --base_model "$BASE_MODEL" \
    --checkpoint "$CHECKPOINT" \
    --prompts_file test_prompts.json \
    --max_tokens 256 \
    --save_results "$SAVE_FILE"

EXIT_CODE=$?

echo ""
echo "======================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  ✅ Comparison Complete!"
    echo "======================================================================"
    echo ""
    echo "📊 Results saved to: $SAVE_FILE"
    echo ""
    echo "To view results:"
    echo "  cat $SAVE_FILE | jq ."
else
    echo "  ❌ Comparison Failed with exit code: $EXIT_CODE"
    echo "======================================================================"
fi
echo ""
