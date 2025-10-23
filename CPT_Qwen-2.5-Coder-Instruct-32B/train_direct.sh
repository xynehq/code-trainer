GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' 

echo ""
echo -e "${CYAN}============================================${NC}"
echo -e "${CYAN}  Hyperswitch CPT - Direct Training${NC}"
echo -e "${CYAN}  Transformers + TRL + PEFT + DeepSpeed${NC}"
echo -e "${CYAN}============================================${NC}"
echo -e "${GREEN}Model:${NC} Qwen2.5-Coder-14B (Base)"
echo -e "${GREEN}Hardware:${NC} 2x H200 GPUs"
echo -e "${GREEN}Method:${NC} LoRA (Rank 128, Alpha 256)"
echo -e "${CYAN}============================================${NC}"
echo ""

echo -e "${YELLOW}Running pre-flight checks...${NC}"

if [ ! -f "dataset/all_data.jsonl" ]; then
    echo -e "${RED}✗ Error: dataset/all_data.jsonl not found!${NC}"
    echo -e "${RED}   Please ensure your dataset is prepared.${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Dataset found (all_data.jsonl)"

python3 -c "import transformers, peft, trl, deepspeed" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Error: Required packages not installed!${NC}"
    echo "Please run: pip install -r requirements.txt"
    exit 1
fi
echo -e "${GREEN}✓${NC} All packages installed"

echo -e "${GREEN}✓${NC} Ready to start training"

echo ""
echo -e "${CYAN}GPU Status:${NC}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
echo ""

echo -e "${CYAN}Dataset:${NC}"
TOTAL_LINES=$(wc -l < dataset/all_data.jsonl)
TRAIN_LINES=$(python3 -c "print(int($TOTAL_LINES * 0.9))")
VAL_LINES=$(python3 -c "print(int($TOTAL_LINES * 0.1))")
echo "  Total samples: $TOTAL_LINES"
echo "  Training (90%): ~$TRAIN_LINES samples"
echo "  Validation (10%): ~$VAL_LINES samples"
echo ""

echo -e "${CYAN}Training Configuration:${NC}"
TRAIN_CONFIG=$(python3 << 'PYEOF'
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

training = config.get('training', {})
dataset = config.get('dataset', {})
model = config.get('model', {})

print(f"MICRO_BATCH={training.get('micro_batch_size', 2)}")
print(f"GRAD_ACCUM={training.get('gradient_accumulation_steps', 8)}")
print(f"LR={training.get('learning_rate', 0.00002)}")
print(f"EPOCHS={training.get('num_epochs', 5)}")
print(f"SEQ_LEN={dataset.get('max_tokens', 4096)}")
print(f"LORA_R={training.get('lora', {}).get('r', 64)}")
print(f"LORA_ALPHA={training.get('lora', {}).get('alpha', 128)}")
print(f"SAMPLE_PACKING={training.get('sample_packing', False)}")
print(f"MODEL_NAME={model.get('name', 'Unknown')}")
PYEOF
)
eval "$TRAIN_CONFIG"

EFFECTIVE_BATCH=$((MICRO_BATCH * GRAD_ACCUM * 2))
SAMPLE_PACKING_STATUS="Disabled"
if [ "$SAMPLE_PACKING" = "True" ]; then
    SAMPLE_PACKING_STATUS="Enabled"
fi

echo "  Model: ${MODEL_NAME}"
echo "  Batch size: ${MICRO_BATCH} micro × ${GRAD_ACCUM} grad_accum × 2 GPUs = ${EFFECTIVE_BATCH} effective"
echo "  Learning rate: ${LR}"
echo "  Epochs: ${EPOCHS}"
echo "  LoRA: Rank ${LORA_R}, Alpha ${LORA_ALPHA}"
echo "  Context length: ${SEQ_LEN} tokens"
echo "  Sample packing: ${SAMPLE_PACKING_STATUS}"
echo ""

echo -e "${YELLOW}Press Ctrl+C in the next 5 seconds to abort...${NC}"
sleep 5
echo ""

echo -e "${CYAN}============================================${NC}"
echo -e "${GREEN}Starting Training...${NC}"
echo -e "${CYAN}============================================${NC}"
echo ""
echo -e "${BLUE}Training logs:${NC} train_log.txt"
echo -e "${BLUE}Checkpoints:${NC} outputs/hyperswitch-qwen2.5-coder-14b-lora"
echo -e "${BLUE}Monitoring:${NC} TensorBoard"
echo ""
echo -e "${YELLOW}View training metrics:${NC}"
echo "  • Console: Watch output below (loss, LR, etc.)"
echo "  • Logs: tail -f train_log.txt"
echo "  • TensorBoard: tensorboard --logdir outputs/hyperswitch-qwen2.5-coder-14b-lora/logs --port 6006"
echo "  • GPUs: watch -n 1 nvidia-smi"
echo ""
echo -e "${CYAN}============================================${NC}"
echo ""

(
    while true; do
        sleep 30
        echo ""
        echo -e "${CYAN}=== GPU Status [$(date '+%H:%M:%S')] ===${NC}"
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader | \
        awk -F', ' '{printf "  GPU %s: %s util | %s / %s VRAM | %s°C\n", $1, $2, $3, $4, $5}'
        echo ""
    done
) &
GPU_MONITOR_PID=$!

trap "kill $GPU_MONITOR_PID 2>/dev/null" EXIT

echo -e "${GREEN}Launching training with DeepSpeed...${NC}"
echo ""

deepspeed --num_gpus=2 train_direct.py 2>&1 | tee train_log.txt

TRAIN_EXIT_CODE=$?

echo ""
echo -e "${CYAN}============================================${NC}"
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}Training completed successfully!${NC}"
else
    echo -e "${RED}Training failed with exit code: $TRAIN_EXIT_CODE${NC}"
fi
echo -e "${CYAN}============================================${NC}"
echo ""

echo -e "${CYAN}Final GPU Status:${NC}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
echo ""

exit $TRAIN_EXIT_CODE