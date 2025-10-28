#!/usr/bin/env bash
set -euo pipefail

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}============================================${NC}"
echo -e "${CYAN}  Qwen-2.5 Coder - TPU Training (XLA)       ${NC}"
echo -e "${CYAN}  Transformers + TRL + PEFT on TPU          ${NC}"
echo -e "${CYAN}============================================${NC}"

if [ ! -f "dataset/all_data.jsonl" ]; then
  echo -e "${RED}✗ Error: dataset/all_data.jsonl not found!${NC}"
  exit 1
fi
echo -e "${GREEN}✓${NC} Dataset found (all_data.jsonl)"

python3 - <<'PY'
import importlib, sys
missing=[]
for m in ["torch","transformers","trl","peft","datasets","torch_xla","torch_xla.core.xla_model"]:
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)
if missing:
    print("Missing packages:", ", ".join(missing))
    sys.exit(2)
print("OK")
PY
status=$?
if [ $status -ne 0 ]; then
  echo -e "${RED}✗ Missing TPU deps. Install torch-xla matched to torch.${NC}"
  echo -e "${YELLOW}First check PyTorch version:${NC} python -c \"import torch; print(torch.__version__)\""
  echo -e "${YELLOW}Then install:${NC} pip install -r requirements-tpu.txt"
  echo -e "${YELLOW}Or manually:${NC} pip install torch==2.5.1 torchvision==0.20.1 torch-xla[tpu]==2.5.1"
  exit 1
fi
echo -e "${GREEN}✓${NC} TPU dependencies available"

echo -e "${YELLOW}Launching training (Trainer will spawn 8 TPU cores)...${NC}"
echo -e "${YELLOW}Note: tpu_num_cores=8 in TrainingArguments handles multi-core spawning${NC}"
set -x
python train_direct.py 2>&1 | tee train_log_tpu.txt
set +x

echo -e "${GREEN}✓ TPU training finished${NC}"
