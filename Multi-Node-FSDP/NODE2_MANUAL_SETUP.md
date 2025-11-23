# 🛠️ MANUAL NODE 2 SETUP GUIDE - Step by Step

**You are currently on Node 2 (innmi1srh2-p021)**

This guide walks you through setting up Node 2 completely from scratch while you're logged into it.

---

## Current Status Check

You should be seeing:
```bash
(.venv) root@innmi1srh2-p021:/workspace/distTest#
```

This means:
- ✅ You're on Node 2 (innmi1srh2-p021)
- ✅ You have a virtual environment (.venv) activated
- ✅ You're in /workspace/distTest directory

---

## Step 1: Verify Model and Dataset Paths

First, let's check if the model and dataset are accessible from this node.

### Check Model (212GB GLM-4.5-Air)

```bash
# Correct command (you typo'd 's' instead of 'ls')
ls /workspace/Avinash/models/GLM-4.5-Air
```

**Expected Output:**
```
config.json
configuration_glm.py
generation_config.json
modeling_glm.py
pytorch_model-00001-of-00078.bin
pytorch_model-00002-of-00078.bin
...
tokenization_chatglm.py
tokenizer_config.json
```

**If you see "No such file or directory":**
- The path doesn't exist on Node 2
- You'll need to either:
  - Option A: Copy from Node 1 (takes time, 212GB)
  - Option B: Check if there's a shared mount

### Check Dataset

```bash
ls -lh /workspace/Avinash/dataset/all_data.jsonl
```

**Expected Output:**
```
-rw-r--r-- 1 root root XXXM Nov 22 XX:XX /workspace/Avinash/dataset/all_data.jsonl
```

**If file doesn't exist:**
- Need to copy from Node 1

### Quick Test Read

```bash
# Test if you can read the model config
cat /workspace/Avinash/models/GLM-4.5-Air/config.json | head -20

# Test if you can read the dataset (first line)
head -n 1 /workspace/Avinash/dataset/all_data.jsonl
```

---

## Step 2: Verify distTest Code Exists

```bash
# Check you're in the right directory
pwd
# Expected: /workspace/distTest

# List files in distTest
ls -lh
```

**Expected Files:**
```
COMPLETE_MULTINODE_GUIDE.md
config.yaml
launch_node0.sh
launch_node1.sh
requirements.txt
train_fsdp.py
README.md
...
```

**If train_fsdp.py is missing:**
```bash
# The code wasn't copied to Node 2 yet
# You need to copy it - see "Manual Code Copy" section below
```

---

## Step 3: Install Python Dependencies

Your virtual environment is already activated (.venv shows in prompt).

### Option A: Install from requirements.txt (Recommended)

```bash
# Make sure you're in /workspace/distTest
cd /workspace/distTest

# Install all dependencies
pip install -r requirements.txt

# This will install:
# - torch>=2.0.0
# - transformers>=4.30.0
# - accelerate>=0.20.0
# - peft>=0.7.0
# - datasets>=2.14.0
# - pyyaml>=6.0
# - tqdm>=4.65.0
```

**Expected Output:**
```
Collecting torch>=2.0.0
...
Successfully installed torch-2.9.1 transformers-4.57.1 ...
```

### Option B: Install Individually

If requirements.txt is missing or you want to install one by one:

```bash
pip install torch>=2.0.0
pip install transformers>=4.30.0
pip install accelerate>=0.20.0
pip install peft>=0.7.0
pip install datasets>=2.14.0
pip install pyyaml>=6.0
pip install tqdm>=4.65.0
```

### Verify Installation

```bash
# Check all packages are installed
pip list | grep -E "torch|transformers|peft|datasets|accelerate|yaml|tqdm"
```

**Expected Output:**
```
accelerate               1.12.0
datasets                 4.4.1
peft                     0.18.0
PyYAML                   6.0.2
torch                    2.9.1
tqdm                     4.67.1
transformers             4.57.1
```

---

## Step 4: Verify CUDA and NCCL

### Check CUDA

```bash
# Verify PyTorch can see GPUs
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

**Expected Output:**
```
CUDA: True, GPUs: 8
```

**If "CUDA: False":**
- PyTorch wasn't installed with CUDA support
- Reinstall: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

### Check NCCL

```bash
python3 -c "import torch.distributed as dist; print('NCCL:', dist.is_nccl_available())"
```

**Expected Output:**
```
NCCL: True
```

### Check GPU Status

```bash
nvidia-smi
```

**Expected:**
- Shows 8 GPUs (H200 or similar)
- All GPUs should be idle (0% utilization) right now

---

## Step 5: Verify Network Interface

NCCL needs the correct network interface for inter-node communication.

### Find Network Interface

```bash
# List all network interfaces
ifconfig
```

**Look for the interface connected to your cluster network.**

Example output:
```
enp26s0np0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 10.11.7.XX  netmask 255.255.252.0  broadcast 10.11.7.255
        ...
```

**The interface name to use:** `enp26s0np0`

### Verify Interface in Launch Script

```bash
# Check what interface is set in launch_node1.sh
grep NCCL_SOCKET_IFNAME launch_node1.sh
```

**Expected:**
```
export NCCL_SOCKET_IFNAME=enp26s0np0
```

**If your interface is different:**
```bash
# Edit the launch script
nano launch_node1.sh

# Change this line to match your interface:
export NCCL_SOCKET_IFNAME=<your_interface_name>

# Save and exit (Ctrl+X, Y, Enter)
```

---

## Step 6: Test Configuration Loading

Verify the config.yaml can be read and has correct paths:

```bash
# Check config paths
python3 -c "import yaml; c = yaml.safe_load(open('config.yaml')); print('Model:', c['model']['name_or_path']); print('Dataset:', c['data']['dataset_path'])"
```

**Expected Output:**
```
Model: /workspace/Avinash/models/GLM-4.5-Air
Dataset: /workspace/Avinash/dataset/all_data.jsonl
```

**If you see different paths:**
- The config.yaml might be different from Node 1
- They MUST match exactly on both nodes!

---

## Step 7: Test Import Training Script

Make sure the training script can be imported without errors:

```bash
# Test imports
python3 -c "import train_fsdp; print('✓ Script imports successfully')"
```

**Expected Output:**
```
✓ Script imports successfully
```

**If you see errors:**
- Missing dependencies - check Step 3
- Syntax errors - the script might be corrupted during copy

---

## Step 8: Pre-Launch Checklist

Before launching, verify everything:

```bash
# Run this comprehensive check script
cat << 'EOF' > check_node2.sh
#!/bin/bash
echo "=== Node 2 Pre-Launch Checklist ==="
echo ""
echo "1. Hostname:"
hostname
echo ""
echo "2. GPUs Available:"
python3 -c "import torch; print(f'   CUDA: {torch.cuda.is_available()}, Count: {torch.cuda.device_count()}')"
echo ""
echo "3. NCCL Available:"
python3 -c "import torch.distributed as dist; print(f'   NCCL: {dist.is_nccl_available()}')"
echo ""
echo "4. Model Exists:"
if [ -d "/workspace/Avinash/models/GLM-4.5-Air" ]; then
    echo "   ✓ Model directory found"
else
    echo "   ✗ Model NOT found"
fi
echo ""
echo "5. Dataset Exists:"
if [ -f "/workspace/Avinash/dataset/all_data.jsonl" ]; then
    echo "   ✓ Dataset file found"
else
    echo "   ✗ Dataset NOT found"
fi
echo ""
echo "6. Training Script:"
if [ -f "train_fsdp.py" ]; then
    echo "   ✓ train_fsdp.py found"
else
    echo "   ✗ train_fsdp.py NOT found"
fi
echo ""
echo "7. Launch Script:"
if [ -f "launch_node1.sh" ] && [ -x "launch_node1.sh" ]; then
    echo "   ✓ launch_node1.sh found and executable"
else
    echo "   ✗ launch_node1.sh NOT found or not executable"
fi
echo ""
echo "8. Network Interface:"
IFACE=$(grep NCCL_SOCKET_IFNAME launch_node1.sh | cut -d'=' -f2)
if ifconfig $IFACE &> /dev/null; then
    echo "   ✓ Interface $IFACE exists"
else
    echo "   ✗ Interface $IFACE NOT found - UPDATE launch_node1.sh!"
fi
echo ""
echo "9. Python Packages:"
python3 -c "import torch, transformers, peft, datasets; print('   ✓ All packages imported')" 2>&1
echo ""
echo "=================================="
echo "If all items show ✓, you're ready to launch!"
echo "=================================="
EOF

chmod +x check_node2.sh
./check_node2.sh
```

**Review the output:**
- All items should show ✓ (checkmark)
- If any show ✗, fix them before proceeding

---

## Step 9: Ready to Launch!

Once all checks pass, **WAIT for Node 1 (master) to start first**.

### Coordination with Node 1

**Timing is critical:**

1. **Node 1 (master)** MUST launch first
2. **Node 2 (worker)** launches within 30 seconds after

### When Node 1 Starts...

You'll need to be ready with this command:

```bash
cd /workspace/distTest
./launch_node1.sh 2>&1 | tee multinode_training.log
```

**DO NOT run this yet!** Wait for confirmation that Node 1 has started.

---

## Expected Startup Sequence

### When You Launch Node 2:

```bash
./launch_node1.sh 2>&1 | tee multinode_training.log
```

**Phase 1: NCCL Initialization (first 10-30 seconds)**
```
=========================================
Launching WORKER NODE (Node 1)
Master: 10.11.7.50:29600
Total Nodes: 2
GPUs per Node: 8
=========================================

innmi1srh2-p021:XXXXX:XXXXX [0] NCCL INFO Bootstrap : Using enp26s0np0:10.11.7.XX<0>
innmi1srh2-p021:XXXXX:XXXXX [0] NCCL INFO NET/IB : Using interface enp26s0np0
innmi1srh2-p021:XXXXX:XXXXX [0] NCCL INFO Connected to master 10.11.7.50:29600
...
```

**Phase 2: Connection Established (~30 seconds)**
```
================================================================================
Training Configuration:
  Hostname: innmi1srh2-p021
  World Size: 16
  Number of Nodes: 2
  GPUs per Node: 8
  Master: 10.11.7.50:29600
================================================================================
```

**Phase 3: Model Loading (1-3 minutes)**
```
Loading model: /workspace/Avinash/models/GLM-4.5-Air
...
✓ Gradient checkpointing enabled
trainable params: 126,615,552 || all params: 106,978,861,056 || trainable%: 0.1184
✓ All parameters converted to torch.bfloat16
✓ Model wrapped with FSDP
```

**Phase 4: Dataset Loading (2-5 minutes)**
```
Loading dataset from: /workspace/Avinash/dataset/all_data.jsonl
[Tokenizing progress bars will show]
Train samples: 16011
Eval samples: 1780
```

**Phase 5: Training Begins!**
```
Starting training...
Running baseline evaluation...
[Node 2 will be SILENT during training - only Node 1 prints metrics]
```

**⚠️ IMPORTANT:** 
- Node 2 will NOT print training step logs
- Only NCCL debug info and initial setup
- This is NORMAL behavior
- Check `nvidia-smi` to confirm GPUs are working

---

## Monitoring on Node 2

### Check GPU Usage

```bash
# In a separate terminal/screen on Node 2
watch -n 1 nvidia-smi
```

**Expected:**
- All 8 GPUs at 95-100% utilization
- Memory usage near full (138/140 GB per GPU)

### Check Training Process

```bash
# Check if Python processes are running
ps aux | grep train_fsdp.py
```

**Expected:**
- 8-9 Python processes (1 per GPU + 1 master)

### View Worker Logs

```bash
# Watch the log file
tail -f multinode_training.log

# Search for errors
grep -i "error\|failed\|exception" multinode_training.log
```

**Node 2 logs will be mostly:**
- NCCL communication messages
- Initial setup output
- Very little training output (that's on Node 1)

---

## Troubleshooting

### Issue: Model/Dataset Not Found

**Error:**
```
FileNotFoundError: /workspace/Avinash/models/GLM-4.5-Air
```

**Fix:**
```bash
# Check if it's a symbolic link issue
ls -la /workspace/Avinash/models/

# Try absolute path
ls -la $(realpath /workspace/Avinash/models/GLM-4.5-Air)

# If truly missing, you need to copy from Node 1
# See "Manual Copy" section below
```

### Issue: NCCL Connection Timeout

**Error:**
```
RuntimeError: Timed out initializing process group
```

**Fix:**
```bash
# 1. Check Node 1 is running
# 2. Verify network connectivity
ping -c 3 10.11.7.50

# 3. Check firewall
sudo firewall-cmd --list-all

# 4. Verify interface
ifconfig enp26s0np0

# 5. Kill and restart
pkill -9 -f train_fsdp.py
# Wait for Node 1 to restart, then try again
```

### Issue: Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'peft'
```

**Fix:**
```bash
# Activate virtual environment
source .venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt

# Verify
pip list | grep peft
```

---

## Manual Code Copy (If Needed)

If training code is missing on Node 2:

### From Node 1

```bash
# On Node 1 (innmi1srh2-p040)
cd /workspace
scp -r distTest root@innmi1srh2-p021:/workspace/
```

### Or Copy Individual Files

If you only need specific files:

```bash
# On Node 1
cd /workspace/distTest
scp train_fsdp.py config.yaml launch_node1.sh requirements.txt root@innmi1srh2-p021:/workspace/distTest/
```

---

## Manual Model/Dataset Copy (If Needed)

**⚠️ WARNING:** This copies 212GB - will take 30-60 minutes!

### Copy Model

```bash
# On Node 1
cd /workspace/Avinash
rsync -avz --progress models/GLM-4.5-Air root@innmi1srh2-p021:/workspace/Avinash/models/
```

### Copy Dataset

```bash
# On Node 1
rsync -avz --progress dataset/all_data.jsonl root@innmi1srh2-p021:/workspace/Avinash/dataset/
```

---

## Final Checklist Before Launch

Run this on Node 2:

```bash
# ✅ Virtual environment activated
echo $VIRTUAL_ENV  # Should show .venv path

# ✅ In correct directory
pwd  # Should be /workspace/distTest

# ✅ All dependencies installed
pip list | grep -E "torch|transformers|peft|datasets"

# ✅ CUDA available
python3 -c "import torch; print(torch.cuda.device_count())"  # Should be 8

# ✅ Model accessible
ls /workspace/Avinash/models/GLM-4.5-Air/config.json

# ✅ Dataset accessible
wc -l /workspace/Avinash/dataset/all_data.jsonl  # Should show ~17791

# ✅ Launch script executable
ls -la launch_node1.sh  # Should have 'x' permission

# ✅ Ready to launch!
```

---

## Launch Command (When Node 1 is Ready)

```bash
cd /workspace/distTest
./launch_node1.sh 2>&1 | tee multinode_training.log
```

**After launching:**
- Watch for "Connected to master" message
- Check GPU utilization: `nvidia-smi`
- Monitor log file: `tail -f multinode_training.log`

---

## Summary

You're on **Node 2 (innmi1srh2-p021)** with:
- ✅ Virtual environment activated
- ✅ In /workspace/distTest directory

**Next steps:**
1. Run the pre-launch checklist (Step 8)
2. Wait for Node 1 to start
3. Launch Node 2 within 30 seconds
4. Monitor with `nvidia-smi`

**Ready when Node 1 starts!** 🚀
