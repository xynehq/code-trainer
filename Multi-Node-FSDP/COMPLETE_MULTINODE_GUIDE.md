# 🚀 COMPLETE Multi-Node Training Setup Guide

## Table of Contents
1. [Prerequisites & Architecture](#prerequisites--architecture)
2. [Node 2 Setup (Step-by-Step)](#node-2-setup-step-by-step)
3. [Launch Sequence](#launch-sequence)
4. [What Happens During Training](#what-happens-during-training)
5. [Troubleshooting](#troubleshooting)

---

## Prerequisites & Architecture

### How Multi-Node Training Works

```
┌─────────────────────────────────────────────────────────┐
│                    MASTER NODE (Node 0)                  │
│              IP: 10.11.7.50, Port: 29600                 │
│                                                           │
│  ┌────────┐ ┌────────┐ ┌────────┐ ... ┌────────┐       │
│  │ GPU 0  │ │ GPU 1  │ │ GPU 2  │     │ GPU 7  │       │
│  └────────┘ └────────┘ └────────┘     └────────┘       │
│    Rank 0    Rank 1    Rank 2          Rank 7          │
└─────────────────────────────────────────────────────────┘
                          │
                   NCCL Communication
                   (All-Reduce, Barriers)
                          │
┌─────────────────────────────────────────────────────────┐
│                    WORKER NODE (Node 1)                  │
│              Connects to Master: 10.11.7.50              │
│                                                           │
│  ┌────────┐ ┌────────┐ ┌────────┐ ... ┌────────┐       │
│  │ GPU 0  │ │ GPU 1  │ │ GPU 2  │     │ GPU 7  │       │
│  └────────┘ └────────┘ └────────┘     └────────┘       │
│    Rank 8    Rank 9    Rank 10         Rank 15         │
└─────────────────────────────────────────────────────────┘
```

**Key Concepts:**
- **Each GPU gets a unique global rank** (0-15 for 16 GPUs)
- **FSDP shards the model** across all 16 GPUs
- **Gradients synchronized** via NCCL All-Reduce
- **Data split equally** - each GPU processes different batches
- **Checkpoints saved on all nodes** - sharded state per rank

---

## Node 2 Setup (Step-by-Step)

### Question 1: Do I need the code on Node 2?
**YES!** Both nodes need identical training code.

### Question 2: Do I need the model on Node 2?
**It depends on your setup. Choose ONE option:**

#### Option A: Shared Network Storage (RECOMMENDED)
If `/workspace` is a **shared filesystem** (NFS, Lustre, etc.) visible to both nodes:
- ✅ **Model location**: `/workspace/Avinash/models/GLM-4.5-Air` (already exists)
- ✅ **Dataset location**: `/workspace/Avinash/dataset/all_data.jsonl` (already exists)
- ✅ **No copying needed!** Both nodes read from the same location
- ✅ **Verify**: SSH to Node 2 and run `ls /workspace/Avinash/models/GLM-4.5-Air`

#### Option B: Local Storage on Each Node
If each node has **separate storage**:
- ❌ **Not recommended** - wastes 212GB space and time
- You would need to copy model and dataset to Node 2

**For this guide, we assume SHARED STORAGE (Option A)**

### Question 3: Must paths be the same?
**YES!** The `config.yaml` paths must point to locations accessible from both nodes.

---

## Detailed Setup Steps for Node 2

### Step 1: Verify Node 2 Connectivity

From **Node 0 (this node)**, test connection to Node 2:

```bash
# Find Node 2 IP (replace with your actual Node 2 hostname/IP)
# Example: ping node2-hostname or ping 10.11.7.51

ping <NODE2_IP>  # Should respond
ssh <NODE2_IP>   # Should connect without errors
```

**Expected:** You can SSH to Node 2 successfully.

### Step 2: Check Shared Storage on Node 2

SSH to Node 2 and verify model/dataset are accessible:

```bash
ssh <NODE2_IP>

# Check model exists
ls -lh /workspace/Avinash/models/GLM-4.5-Air
# Expected: You see config.json, pytorch_model.bin files, etc.

# Check dataset exists  
ls -lh /workspace/Avinash/dataset/all_data.jsonl
# Expected: You see the JSONL file (~XXX MB)

# Check workspace is writable
touch /workspace/test_write && rm /workspace/test_write
# Expected: No errors

exit  # Return to Node 0
```

**If files are NOT visible:** Your storage is NOT shared. You'll need to copy files (see Appendix).

### Step 3: Copy Training Code to Node 2

From **Node 0**, copy the distTest directory to Node 2:

```bash
# From Node 0
cd /workspace

# Copy entire distTest directory to Node 2
scp -r distTest <NODE2_IP>:/workspace/

# Verify copy was successful
ssh <NODE2_IP> "ls -la /workspace/distTest"
# Expected: You see train_fsdp.py, config.yaml, launch_node1.sh, etc.
```

### Step 4: Setup Python Environment on Node 2

SSH to Node 2 and install dependencies:

```bash
ssh <NODE2_IP>

cd /workspace/distTest

# Check Python version (should match Node 0)
python3 --version
# Expected: Python 3.10+ 

# Create virtual environment (or use existing one)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch transformers accelerate peft datasets pyyaml tqdm

# Verify CUDA is available
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
# Expected: CUDA available: True, Devices: 8

# Verify NCCL
python3 -c "import torch.distributed as dist; print('NCCL available:', dist.is_nccl_available())"
# Expected: NCCL available: True

exit  # Return to Node 0
```

### Step 5: Verify Network Interface on Node 2

The NCCL network interface must match on both nodes:

```bash
ssh <NODE2_IP>

# Check network interface
ifconfig | grep -A 1 enp26s0np0
# OR
ip addr show enp26s0np0

# Expected: You see the interface with an IP address
# If NOT found, find the correct interface:
ifconfig | grep -E "^[a-z]" | awk '{print $1}'
# Look for interface with your cluster network IP

exit
```

**If interface name is DIFFERENT on Node 2:**
- Edit `launch_node1.sh` on Node 2
- Change `NCCL_SOCKET_IFNAME=enp26s0np0` to match Node 2's interface

---

## Launch Sequence

### Pre-Launch Checklist

On **Node 0**, verify everything is ready:

```bash
cd /workspace/distTest

# ✅ Check config paths
cat config.yaml | grep -E "name_or_path|dataset_path"
# Expected:
#   name_or_path: "/workspace/Avinash/models/GLM-4.5-Air"
#   dataset_path: "/workspace/Avinash/dataset/all_data.jsonl"

# ✅ Check launch scripts are executable
ls -la launch_node*.sh
# Expected: -rwxr-xr-x (executable)

# ✅ Verify Node 2 can be reached
ping -c 3 <NODE2_IP>
# Expected: 0% packet loss
```

### Launch Step-by-Step

#### Step 1: Launch Master Node (Node 0) FIRST

On **Node 0 (this node - 10.11.7.50)**:

```bash
cd /workspace/distTest

# Launch master node
./launch_node0.sh 2>&1 | tee multinode_training.log
```

**Expected Output:**
```
=========================================
Launching MASTER NODE (Node 0)
Master: 10.11.7.50:29600
Total Nodes: 2
GPUs per Node: 8
=========================================

innmi1srh2-p040:XXXXX:XXXXX [0] NCCL INFO Bootstrap : Using enp26s0np0:10.11.7.50<0>
innmi1srh2-p040:XXXXX:XXXXX [0] NCCL INFO NET/Plugin : No plugin found (libnccl-net.so), using internal implementation
innmi1srh2-p040:XXXXX:XXXXX [0] NCCL INFO NET/IB : Using interface enp26s0np0
...
[Master node is now WAITING for worker node to connect]
[THIS WILL WAIT UP TO 30 MINUTES - don't panic!]
```

**🚨 IMPORTANT:** 
- Master will appear "stuck" - this is NORMAL
- It's waiting for the worker node
- You have ~30 minutes to launch worker (but do it ASAP)

#### Step 2: Launch Worker Node (Node 1) IMMEDIATELY

Open a **NEW terminal/SSH session** to Node 2:

```bash
# SSH to Node 2
ssh <NODE2_IP>

# Navigate to training directory
cd /workspace/distTest

# Activate virtual environment if needed
source .venv/bin/activate

# Launch worker node
./launch_node1.sh 2>&1 | tee multinode_training.log
```

**Expected Output on Node 1:**
```
=========================================
Launching WORKER NODE (Node 1)
Master: 10.11.7.50:29600
Total Nodes: 2
GPUs per Node: 8
=========================================

innmi1srh2-p041:XXXXX:XXXXX [0] NCCL INFO Bootstrap : Using enp26s0np0:10.11.7.XX<0>
innmi1srh2-p041:XXXXX:XXXXX [0] NCCL INFO NET/IB : Using interface enp26s0np0
innmi1srh2-p041:XXXXX:XXXXX [0] NCCL INFO Connected to master 10.11.7.50:29600
...
[Worker connecting to master...]
```

#### Step 3: Connection Established

Once worker connects, **BOTH nodes** will show:

```
================================================================================
Training Configuration:
  Hostname: <node-hostname>
  World Size: 16
  Number of Nodes: 2
  GPUs per Node: 8
  Master: 10.11.7.50:29600
================================================================================

Loading model: /workspace/Avinash/models/GLM-4.5-Air
...
```

**Then training begins across all 16 GPUs!**

---

## What Happens During Training

### 1. Initial Synchronization (First ~5 minutes)

**On Node 0 (Rank 0 only):**
- ✅ Loads model from `/workspace/Avinash/models/GLM-4.5-Air`
- ✅ Loads dataset from `/workspace/Avinash/dataset/all_data.jsonl`
- ✅ Tokenizes dataset (all ranks do this)

**On Node 1 (Ranks 8-15):**
- ✅ Loads model (from shared storage or local copy)
- ✅ Loads dataset (from shared storage)
- ✅ Tokenizes dataset

**FSDP Broadcasts model state from Rank 0 to all ranks**

### 2. Data Distribution

Each GPU processes **different batches**:

```
Epoch 1, Step 1:
  Rank 0:  Batch 0  (samples 0-8191)
  Rank 1:  Batch 1  (samples 8192-16383)
  ...
  Rank 15: Batch 15 (samples 122880-131071)
```

### 3. Forward & Backward Pass

Each rank:
1. Forward pass on its batch
2. Computes loss
3. Backward pass (computes gradients)
4. **NCCL All-Reduce**: Synchronizes gradients across all 16 GPUs
5. Optimizer step (each rank updates its model shard)

### 4. Checkpointing

Every `save_steps` (20 steps):

**Each rank saves its own shard:**
```
/workspace/distTest/glm_fsdp_output/checkpoint-20/
  ├── model_state_rank0.pt   (on Node 0, GPU 0)
  ├── model_state_rank1.pt   (on Node 0, GPU 1)
  ...
  ├── model_state_rank8.pt   (on Node 1, GPU 0)
  ...
  ├── model_state_rank15.pt  (on Node 1, GPU 7)
  ├── optimizer.pt           (Rank 0 only)
  ├── scheduler.pt           (Rank 0 only)
  ├── adapter_model.safetensors  (Rank 0 only - consolidated LoRA)
  └── metadata.json          (Rank 0 only)
```

**⚠️ IMPORTANT:** 
- Sharded checkpoints are distributed across nodes
- To resume, you need ALL shard files from ALL nodes
- LoRA adapters (`adapter_model.safetensors`) are only on Rank 0 (Node 0)

### 5. Evaluation

Every `eval_steps` (20 steps):

- All ranks participate in evaluation
- Each rank evaluates different batches
- Losses are gathered and averaged
- **Only Rank 0 prints** the final result

### 6. Logging

**Only Rank 0** (Node 0, GPU 0) prints training logs:
```
Step 1/1502 | Loss: 3.68 | LR: 1.33e-07 | ...
```

Other ranks are silent (but working!).

---

## Monitoring Training

### On Node 0 (Master):

```bash
# Watch live training logs
tail -f /workspace/distTest/multinode_training.log

# Check GPU utilization (should show all 8 GPUs at ~100%)
watch nvidia-smi

# Monitor specific metrics
grep "Step.*Loss" /workspace/distTest/multinode_training.log | tail -20

# Check evaluation results
grep "Eval Loss" /workspace/distTest/multinode_training.log
```

### On Node 1 (Worker):

```bash
# Worker logs (mostly NCCL debug info)
tail -f /workspace/distTest/multinode_training.log

# GPU utilization (should show all 8 GPUs at ~100%)
watch nvidia-smi

# Worker won't print training metrics - this is normal!
```

### Check Both Nodes Are Working:

```bash
# On Node 0
ssh <NODE2_IP> "nvidia-smi | grep python"

# Expected: 8 Python processes using GPUs on Node 2
```

---

## Troubleshooting

### Issue 1: "Connection Timeout" or "Address already in use"

**Error:**
```
RuntimeError: The server socket has failed to listen on any local network address.
```

**Solution:**
```bash
# On Node 0, kill any existing training
pkill -9 -f train_fsdp.py

# Change port in BOTH launch scripts
# Edit launch_node0.sh: --master_port=29601
# Edit launch_node1.sh: --master_port=29601

# Relaunch
```

### Issue 2: "Model/Dataset Not Found" on Node 2

**Error on Node 1:**
```
FileNotFoundError: /workspace/Avinash/models/GLM-4.5-Air
```

**Solution:**
```bash
# Check if path exists on Node 2
ssh <NODE2_IP> "ls /workspace/Avinash/models/GLM-4.5-Air"

# If NOT found, either:
# Option A: Fix mount/shared storage
# Option B: Copy model to Node 2 (see Appendix A)
```

### Issue 3: NCCL WARN "No route to host"

**Error:**
```
NCCL WARN ... connect to 10.11.7.50<12345> failed: No route to host
```

**Solution:**
```bash
# Check firewall on Node 0
sudo firewall-cmd --list-all  # Should allow NCCL ports

# Or disable firewall temporarily (if safe)
sudo systemctl stop firewalld  # On Node 0

# Verify ping works
ping -c 3 10.11.7.50  # From Node 2
```

### Issue 4: Nodes Start But One Hangs

**Symptom:** One node progresses, other is stuck at "Initializing process group"

**Solution:**
```bash
# Kill on BOTH nodes
# On Node 0:
pkill -9 -f train_fsdp.py

# On Node 2:
ssh <NODE2_IP> "pkill -9 -f train_fsdp.py"

# Check NCCL_SOCKET_IFNAME matches on both nodes
# Relaunch: Master first, worker within 30 seconds
```

---

## Appendix A: Copying Model to Node 2 (If Needed)

If you DON'T have shared storage:

```bash
# From Node 0
cd /workspace/Avinash

# Copy model to Node 2 (this will take time - 212GB)
rsync -avz --progress models/GLM-4.5-Air <NODE2_IP>:/workspace/Avinash/models/

# Copy dataset
rsync -avz --progress dataset/all_data.jsonl <NODE2_IP>:/workspace/Avinash/dataset/
```

---

## Quick Reference

### Launch Commands

**Node 0 (Master):**
```bash
cd /workspace/distTest && ./launch_node0.sh 2>&1 | tee multinode_training.log
```

**Node 1 (Worker):**
```bash
cd /workspace/distTest && ./launch_node1.sh 2>&1 | tee multinode_training.log
```

### Stop Training

```bash
# On Node 0
pkill -9 -f train_fsdp.py

# On Node 1
ssh <NODE2_IP> "pkill -9 -f train_fsdp.py"
```

### Resume Training

If training crashed or you stopped it:

```bash
# Same launch commands, but add checkpoint path:
# On Node 0:
./launch_node0.sh --resume_from_checkpoint ./glm_fsdp_output/checkpoint-100

# On Node 1 (within 30 seconds):
./launch_node1.sh --resume_from_checkpoint ./glm_fsdp_output/checkpoint-100
```

**⚠️ CRITICAL:** ALL checkpoint shard files must exist on their original nodes!

---

## Summary Checklist

Before launching, verify:

- [ ] Node 2 is reachable via SSH
- [ ] Model accessible from both nodes (shared storage OR copied)
- [ ] Dataset accessible from both nodes (shared storage OR copied)
- [ ] Training code copied to Node 2 (`/workspace/distTest`)
- [ ] Dependencies installed on Node 2 (torch, transformers, peft, etc.)
- [ ] Network interface correct on both nodes (`ifconfig`)
- [ ] Port 29600 not in use on Node 0
- [ ] Both nodes have GPU access (`nvidia-smi` works)

**Launch sequence:**
1. ✅ Start Node 0 **FIRST** → Wait for "waiting for connections"
2. ✅ Start Node 1 **within 30 seconds**
3. ✅ Both nodes show "World Size: 16"
4. ✅ Training begins!

---

**You're ready for multi-node training! 🚀**
