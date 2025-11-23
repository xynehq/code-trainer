# 🚀 Multi-Node Launch Guide

## ✅ Single-Node Training Verified!

**Results from 8 GPU single-node test:**
- Baseline Perplexity: **15.98** ✓
- Training Loss: **~3.0** ✓  
- Gradients: **Stable (0.8-1.5)** ✓
- Speed: **0.03-0.04 it/s per GPU** ✓

---

## 🎯 Multi-Node Deployment (2 Nodes × 8 GPUs = 16 GPUs Total)

### Prerequisites

1. **Same code on both nodes**: Ensure `/workspace/distTest/` exists on both nodes with identical files
2. **Network accessibility**: Master node (10.11.7.50) must be reachable from worker node
3. **Same environment**: Both nodes should have the same Python environment and dependencies

### Launch Sequence

#### **Step 1: Launch Master Node (Node 0)**

On the **master node (10.11.7.50)**, run:

```bash
cd /workspace/distTest
./launch_node0.sh 2>&1 | tee multinode_training.log
```

**Expected output:**
```
========================================
Launching MASTER NODE (Node 0)
Master: 10.11.7.50:29600
Total Nodes: 2
GPUs per Node: 8
========================================
[NCCL INFO] ... waiting for connections...
```

The master will wait for the worker node to connect.

#### **Step 2: Launch Worker Node (Node 1)**

**Within 30 seconds**, SSH to the **worker node** and run:

```bash
cd /workspace/distTest
./launch_node1.sh 2>&1 | tee multinode_training.log
```

#### **Step 3: Verify Connection**

Once the worker connects, you should see on **both nodes**:
```
Training Configuration:
  Hostname: <hostname>
  World Size: 16
  Number of Nodes: 2
  GPUs per Node: 8
  Master: 10.11.7.50:29600
```

Then training will begin across all 16 GPUs!

---

## 📊 Expected Performance

With **16 GPUs** (2 nodes × 8):
- **Effective batch size**: `1 × 2 (grad_accum) × 16 (GPUs) = 32`
- **Speed improvement**: ~2x faster (6-8 hours instead of 23 hours)
- **Steps per epoch**: ~501 (half of single-node's 1001)
- **Total steps**: ~1502 for 3 epochs

---

## 🔧 Troubleshooting

### Connection Timeout
```
ERROR: Timed out initializing process group
```
**Fix:**
- Start master node first
- Start worker within 30 seconds
- Check firewall allows port 29600

### NCCL Errors
```
NCCL WARN ... Connection refused
```
**Fix:**
- Verify network interface: `ifconfig | grep enp26s0np0`
- Check master IP is correct: `ping 10.11.7.50`
- Ensure NCCL_SOCKET_IFNAME matches on both nodes

### Version Mismatch
```
RuntimeError: NCCL version mismatch
```
**Fix:**
- Ensure same PyTorch version on both nodes
- Reinstall dependencies if needed

---

## 📝 Monitoring Training

### On Master Node:
```bash
# Watch training progress
tail -f multinode_training.log

# Check specific metrics
grep "Step.*Loss" multinode_training.log | tail -20
```

### Check GPU Usage:
```bash
nvidia-smi
```

All 8 GPUs on each node should show ~100% utilization.

---

## 🎯 Ready to Launch!

The scripts are ready:
- ✅ `launch_node0.sh` - Master node script
- ✅ `launch_node1.sh` - Worker node script  
- ✅ `config.yaml` - Tuned for GLM-4.5-Air
- ✅ `train_fsdp.py` - Fixed with proper masking

**Launch master node first, then worker node within 30 seconds!**
