#!/bin/bash

# ============================================================
# NCCL Network Configuration for High-Speed RDMA
# ============================================================

# CRITICAL: Enable InfiniBand/RoCE for RDMA
export NCCL_IB_DISABLE=0           # Enable IB/RoCE
export NCCL_P2P_DISABLE=0          # Enable P2P transfers

# Network Interface Selection (tries bond0 first, then falls back)
export NCCL_SOCKET_IFNAME=bond0,enp26s0np0

# GPU Direct RDMA (if supported by your network cards)
export NCCL_NET_GDR_LEVEL=5        # Max GPU Direct RDMA
export NCCL_IB_GID_INDEX=3         # RoCE v2 (use 0 for RoCE v1)

# Performance Tuning
export NCCL_IB_TIMEOUT=23          # Increase timeout for stability
export NCCL_IB_RETRY_CNT=7         # Retry count for reliability

# Debugging (set to WARN after confirming it works)
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET

echo "========================================="
echo "Launching WORKER NODE (Node 1) with RDMA"
echo "Total Nodes: 2"
echo "GPUs per Node: 8"
echo "Network: RoCE/RDMA (bond0)"
echo "========================================="

# Launch worker node (node 1)
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=10.11.7.50 \
    --master_port=29600 \
    train_fsdp.py \
    --config config.yaml
