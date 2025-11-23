#!/bin/bash

# FSDP Multi-Node Training Launch Script
# This script demonstrates how to launch training across multiple nodes

# === CONFIGURATION ===
# Network interface for NCCL communication
export NCCL_SOCKET_IFNAME=enp26s0np0

# Enable NCCL debugging (optional, remove for production)
export NCCL_DEBUG=INFO

# Master node configuration
MASTER_ADDR="10.11.7.50"  # IP address of the master node
MASTER_PORT="29600"       # Port for distributed communication

# Node configuration
NUM_NODES=2               # Total number of nodes
NUM_GPUS_PER_NODE=8       # Number of GPUs per node
NODE_RANK="${1:-0}"       # Node rank (0 for master, 1+ for workers) - passed as argument

# Training configuration
CONFIG_FILE="${2:-config.yaml}"  # Path to config file
RESUME_CHECKPOINT="${3:-}"       # Optional: path to checkpoint for resuming

echo "========================================="
echo "FSDP Multi-Node Training"
echo "========================================="
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Node Rank: ${NODE_RANK}/${NUM_NODES}"
echo "GPUs per Node: ${NUM_GPUS_PER_NODE}"
echo "Config: ${CONFIG_FILE}"
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "Resuming from: ${RESUME_CHECKPOINT}"
fi
echo "========================================="
echo ""

# Build torchrun command
CMD="torchrun \
    --nproc_per_node=${NUM_GPUS_PER_NODE} \
    --nnodes=${NUM_NODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    train_fsdp.py \
    --config ${CONFIG_FILE}"

# Add resume checkpoint if specified
if [ -n "$RESUME_CHECKPOINT" ]; then
    CMD="${CMD} --resume_from_checkpoint ${RESUME_CHECKPOINT}"
fi

# Execute training
echo "Launching training..."
echo "Command: ${CMD}"
echo ""

eval ${CMD}

# === USAGE EXAMPLES ===
#
# Single Node (8 GPUs):
# ./launch_multinode.sh 0 config.yaml
#
# Master Node (Node 0):
# ./launch_multinode.sh 0 config.yaml
#
# Worker Node (Node 1):
# SSH into worker node and run:
# ./launch_multinode.sh 1 config.yaml
#
# Resume from checkpoint:
# ./launch_multinode.sh 0 config.yaml ./output/checkpoint-1000
#
# === NOTES ===
# 1. Update MASTER_ADDR to the IP of your master node
# 2. Update NCCL_SOCKET_IFNAME to match your network interface (run 'ifconfig' to find it)
# 3. Ensure all nodes can communicate on the specified port (check firewall)
# 4. Make sure the same code and config exist on all nodes
# 5. Launch master node first, then worker nodes
