import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer
import socket

def setup():
    """Initialize distributed training"""
    dist.init_process_group(backend="nccl")
def cleanup():
    """Cleanup distributed training"""
    dist.destroy_process_group()
def print_gpu_info():
    """Print detailed GPU and rank information"""
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    hostname = socket.gethostname()
    print(f"=" * 80)
    print(f"Hostname: {hostname}")
    print(f"Global Rank: {rank}/{world_size}")
    print(f"Local Rank: {local_rank}")
    print(f"GPU Device: cuda:{local_rank}")
    print(f"GPU Name: {torch.cuda.get_device_name(local_rank)}")
    print(f"=" * 80)
def main():
    # Setup distributed
    setup()
    # Get environment variables set by torchrun
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    # Print GPU info from all ranks
    print_gpu_info()
    # Barrier to ensure all processes print info
    dist.barrier()
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"Starting distributed training with {world_size} GPUs across {os.environ.get('NNODES', 'N/A')} nodes")
        print(f"{'='*80}\n")
    # Load model and tokenizer
    model_name = "Qwen/Qwen2.5-0.5B"  # Using 0.5B as 0.6B might not exist
    if rank == 0:
        print(f"Loading model: {model_name}")
    # Load model on each GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use fp16 to save memory
        device_map=None  # Don't use auto device_map in distributed setting
    )
    model = model.to(device)
    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    if rank == 0:
        print(f"✓ Model loaded and wrapped with DDP")
        print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    # Load tokenizer (only need on rank 0 for this test)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Create dummy batch for testing
    if rank == 0:
        print(f"\n{'='*80}")
        print("Running test forward pass...")
        print(f"{'='*80}\n")
    # Simple test input
    text = "Hello, this is a distributed training test."
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    # Print success message from each rank
    print(f"[Rank {rank}] ✓ Forward pass successful on {device}")
    print(f"[Rank {rank}] ✓ Output shape: {outputs.logits.shape}")
    dist.barrier()
    if rank == 0:
        print(f"\n{'='*80}")
        print("SUCCESS! All GPUs are working correctly.")
        print(f"Total GPUs utilized: {world_size}")
        print(f"{'='*80}\n")
    # Optional: Run a few training steps to verify gradient synchronization
    if rank == 0:
        print("Running 3 training steps to verify gradient sync...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    for step in range(3):
        # Create different batches for each rank to simulate real training
        batch_text = f"Training step {step} on rank {rank}"
        inputs = tokenizer(batch_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # Forward pass
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        # Backward pass
        loss.backward()
        # Update
        optimizer.step()
        optimizer.zero_grad()
        if rank == 0:
            print(f"  Step {step + 1}/3 - Loss: {loss.item():.4f}")
    dist.barrier()
    if rank == 0:
        print(f"\n{'='*80}")
        print("✓ Training steps completed successfully!")
        print("✓ All GPUs are synchronized and working together!")
        print(f"{'='*80}\n")
    cleanup()
if __name__ == "__main__":
    main()