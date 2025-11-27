import os
import math
import json
import time
import shutil
import argparse
import yaml
import traceback
import logging
import gc
import functools
from pathlib import Path
from datetime import datetime, timedelta
import socket

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [Rank %(rank)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def setup_distributed():
    """Initialize distributed training with NCCL backend"""
    if not dist.is_initialized():
        # Extended timeout for multinode setups
        timeout = timedelta(minutes=180)
        dist.init_process_group(backend="nccl", timeout=timeout)
    
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    torch.cuda.set_device(local_rank)
    
    return local_rank, rank, world_size


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def print_rank0(message, rank):
    """Print only from rank 0"""
    if rank == 0:
        print(message)


def log_error(error_log_path, error_type, error_message, step=None, epoch=None, traceback_str=None):
    """Log errors to error.jsonl file"""
    error_entry = {
        "timestamp": time.time(),
        "datetime": datetime.now().isoformat(),
        "error_type": error_type,
        "error_message": str(error_message),
        "step": step,
        "epoch": epoch,
        "traceback": traceback_str
    }
    try:
        with open(error_log_path, "a") as f:
            f.write(json.dumps(error_entry) + "\n")
    except Exception as e:
        print(f"Failed to log error: {e}")


def load_config(config_path):
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def format_time(seconds):
    """Format seconds into human-readable time"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"


def download_model_rank0(model_path, cache_dir, rank):
    """
    Download model on rank 0, other ranks wait and load from cache.
    Supports both HuggingFace repo names and local paths.
    """
    # Check if it's a local path
    if os.path.exists(model_path):
        print_rank0(f"Loading model from local path: {model_path}", rank)
        return model_path
    
    # It's a HuggingFace repo - download on rank 0 first
    if rank == 0:
        print(f"Rank 0: Downloading model from HuggingFace: {model_path}")
        from transformers import AutoModel
        try:
            # Download model files to cache
            AutoModel.from_pretrained(
                model_path,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            print(f"Rank 0: Model downloaded successfully")
        except Exception as e:
            print(f"Rank 0: Error downloading model: {e}")
            raise
    
    # Synchronize all ranks
    dist.barrier()
    
    return model_path


def load_and_prepare_model(config, local_rank, rank, world_size):
    """Load model and wrap with FSDP + LoRA"""
    model_path = config['model']['name_or_path']
    cache_dir = config['model'].get('cache_dir', None)
    
    # Download model if needed (rank 0 first, others wait)
    model_path = download_model_rank0(model_path, cache_dir, rank)
    
    print_rank0(f"Loading model: {model_path}", rank)
    
    # CRITICAL: Load model on CPU first to avoid OOM
    # FSDP will shard it across GPUs automatically
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=config['model'].get('trust_remote_code', True),
        torch_dtype=torch.bfloat16 if config['training']['bf16'] else torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map=None,  # Don't auto-distribute, let FSDP handle it
    )
    
    # Disable cache for training
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    
    # Enable gradient checkpointing if configured
    if config['training'].get('gradient_checkpointing', False):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        print_rank0("✓ Gradient checkpointing enabled", rank)
    
    # Add LoRA BEFORE moving to GPU or wrapping with FSDP
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias'],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    
    if rank == 0:
        model.print_trainable_parameters()
    
    # CRITICAL: Ensure all parameters are in the same dtype for FSDP
    # LoRA parameters are initialized in FP32 by default
    target_dtype = torch.bfloat16 if config['training']['bf16'] else torch.float16
    for param in model.parameters():
        if param.dtype != target_dtype:
            param.data = param.data.to(target_dtype)
    
    print_rank0(f"✓ All parameters converted to {target_dtype}", rank)
    
    # Wrap with FSDP BEFORE moving to GPU
    # FSDP will handle moving shards to GPU
    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy,
        min_num_params=200000000  # Wrap layers with >200M params
    )
    
    fsdp_config = {
        "auto_wrap_policy": auto_wrap_policy,
        "mixed_precision": None,  # We handle precision via torch_dtype
        "sharding_strategy": torch.distributed.fsdp.ShardingStrategy.FULL_SHARD,
        "device_id": torch.cuda.current_device(),
        "sync_module_states": True,
        "use_orig_params": True,
    }
    
    model = FSDP(model, **fsdp_config)
    
    print_rank0("✓ Model wrapped with FSDP", rank)
    
    return model


def prepare_datasets(config, tokenizer, rank):
    """Load and prepare train/eval datasets"""
    dataset_path = config['data']['dataset_path']
    text_column = config['data']['text_column']
    max_length = config['data']['max_length']
    val_split = config['data'].get('val_split', 0.1)
    
    print_rank0(f"Loading dataset from: {dataset_path}", rank)
    
    # Load dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors=None
        )
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    # Add labels (same as input_ids for causal LM)
    def add_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples
    
    tokenized_dataset = tokenized_dataset.map(add_labels, batched=True)
    
    # Split train/eval
    split = tokenized_dataset.train_test_split(test_size=val_split, seed=42)
    train_dataset = split["train"]
    eval_dataset_full = split["test"]
    
    # Apply deterministic eval dataset reduction if configured
    # CRITICAL: Use SAME subset every time to avoid jittery loss curves
    eval_dataset_type = config['training'].get('eval_dataset_type', 'full')
    if eval_dataset_type == 'subset':
        eval_subset_size = config['training'].get('eval_subset_size', 160)
        original_eval_size = len(eval_dataset_full)
        
        # Deterministic selection: shuffle with fixed seed, then take first N
        # This ensures the SAME samples are used every evaluation
        eval_dataset = eval_dataset_full.shuffle(seed=42).select(range(min(eval_subset_size, original_eval_size)))
        
        print_rank0(f"Using deterministic eval subset: {len(eval_dataset)}/{original_eval_size} samples (seed=42, eval_dataset_type='{eval_dataset_type}')", rank)
    else:
        eval_dataset = eval_dataset_full
        print_rank0(f"Using full eval dataset: {len(eval_dataset)} samples", rank)
    
    print_rank0(f"Train samples: {len(train_dataset)}", rank)
    print_rank0(f"Eval samples: {len(eval_dataset)}", rank)
    
    return train_dataset, eval_dataset


def create_dataloaders(train_dataset, eval_dataset, config, rank, world_size):
    """Create distributed dataloaders"""
    batch_size = config['training']['per_device_train_batch_size']
    eval_batch_size = config['training'].get('per_device_eval_batch_size', batch_size)
    
    # Distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=config['training']['seed']
    )
    
    eval_sampler = DistributedSampler(
        eval_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # Collate function with PROPER LABEL MASKING
    def collate_fn(batch):
        input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
        attention_mask = torch.stack([torch.tensor(item['attention_mask']) for item in batch])
        labels = torch.stack([torch.tensor(item['labels']) for item in batch])
        
        # CRITICAL: Mask padding tokens in labels by setting them to -100
        # CrossEntropyLoss ignores index -100, preventing padding from affecting loss
        # This is the key fix for proper perplexity calculation!
        labels[attention_mask == 0] = -100
        
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        sampler=eval_sampler,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    
    return train_dataloader, eval_dataloader, train_sampler


def evaluate(model, eval_dataloader, rank, world_size):
    """Optimized evaluation with minimal synchronization - single all-reduce"""
    from tqdm import tqdm
    
    model.eval()
    
    # Accumulate locally WITHOUT synchronization (key optimization!)
    local_total_loss = 0.0
    local_num_batches = 0
    
    # Single barrier at start
    dist.barrier()
    
    # Progress bar only on rank 0
    if rank == 0:
        pbar = tqdm(eval_dataloader, desc="Evaluating", leave=False)
    else:
        pbar = eval_dataloader
    
    with torch.no_grad():
        for batch in pbar:
            batch = {k: v.cuda() for k, v in batch.items()}
            outputs = model(**batch)
            
            # Accumulate locally - NO all_reduce here! (was the bottleneck)
            local_total_loss += outputs.loss.item()
            local_num_batches += 1
    
    # SINGLE all-reduce at the end instead of 112!
    # This reduces network overhead from ~33s×112 to ~33s×1
    stats = torch.tensor([local_total_loss, local_num_batches], 
                         dtype=torch.float32, device='cuda')
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    
    # Calculate global average
    global_total_loss = stats[0].item()
    global_num_batches = stats[1].item()
    
    avg_loss = global_total_loss / global_num_batches if global_num_batches > 0 else float('inf')
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    
    model.train()
    
    # Single barrier at end
    dist.barrier()
    
    return {"eval_loss": avg_loss, "eval_perplexity": perplexity}


def save_checkpoint(model, tokenizer, optimizer, scheduler, output_dir, step, rank, config, eval_loss=None):
    """
    Smart Checkpointing: Lite vs Full mode
    - Lite: Only LoRA adapters (~1GB, 10 seconds) - For experimentation
    - Full: Optimizer + adapters (~200GB, 15+ mins) - For production resume
    """
    dist.barrier()
    
    checkpoint_name = f"checkpoint-{step}"
    save_dir = os.path.join(output_dir, checkpoint_name)
    
    # Get checkpoint mode from config (default to "lite" for fast saves)
    checkpoint_mode = config['training'].get('checkpoint_mode', 'lite')
    
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n{'='*60}")
        if checkpoint_mode == 'lite':
            print(f"⚡ LITE CHECKPOINT at Step {step} (Adapters Only)")
        else:
            print(f"💾 FULL CHECKPOINT at Step {step} (Optimizer + Adapters)")
        print(f"{'='*60}")
    
    dist.barrier()
    
    # ============================================================
    # MODE 1: LITE SAVE (Fast - 10 seconds)
    # ROBUST FIX: Manual LoRA filtering from FSDP state dict
    # ============================================================
    if checkpoint_mode == 'lite':
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, ShardedStateDictConfig
        from safetensors.torch import save_file
        
        # All ranks create directory (safe with exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        
        if rank == 0:
            print(f"⚡ SHARDED CHECKPOINT (Network Bypass Mode)...")
        
        # ============================================================
        # SHARDED SAVE: Zero Network Traffic, Parallel I/O
        # Each GPU saves its own ~15MB LoRA shard independently
        # ============================================================
        
        # 1. Config: Offload to CPU to save VRAM
        save_policy = ShardedStateDictConfig(offload_to_cpu=True)
        
        # 2. Get LOCAL shard only - NO network communication!
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, save_policy):
            # This returns ONLY the local shard for this GPU (~12GB base + LoRA)
            cpu_state = model.state_dict()
            
            # 3. Filter: Keep ONLY LoRA keys (reduces to ~15MB per GPU)
            lora_state = {
                k: v for k, v in cpu_state.items() 
                if "lora_" in k or "modules_to_save" in k
            }
            
            # 4. Clean keys and convert ShardedTensor to regular tensors
            clean_lora_state = {}
            for k, v in lora_state.items():
                new_k = k.replace("_fsdp_wrapped_module.", "")
                # Extract local tensor from ShardedTensor
                if hasattr(v, 'local_shards') and len(v.local_shards()) > 0:
                    clean_lora_state[new_k] = v.local_shards()[0].tensor
                else:
                    clean_lora_state[new_k] = v
            
            # 5. Save unique file per rank (~15MB each)
            # Files: adapter_model.rank0.safetensors, rank1.safetensors, ...
            shard_file = f"adapter_model.rank{rank}.safetensors"
            shard_path = os.path.join(save_dir, shard_file)
            
            save_file(clean_lora_state, shard_path)
            
            print(f"[Rank {rank}] Saved {len(clean_lora_state)} keys to {shard_file}")
            
            # Cleanup
            del cpu_state
            del lora_state
            del clean_lora_state
            gc.collect()
        
        # 6. Metadata (Rank 0 only)
        if rank == 0:
            tokenizer.save_pretrained(save_dir)
            metadata = {
                "step": step,
                "eval_loss": eval_loss,
                "checkpoint_mode": "sharded_lora",
                "format": "sharded_lora_safetensors",
                "num_shards": dist.get_world_size(),
                "timestamp": time.time(),
                "datetime": datetime.now().isoformat(),
            }
            with open(os.path.join(save_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
        
        # 7. Final barrier - everyone waits for parallel writes to complete
        dist.barrier()
        
        if rank == 0:
            print(f"✅ SHARDED CHECKPOINT COMPLETE (~10s)")
            print(f"{'='*60}\n")
    
    
    # ============================================================
    # MODE 2: FULL SAVE (Slow - 15+ minutes on TCP)
    # Saves optimizer + scheduler + model for exact resume
    # ============================================================
    else:  # checkpoint_mode == 'full'
        if rank == 0:
            print(f"Saving full checkpoint (optimizer + adapters)...")
        
        # Save optimizer shards (each rank saves its own)
        optimizer_path = os.path.join(save_dir, f"optimizer_rank{rank}.pt")
        scheduler_path = os.path.join(save_dir, f"scheduler_rank{rank}.pt")
        
        torch.save(optimizer.state_dict(), optimizer_path)
        torch.save(scheduler.state_dict(), scheduler_path)
        
        if rank == 0:
            print(f"All ranks saved optimizer/scheduler shards")
        
        # Gather full model state to CPU (SLOW on TCP!)
        from torch.distributed.fsdp import StateDictType, FullStateDictConfig
        
        if rank == 0:
            print(f"Gathering model state to CPU (this will take 10-15 minutes on TCP)...")
        
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            cpu_state = model.state_dict()
            
            if rank == 0:
                # Filter LoRA params
                lora_state = {
                    k: v for k, v in cpu_state.items() 
                    if "lora_" in k or "modules_to_save" in k
                }
                
                # Save adapters
                if hasattr(model, 'module'):
                    peft_model = model.module
                else:
                    peft_model = model
                
                peft_model.save_pretrained(save_dir, state_dict=lora_state, safe_serialization=True)
                tokenizer.save_pretrained(save_dir)
                
                # Save metadata
                metadata = {
                    "step": step,
                    "eval_loss": eval_loss,
                    "checkpoint_mode": "full",
                    "timestamp": time.time(),
                    "datetime": datetime.now().isoformat(),
                    "world_size": dist.get_world_size(),
                }
                with open(os.path.join(save_dir, "metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=2)
                
                # Cleanup
                del cpu_state
                del lora_state
                gc.collect()
                
                print(f"✓ Saved full checkpoint to {save_dir}")
                print(f"{'='*60}")
                print(f"✅ FULL CHECKPOINT COMPLETE")
                print(f"{'='*60}\n")
            else:
                # Other ranks participate in gather but get nothing
                model.state_dict()
    
    dist.barrier()
    return checkpoint_name
    """
    Robust FSDP Multi-Node Checkpointing:
    1. Uses FSDP's state_dict context manager with CPU offloading
    2. Gathers only LoRA params to CPU RAM (not GPU) to avoid OOM
    3. Saves sharded optimizer state per rank
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
    
    dist.barrier()
    
    checkpoint_name = f"checkpoint-{step}"
    save_dir = os.path.join(output_dir, checkpoint_name)
    
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        print(f"\\n{'='*60}")
        print(f"💾 SAVING CHECKPOINT at Step {step}")
        print(f"{'='*60}")
    
    dist.barrier()
    
    # --- A. SAVE SHARDED OPTIMIZER/SCHEDULER STATE (per rank) ---
    # Each rank saves its own optimizer shard - this is fast and doesn't require gathering
    if rank == 0:
        print(f"Saving optimizer and scheduler state...")
    
    optimizer_path = os.path.join(save_dir, f"optimizer_rank{rank}.pt")
    scheduler_path = os.path.join(save_dir, f"scheduler_rank{rank}.pt")
    
    torch.save(optimizer.state_dict(), optimizer_path)
    torch.save(scheduler.state_dict(), scheduler_path)
    
    if rank == 0:
        print(f"[Rank {rank}] Saved optimizer/scheduler shards")
    
    # --- B. SAVE FULL MODEL STATE WITH CPU OFFLOADING ---
    # CRITICAL: Use FullStateDictConfig to gather to CPU RAM, not GPU
    # This avoids the NCCL timeout from trying to all-gather on GPU
    if rank == 0:
        print(f"Gathering model state to CPU (avoiding GPU OOM)...")
    
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        # This gathers the FULL model state dict to CPU RAM on rank 0 only
        # With H200 nodes having 1TB+ CPU RAM, 212GB model fits easily
        cpu_state = model.state_dict()
        
        if rank == 0:
            # Filter: We only want LoRA parameters (trainable)
            lora_state = {
                k: v for k, v in cpu_state.items() 
                if "lora_" in k or "modules_to_save" in k
            }
            
            print(f"Saving LoRA adapters ({len(lora_state)} parameters)...")
            
            # Save using safetensors format
            from peft import PeftModel
            adapter_path = os.path.join(save_dir, "adapter_model.safetensors")
            
            # Get the base PEFT model to access save_pretrained
            if hasattr(model, 'module'):
                peft_model = model.module  # Unwrap FSDP
            else:
                peft_model = model
            
            # Save LoRA adapters
            peft_model.save_pretrained(
                save_dir,
                state_dict=lora_state,
                safe_serialization=True
            )
            
            # Save tokenizer
            tokenizer.save_pretrained(save_dir)
            
            # Save training metadata
            metadata = {
                "step": step,
                "eval_loss": eval_loss,
                "timestamp": time.time(),
                "datetime": datetime.now().isoformat(),
                "world_size": dist.get_world_size(),
            }
            with open(os.path.join(save_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✓ Saved LoRA adapters and metadata to {save_dir}")
            
            # Clean up CPU memory
            del cpu_state
            del lora_state
            gc.collect()
            
            # Handle checkpoint rotation
            if eval_loss is not None:
                # Rotate old checkpoints
                existing_checkpoints = sorted(
                    [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
                    key=lambda x: int(x.split("-")[1])
                )
                
                max_checkpoints = config['training'].get('save_total_limit', 3)
                if len(existing_checkpoints) > max_checkpoints:
                    for old_ckpt in existing_checkpoints[:-max_checkpoints]:
                        old_ckpt_path = os.path.join(output_dir, old_ckpt)
                        if old_ckpt != checkpoint_name and os.path.exists(old_ckpt_path):
                            shutil.rmtree(old_ckpt_path, ignore_errors=True)
                            print(f"Removed old checkpoint: {old_ckpt}")
    
    # Final barrier to ensure all ranks complete before continuing
    dist.barrier()
    
    if rank == 0:
        print(f"{'='*60}")
        print(f"✓ Checkpoint {checkpoint_name} saved successfully")
        print(f"{'='*60}\n")
    
    return checkpoint_name


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, rank):
    """Load checkpoint for resuming training"""
    print_rank0(f"Loading checkpoint from {checkpoint_path}", rank)
    
    # Load sharded model state
    model_state_path = os.path.join(checkpoint_path, f"model_state_rank{rank}.pt")
    if os.path.exists(model_state_path):
        model_state = torch.load(model_state_path, map_location=f"cuda:{rank}")
        model.load_state_dict(model_state)
        print_rank0(f"✓ Loaded model state from {model_state_path}", rank)
    
    # Load optimizer and scheduler on all ranks
    optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
    scheduler_path = os.path.join(checkpoint_path, "scheduler.pt")
    
    if os.path.exists(optimizer_path):
        optimizer_state = torch.load(optimizer_path, map_location=f"cuda:{rank}")
        optimizer.load_state_dict(optimizer_state)
        print_rank0("✓ Loaded optimizer state", rank)
    
    if os.path.exists(scheduler_path):
        scheduler_state = torch.load(scheduler_path, map_location=f"cuda:{rank}")
        scheduler.load_state_dict(scheduler_state)
        print_rank0("✓ Loaded scheduler state", rank)
    
    # Parse step from checkpoint name
    try:
        step = int(checkpoint_path.split("-")[-1])
        print_rank0(f"✓ Resuming from step {step}", rank)
        return step
    except:
        print_rank0("Could not parse step from checkpoint name, starting from 0", rank)
        return 0


def log_expert_usage(model, outputs, step, rank, log_file_path):
    """Log MoE expert usage statistics if available"""
    if not (hasattr(outputs, "router_logits") and outputs.router_logits is not None):
        return
    
    if rank != 0:
        return
    
    # Get model config
    unwrapped = model.module
    config = unwrapped.config if hasattr(unwrapped, "config") else model.config
    
    num_experts = getattr(config, "n_routed_experts", None) or getattr(config, "num_experts", None)
    if num_experts is None:
        return
    
    num_layers = len(outputs.router_logits)
    layer_indices_to_log = [0, num_layers // 2]
    
    expert_usage_data = {}
    
    for layer_idx in layer_indices_to_log:
        if layer_idx >= len(outputs.router_logits):
            continue
        
        router_logit = outputs.router_logits[layer_idx]
        expert_choices = torch.argmax(router_logit, dim=-1)
        
        unique_experts, expert_counts = torch.unique(expert_choices.flatten(), return_counts=True)
        
        full_usage = torch.zeros(num_experts, dtype=torch.long, device=expert_choices.device)
        full_usage[unique_experts] = expert_counts
        
        total_usage = full_usage.sum()
        expert_usage_percent = full_usage.float() / total_usage * 100 if total_usage > 0 else full_usage.float()
        
        expert_usage_data[f"layer_{layer_idx}"] = {
            "expert_counts": {f"expert_{i}": int(count) for i, count in enumerate(full_usage)},
            "expert_percentages": {f"expert_{i}": round(float(percent), 2) for i, percent in enumerate(expert_usage_percent)},
            "total_tokens": int(total_usage)
        }
    
    if log_file_path and step % 100 == 0:
        log_entry = {
            "step": step,
            "timestamp": time.time(),
            "expert_usage": expert_usage_data
        }
        with open(log_file_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")


def main():
    parser = argparse.ArgumentParser(description="FSDP Multi-node Training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Setup distributed
    local_rank, rank, world_size = setup_distributed()
    
    # Print node info from EACH rank to confirm all nodes are active
    hostname = socket.gethostname()
    print(f"[Rank {rank}] Node: {hostname} | Local Rank: {local_rank} | GPU: cuda:{local_rank}")
    
    # Wait for all ranks to print their info
    dist.barrier()
    
    # Master prints summary
    if rank == 0:
        print("\n" + "=" * 80)
        print(f"✓ ALL NODES CONNECTED - Distributed Training Ready")
        print("=" * 80)
        print(f"Training Configuration:")
        print(f"  Master Node: {hostname}")
        print(f"  World Size: {world_size} GPUs across {os.environ.get('NNODES', 'N/A')} nodes")
        print(f"  GPUs per Node: {os.environ.get('NPROC_PER_NODE', 'N/A')}")
        print(f"  Master Address: {os.environ.get('MASTER_ADDR', 'N/A')}:{os.environ.get('MASTER_PORT', 'N/A')}")
        print(f"  Backend: NCCL")
        print(f"  Timeout: 30 minutes")
        print("=" * 80)
    
    # Load config
    config = load_config(args.config)
    
    # Setup directories
    output_dir = config['training']['output_dir']
    logs_dir = os.path.join(output_dir, "logs")
    
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
    
    dist.barrier()
    
    # Set seed
    torch.manual_seed(config['training']['seed'])
    torch.cuda.manual_seed_all(config['training']['seed'])
    
    # Load tokenizer
    print_rank0("Loading tokenizer...", rank)
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['name_or_path'],
        trust_remote_code=config['model'].get('trust_remote_code', True),
        cache_dir=config['model'].get('cache_dir', None)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load and prepare model
    if rank == 0:
        print("\n" + "=" * 80)
        print("PHASE 1: Model Loading and FSDP Initialization")
        print("=" * 80)
    dist.barrier()
    
    model = load_and_prepare_model(config, local_rank, rank, world_size)
    
    # Confirm FSDP is ready on all ranks
    dist.barrier()
    if rank == 0:
        print("\n" + "=" * 80)
        print("✓ FSDP MODEL READY on all ranks")
        print(f"✓ Model sharded across {world_size} GPUs")
        print("=" * 80)
    
    # Prepare datasets
    if rank == 0:
        print("\n" + "=" * 80)
        print("PHASE 2: Dataset Loading and Tokenization")
        print("=" * 80)
    dist.barrier()
    train_dataset, eval_dataset = prepare_datasets(config, tokenizer, rank)
    
    # Create dataloaders
    train_dataloader, eval_dataloader, train_sampler = create_dataloaders(
        train_dataset, eval_dataset, config, rank, world_size
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config['training']['learning_rate']),
        weight_decay=config['training'].get('weight_decay', 0.01),
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Calculate training steps
    num_epochs = config['training']['num_train_epochs']
    gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
    
    num_batches_per_epoch = len(train_dataloader)
    num_update_steps_per_epoch = num_batches_per_epoch // gradient_accumulation_steps
    max_train_steps = num_epochs * num_update_steps_per_epoch
    
    warmup_ratio = config['training'].get('warmup_ratio', 0.1)
    num_warmup_steps = int(warmup_ratio * max_train_steps)
    
    # Scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps
    )
    
    if rank == 0:
        print("\n" + "=" * 80)
        print("PHASE 3: Training Configuration")
        print("=" * 80)
        print(f"Training configuration:")
        print(f"  Batches per epoch (per GPU): {num_batches_per_epoch}")
        print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"  Update steps per epoch: {num_update_steps_per_epoch}")
        print(f"  Total training steps: {max_train_steps}")
        print(f"  Warmup steps: {num_warmup_steps}")
        print(f"  Effective batch size: {config['training']['per_device_train_batch_size']} × {gradient_accumulation_steps} × {world_size} = {config['training']['per_device_train_batch_size'] * gradient_accumulation_steps * world_size}")
        print("=" * 80)
    
    # Resume from checkpoint if specified
    start_step = 0
    if args.resume_from_checkpoint:
        start_step = load_checkpoint(model, optimizer, scheduler, args.resume_from_checkpoint, rank)
    
    # Setup logging paths
    train_log_path = os.path.join(logs_dir, "train_log.jsonl")
    eval_log_path = os.path.join(logs_dir, "eval_log.jsonl")
    expert_log_path = os.path.join(logs_dir, "expert_usage.jsonl")
    error_log_path = os.path.join(logs_dir, "errors.jsonl")
    
    # Confirm all nodes are ready to start training
    dist.barrier()
    
    # Training loop
    if rank == 0:
        print("\n" + "=" * 80)
        print("🚀 STARTING MULTI-NODE TRAINING")
        print("=" * 80)
        print(f"All {world_size} GPUs across {os.environ.get('NNODES', 'N/A')} nodes are ready!")
        print(f"Each node will process different batches in parallel.")
        print(f"Gradients will be synchronized via NCCL All-Reduce.")
        print("=" * 80 + "\n")
    
    # Final barrier before training
    dist.barrier()
    
    global_step = start_step
    best_eval_loss = float('inf')
    start_time = time.time()
    log_loss = 0
    last_log_time = start_time
    
    # Baseline evaluation
    global_step = start_step
    best_eval_loss = float('inf')
    
    # Baseline evaluation (optional based on config)
    if config['training'].get('run_baseline_eval', False):
        if rank == 0:
            print("\nRunning baseline evaluation...")
        baseline_metrics = evaluate(model, eval_dataloader, rank, world_size)
        
        if rank == 0:
            print(f"✓ Baseline Eval Loss: {baseline_metrics['eval_loss']:.4f} | Perplexity: {baseline_metrics['eval_perplexity']:.2f}             \n")
            
            # Log baseline metrics
            baseline_log = {
                "step": 0,
                "epoch": 0,
                "eval_loss": baseline_metrics['eval_loss'],
                "eval_perplexity": baseline_metrics['eval_perplexity'],
                "timestamp": datetime.now().isoformat()
            }
            with open(eval_log_path, 'a') as f:
                f.write(json.dumps(baseline_log) + '\n')
            
            best_eval_loss = baseline_metrics['eval_loss']
    else:
        if rank == 0:
            print("\nSkipping baseline evaluation (run_baseline_eval=False)\n")
    
    for epoch in range(num_epochs):
        if rank == 0:
            print("\n" + "=" * 80)
            print(f"EPOCH {epoch + 1}/{num_epochs}")
            print("=" * 80)
        
        # Synchronize before each epoch
        dist.barrier()
        
        # Set epoch for sampler (ensures different shuffling each epoch)
        train_sampler.set_epoch(epoch)
        
        model.train()
        
        for step, batch in enumerate(train_dataloader):
            try:
                # Move batch to GPU
                batch = {k: v.cuda() for k, v in batch.items()}
                
                # Forward pass
                outputs = model(**batch, output_router_logits=True)
                loss = outputs.loss
                
                # Check for NaN/Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    if rank == 0:
                        log_error(error_log_path, "NaNLoss", "NaN/Inf loss detected", global_step, epoch)
                    print_rank0(f"⚠️  NaN/Inf loss at step {global_step}! Skipping...", rank)
                    optimizer.zero_grad()
                    continue
                
                # Backward pass
                loss = loss / gradient_accumulation_steps
                loss.backward()
                
                # Update weights every gradient_accumulation_steps
                if (step + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                    log_loss += loss.item() * gradient_accumulation_steps
                    
                    # Logging
                    if global_step % config['training']['logging_steps'] == 0:
                        avg_loss = log_loss / config['training']['logging_steps']
                        log_loss = 0
                        
                        current_time = time.time()
                        time_since_last_log = current_time - last_log_time
                        smooth_speed = config['training']['logging_steps'] / time_since_last_log if time_since_last_log > 0 else 0
                        last_log_time = current_time
                        
                        remaining_steps = max_train_steps - global_step
                        eta_seconds = remaining_steps / smooth_speed if smooth_speed > 0 else 0
                        eta_str = format_time(eta_seconds)
                        
                        lr = scheduler.get_last_lr()[0]
                        
                        if rank == 0:
                            log_entry = {
                                "step": global_step,
                                "epoch": epoch + (step / num_batches_per_epoch),
                                "loss": avg_loss,
                                "lr": lr,
                                "steps_per_sec": smooth_speed,
                                "grad_norm": float(total_norm),
                                "eta_seconds": eta_seconds
                            }
                            
                            with open(train_log_path, "a") as f:
                                f.write(json.dumps(log_entry) + "\n")
                            
                            print(f"Step {global_step}/{max_train_steps} | "
                                  f"Loss: {avg_loss:.4f} | "
                                  f"LR: {lr:.2e} | "
                                  f"Speed: {smooth_speed:.2f} it/s | "
                                  f"Grad: {float(total_norm):.2f} | "
                                  f"ETA: {eta_str}")
                    
                    # Expert usage logging
                    if global_step % 100 == 0:
                        log_expert_usage(model, outputs, global_step, rank, expert_log_path)
                    
                    # Evaluation
                    just_evaluated = False
                    if config['training'].get('eval_strategy') == "steps" and global_step % config['training']['eval_steps'] == 0:
                        print_rank0(f"\nRunning evaluation at step {global_step}...", rank)
                        metrics = evaluate(model, eval_dataloader, rank, world_size)
                        eval_loss = metrics['eval_loss']
                        just_evaluated = True
                        
                        if rank == 0:
                            print(f"✓ Eval Loss: {eval_loss:.4f} | Perplexity: {metrics['eval_perplexity']:.2f}")
                            with open(eval_log_path, "a") as f:
                                metrics['step'] = global_step
                                f.write(json.dumps(metrics) + "\n")
                            
                            if eval_loss < best_eval_loss:
                                best_eval_loss = eval_loss
                                print(f"🎉 New best eval loss: {best_eval_loss:.4f}")
                    
                    # Checkpointing
                    if global_step % config['training']['save_steps'] == 0:
                        checkpoint_eval_loss = best_eval_loss if (just_evaluated and best_eval_loss != float('inf')) else None
                        
                        save_checkpoint(
                            model, tokenizer, optimizer, scheduler,
                            output_dir, global_step, rank, config,
                            checkpoint_eval_loss
                        )
            
            except torch.cuda.OutOfMemoryError as e:
                error_msg = f"OOM at step {global_step}: {str(e)}"
                if rank == 0:
                    log_error(error_log_path, "OOM", error_msg, global_step, epoch, traceback.format_exc())
                print_rank0(f"❌ {error_msg}", rank)
                
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                dist.barrier()
                continue
            
            except Exception as e:
                error_msg = f"Error at step {global_step}: {str(e)}"
                if rank == 0:
                    log_error(error_log_path, "TrainingError", error_msg, global_step, epoch, traceback.format_exc())
                    traceback.print_exc()
                print_rank0(f"❌ {error_msg}", rank)
                
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                dist.barrier()
                continue
    
    print_rank0("\n" + "=" * 80, rank)
    print_rank0("Training completed!", rank)
    print_rank0("=" * 80, rank)
    
    # Final checkpoint
    save_checkpoint(
        model, tokenizer, optimizer, scheduler,
        output_dir, global_step, rank, config, None
    )
    
    cleanup_distributed()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        traceback.print_exc()
        raise
