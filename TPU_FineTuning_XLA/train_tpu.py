#!/usr/bin/env python3
"""
TPU Training Script for Qwen2.5-Coder-7B-Instruct with LoRA
Optimized for 8 x v6e TPUs with wandb integration
"""

import os
import sys
import json
import yaml
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import math
from tqdm import tqdm

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
from torch.utils.data import Dataset, DataLoader

# Workaround: Some libraries check for torch.xla, make it available
import sys
sys.modules['torch'].xla = torch_xla

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

import wandb

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [Rank %(rank)s] - %(message)s',
)
logger = logging.getLogger(__name__)


class HyperswitchDataset(Dataset):
    """Dataset loader for the prepared Hyperswitch JSONL data"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 8192):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        # Load JSONL data
        with open(data_path, 'r') as f:
            for line in f:
                self.samples.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.samples)} samples from {data_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample['text']
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # For causal LM, labels are the same as input_ids
        labels = input_ids.clone()
        # Mask padding tokens in labels
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_wandb(config: dict, rank: int):
    """Initialize wandb for experiment tracking"""
    if rank == 0 and config['wandb']['enabled']:
        wandb.login(key=config['wandb']['api_key'])
        wandb.init(
            project=config['wandb']['project'],
            entity=config['wandb']['entity'],
            name=config['wandb']['run_name'],
            config={
                'model': config['model']['name'],
                'model_size': config['model']['size'],
                'lora_r': config['training']['lora']['r'],
                'lora_alpha': config['training']['lora']['alpha'],
                'learning_rate': config['training']['learning_rate'],
                'num_epochs': config['training']['num_epochs'],
                'batch_size': config['training']['micro_batch_size'],
                'gradient_accumulation': config['training']['gradient_accumulation_steps'],
                'tpu_type': config['tpu']['type'],
                'num_devices': config['tpu']['num_devices'],
            }
        )
        logger.info(f"✓ Wandb initialized: {config['wandb']['project']}/{config['wandb']['run_name']}")


def create_lora_model(config: dict, device, rank: int):
    """Create model with LoRA adapters - sequential loading to avoid memory issues"""
    model_name = config['model']['name']
    world_size = xr.world_size()
    
    # Load models sequentially, one rank at a time to avoid overwhelming memory
    for i in range(world_size):
        if rank == i:
            logger.info(f"[Rank {rank}/{world_size-1}] Loading model from HuggingFace...", extra={'rank': rank})
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.bfloat16,
                trust_remote_code=True,
                use_cache=False,
                low_cpu_mem_usage=True,
            )
            
            logger.info(f"[Rank {rank}] Model loaded, applying LoRA...", extra={'rank': rank})
            
            # Configure and apply LoRA
            lora_config = LoraConfig(
                r=config['training']['lora']['r'],
                lora_alpha=config['training']['lora']['alpha'],
                target_modules=config['training']['lora']['target_modules'],
                lora_dropout=config['training']['lora']['dropout'],
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            
            model = get_peft_model(model, lora_config)
            
            if rank == 0:
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in model.parameters())
                logger.info(f"Trainable: {trainable_params:,}/{total_params:,} ({100*trainable_params/total_params:.2f}%)", extra={'rank': rank})
            
            # Enable gradient checkpointing
            if config['training']['gradient_checkpointing']:
                model.gradient_checkpointing_enable()
                model.enable_input_require_grads()
            
            logger.info(f"[Rank {rank}] Moving to TPU...", extra={'rank': rank})
            model.eval()  # Eval mode for faster transfer
            model = model.to(device)
            model.train()  # Back to train mode
            
            logger.info(f"✓ [Rank {rank}] Ready on TPU", extra={'rank': rank})
        
        # All ranks wait for current rank to finish
        xm.rendezvous(f"model_load_rank_{i}")
    
    if rank == 0:
        logger.info("✓ All 8 TPU devices ready for training!", extra={'rank': rank})
    
    return model


def train_loop(rank, config):
    """Main training loop for each TPU core"""
    
    # Get TPU device
    device = xm.xla_device()
    
    # Log device info
    world_size = xr.world_size()
    if rank == 0:
        logger.info(f"=" * 70, extra={'rank': rank})
        logger.info(f"TPU Training Started", extra={'rank': rank})
        logger.info(f"World Size: {world_size} device(s)", extra={'rank': rank})
        logger.info(f"=" * 70, extra={'rank': rank})
    
    # Set random seed for reproducibility
    set_seed(config['dataset']['random_seed'] + rank)
    
    # Log rank information
    logger.info(f"Starting training on rank {rank}, device: {device}", extra={'rank': rank})
    
    # Initialize wandb (only on rank 0)
    setup_wandb(config, rank)
    
    # Load tokenizer
    logger.info("Loading tokenizer...", extra={'rank': rank})
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['name'],
        trust_remote_code=True,
    )
    
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    logger.info(f"✓ Tokenizer loaded, vocab size: {len(tokenizer)}", extra={'rank': rank})
    
    # Create model with LoRA and FSDP sharding
    logger.info("Creating model with LoRA...", extra={'rank': rank})
    model = create_lora_model(config, device, rank)
    logger.info("✓ Model creation complete", extra={'rank': rank})
    
    # Load dataset
    dataset_path = Path(config['dataset']['output_dir']) / "all_data.jsonl"
    logger.info(f"Loading dataset from {dataset_path}", extra={'rank': rank})
    
    dataset = HyperswitchDataset(
        data_path=str(dataset_path),
        tokenizer=tokenizer,
        max_length=config['dataset']['max_tokens'],
    )
    
    # Split dataset into train/val
    train_size = int(config['dataset']['train_split'] * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config['dataset']['random_seed'])
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}", extra={'rank': rank})
    
    # Create data loaders
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=xr.world_size(),
        rank=rank,
        shuffle=True,
        seed=config['dataset']['random_seed'],
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['micro_batch_size'],
        sampler=train_sampler,
        num_workers=4,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['eval_batch_size'],
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )
    
    # Wrap loaders for TPU
    train_loader = pl.MpDeviceLoader(train_loader, device)
    val_loader = pl.MpDeviceLoader(val_loader, device)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    # Calculate total training steps
    num_epochs = config['training']['num_epochs']
    gradient_accumulation_steps = config['training']['gradient_accumulation_steps']
    steps_per_epoch = len(train_loader) // gradient_accumulation_steps
    total_steps = steps_per_epoch * num_epochs
    
    logger.info(f"Training for {num_epochs} epochs, {total_steps} total steps", extra={'rank': rank})
    
    # Setup learning rate scheduler
    lr_scheduler = get_scheduler(
        name=config['training']['lr_scheduler'],
        optimizer=optimizer,
        num_warmup_steps=int(config['training']['warmup_ratio'] * total_steps),
        num_training_steps=total_steps,
    )
    
    # Training state
    global_step = 0
    best_val_loss = float('inf')
    
    logger.info("=" * 70, extra={'rank': rank})
    logger.info("🚀 Starting Training", extra={'rank': rank})
    logger.info("=" * 70, extra={'rank': rank})
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        
        train_sampler.set_epoch(epoch)
        
        # Add progress bar (only on rank 0)
        if rank == 0:
            pbar = tqdm(
                total=steps_per_epoch,  # Total optimizer steps (after gradient accumulation)
                desc=f"Epoch {epoch+1}/{num_epochs}",
                ncols=120,
            )
        
        for step, batch in enumerate(train_loader):
            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
            )
            
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
            
            epoch_loss += loss.item() * gradient_accumulation_steps
            
            # Gradient accumulation
            if (step + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                if config['training']['max_grad_norm'] > 0:
                    xm.reduce_gradients(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config['training']['max_grad_norm']
                    )
                
                # Optimizer step
                xm.optimizer_step(optimizer)
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # Mark step to execute pending operations
                xm.mark_step()
                
                global_step += 1
                epoch_steps += 1
                
                # Update progress bar
                if rank == 0:
                    pbar.update(1)
                
                # Logging
                if global_step % config['training']['logging_steps'] == 0:
                    avg_loss = epoch_loss / epoch_steps
                    perplexity = math.exp(min(avg_loss, 20))  # Cap to prevent overflow
                    lr = optimizer.param_groups[0]['lr']
                    
                    if rank == 0:
                        # Update progress bar postfix
                        pbar.set_postfix({
                            'loss': f'{avg_loss:.4f}',
                            'ppl': f'{perplexity:.2f}',
                            'lr': f'{lr:.2e}',
                            'global_step': f'{global_step}/{total_steps}'
                        })
                        
                        logger.info(
                            f"Epoch {epoch+1}/{num_epochs} | Step {global_step}/{total_steps} | "
                            f"Loss: {avg_loss:.4f} | PPL: {perplexity:.2f} | LR: {lr:.2e}",
                            extra={'rank': rank}
                        )
                        
                        if config['wandb']['enabled']:
                            wandb.log({
                                'train/loss': avg_loss,
                                'train/perplexity': perplexity,
                                'train/learning_rate': lr,
                                'train/epoch': epoch + 1,
                            }, step=global_step)
                
                # Evaluation
                if global_step % config['training']['eval_steps'] == 0:
                    val_loss, val_ppl = evaluate(model, val_loader, device, rank)
                    
                    if rank == 0:
                        logger.info(f"Validation Loss: {val_loss:.4f} | PPL: {val_ppl:.2f}", extra={'rank': rank})
                        
                        if config['wandb']['enabled']:
                            wandb.log({
                                'val/loss': val_loss,
                                'val/perplexity': val_ppl,
                            }, step=global_step)
                        
                        # Save best model
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            save_checkpoint(model, tokenizer, config, global_step, "best")
                    
                    model.train()
                
                # Save checkpoint
                if global_step % config['training']['save_steps'] == 0 and rank == 0:
                    save_checkpoint(model, tokenizer, config, global_step, f"step_{global_step}")
        
        # Close progress bar
        if rank == 0:
            pbar.close()
        
        # End of epoch
        avg_epoch_loss = epoch_loss / epoch_steps
        
        if rank == 0:
            logger.info("=" * 70, extra={'rank': rank})
            logger.info(f"Epoch {epoch+1}/{num_epochs} Complete | Avg Loss: {avg_epoch_loss:.4f}", extra={'rank': rank})
            logger.info("=" * 70, extra={'rank': rank})
    
    # Final evaluation
    final_val_loss, final_val_ppl = evaluate(model, val_loader, device, rank)
    
    if rank == 0:
        logger.info(f"Final Validation Loss: {final_val_loss:.4f} | PPL: {final_val_ppl:.2f}", extra={'rank': rank})
        
        if config['wandb']['enabled']:
            wandb.log({
                'val/final_loss': final_val_loss,
                'val/final_perplexity': final_val_ppl,
            }, step=global_step)
        
        # Save final model
        save_checkpoint(model, tokenizer, config, global_step, "final")
        
        logger.info("=" * 70, extra={'rank': rank})
        logger.info("✅ Training Complete!", extra={'rank': rank})
        logger.info("=" * 70, extra={'rank': rank})
        
        if config['wandb']['enabled']:
            wandb.finish()


def evaluate(model, val_loader, device, rank):
    """Evaluation loop"""
    model.eval()
    total_loss = 0.0
    total_steps = 0
    
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
            )
            
            total_loss += outputs.loss.item()
            total_steps += 1
    
    avg_loss = total_loss / total_steps if total_steps > 0 else 0.0
    
    # Reduce across all TPU cores
    avg_loss_tensor = torch.tensor(avg_loss, device=device)
    avg_loss = xm.mesh_reduce('eval_loss', avg_loss_tensor, lambda x: sum(x) / len(x))
    
    avg_loss_value = avg_loss.item() if torch.is_tensor(avg_loss) else avg_loss
    perplexity = math.exp(min(avg_loss_value, 20))  # Cap to prevent overflow
    
    return avg_loss_value, perplexity


def save_checkpoint(model, tokenizer, config, step, name):
    """Save LoRA adapter checkpoint without moving model between devices"""
    import gc
    import shutil
    import json
    from datetime import datetime
    import torch
    from safetensors.torch import save_file
    
    output_dir = Path("checkpoints") / f"checkpoint-{name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving checkpoint to {output_dir}...", extra={'rank': 0})
    
    # Synchronize all XLA operations before saving
    xm.mark_step()
    xm.wait_device_ops()
    
    # Extract only the trainable LoRA parameters and move to CPU
    # This avoids moving the entire model which causes crashes
    trainable_state_dict = {}
    for name_param, param in model.named_parameters():
        if param.requires_grad:
            # Move only trainable params to CPU
            trainable_state_dict[name_param] = param.detach().cpu()
    
    # Save adapter weights using safetensors
    adapter_file = output_dir / "adapter_model.safetensors"
    save_file(trainable_state_dict, str(adapter_file))
    
    # Save adapter config
    adapter_config = {
        "base_model_name_or_path": config['model']['name'],
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": config['training']['lora']['r'],
        "lora_alpha": config['training']['lora']['alpha'],
        "lora_dropout": config['training']['lora']['dropout'],
        "target_modules": config['training']['lora']['target_modules'],
        "bias": "none",
        "inference_mode": False,
    }
    
    with open(output_dir / "adapter_config.json", 'w') as f:
        json.dump(adapter_config, f, indent=2)
    
    # Clean up tensors
    del trainable_state_dict
    gc.collect()
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    # Save training info
    training_info = {
        "step": step,
        "checkpoint_name": name,
        "timestamp": datetime.now().isoformat(),
        "model_name": config['model']['name'],
        "lora_r": config['training']['lora']['r'],
        "lora_alpha": config['training']['lora']['alpha'],
        "learning_rate": config['training']['learning_rate'],
        "effective_batch_size": config['training']['micro_batch_size'] * config['training']['gradient_accumulation_steps'] * config['tpu']['num_devices'],
    }
    
    with open(output_dir / "training_info.json", 'w') as f:
        json.dump(training_info, f, indent=2)
    
    # Save full training config
    with open(output_dir / "training_config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Clean up old step checkpoints (keep only last 3)
    if name.startswith("step_"):
        checkpoints_dir = Path("checkpoints")
        step_checkpoints = sorted(
            [d for d in checkpoints_dir.glob("checkpoint-step_*")],
            key=lambda x: int(x.name.split("_")[-1])
        )
        # Keep only the last 3 step checkpoints
        for old_ckpt in step_checkpoints[:-3]:
            logger.info(f"Removing old checkpoint: {old_ckpt.name}", extra={'rank': 0})
            shutil.rmtree(old_ckpt)
    
    # Synchronize after saving
    xm.mark_step()
    
    logger.info(f"✓ Checkpoint saved to {output_dir}", extra={'rank': 0})

    # Log to wandb
    if config['wandb']['enabled'] and config['wandb']['log_model']:
        artifact = wandb.Artifact(
            name=f"model-{name}",
            type="model",
            description=f"LoRA checkpoint at step {step}",
        )
        artifact.add_dir(str(output_dir))
        wandb.log_artifact(artifact)


def _mp_fn(rank, config):
    """Multi-process function for TPU training"""
    try:
        train_loop(rank, config)
    except Exception as e:
        logger.error(f"Error in rank {rank}: {e}", extra={'rank': rank}, exc_info=True)
        raise


if __name__ == "__main__":
    # Load configuration
    config = load_config("config.yaml")
    
    logger.info("=" * 70)
    logger.info("TPU Training Configuration")
    logger.info("=" * 70)
    logger.info(f"Model: {config['model']['name']}")
    logger.info(f"TPU Type: {config['tpu']['type']}")
    logger.info(f"LoRA r={config['training']['lora']['r']}, alpha={config['training']['lora']['alpha']}")
    logger.info(f"Batch Size: {config['training']['micro_batch_size']}")
    logger.info(f"Gradient Accumulation: {config['training']['gradient_accumulation_steps']}")
    logger.info(f"Learning Rate: {config['training']['learning_rate']}")
    logger.info(f"Epochs: {config['training']['num_epochs']}")
    logger.info("=" * 70)
    logger.info("Note: Using all available TPU devices (auto-detected)")
    logger.info("=" * 70)
    
    # Launch multi-process training on TPUs
    # Use None to automatically use all available devices
    xmp.spawn(_mp_fn, args=(config,), nprocs=None)
