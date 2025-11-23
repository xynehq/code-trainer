import os
import math
import json
import time
import shutil
import argparse
import yaml
import traceback
from pathlib import Path
from datetime import datetime
import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


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


def load_config(config_path="training_config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def inspect_model_modules(model, accelerator):
    accelerator.print("\n=== Model Architecture Inspection (Sample) ===")
    for name, _ in model.named_modules():
        if "layers.10.self_attn" in name or "layers.10.mlp" in name:
            if "weight" not in name:
                accelerator.print(name)
    accelerator.print("=" * 50 + "\n")


def log_expert_usage(model, outputs, step, accelerator, log_file_path=None):
    if not (hasattr(outputs, "router_logits") and outputs.router_logits is not None):
        return
    num_layers = len(outputs.router_logits)
    layer_indices_to_log = [0, num_layers // 2]
    num_experts = getattr(model.config, "n_routed_experts", None) or getattr(
        model.config, "num_experts", None
    )
    if num_experts is None:
        return
    
    expert_usage_data = {}
    
    for layer_idx in layer_indices_to_log:
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
        
        if accelerator.is_main_process and (step % 100 == 0):
            accelerator.print(f"--- Step {step} | Layer {layer_idx} Expert Usage ---")
            accelerator.print(f"Total tokens processed: {total_usage}")
            
            for i in range(num_experts):
                count = int(full_usage[i])
                percent = float(expert_usage_percent[i])
                status = ""
                if count == 0:
                    status = " [DEAD - NO USAGE]"
                elif percent > 50:
                    status = " [DOMINATING]"
                elif percent < 5:
                    status = " [UNDERUTILIZED]"
                accelerator.print(f"  Expert {i}: {count} times ({percent:.2f}%){status}")
            
            used_experts = (full_usage > 0).sum().item()
            accelerator.print(f"  Active experts: {used_experts}/{num_experts}")
            if used_experts < num_experts:
                dead_experts = [i for i in range(num_experts) if full_usage[i] == 0]
                accelerator.print(f"  Dead experts: {dead_experts}")
            accelerator.print("-" * 50)
    
    if log_file_path is not None and accelerator.is_main_process and (step % 100 == 0):
        log_entry = {
            "step": step,
            "timestamp": time.time(),
            "expert_usage": expert_usage_data
        }
        
        with open(log_file_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")


def get_lora_config(model, accelerator, config):
    """Create LoRA configuration from config file"""
    num_layers = model.config.num_hidden_layers
    lora_cfg = config['lora']
    
    accelerator.print(f"--- Model has {num_layers} transformer layers (0-{num_layers-1}) ---")
    accelerator.print(f"--- LoRA Target Module patterns: {lora_cfg['target_modules']} ---")
    
    return LoraConfig(
        r=lora_cfg['r'],
        lora_alpha=lora_cfg['lora_alpha'],
        target_modules=lora_cfg['target_modules'],
        layers_to_transform=list(range(num_layers)),
        lora_dropout=lora_cfg['lora_dropout'],
        bias=lora_cfg['bias'],
        task_type="CAUSAL_LM",
    )


def format_time(seconds):
    """Format seconds into human-readable time string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"


def evaluate(model, eval_dataloader, accelerator):
    """Run evaluation and return metrics"""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in eval_dataloader:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                # Gather loss across all processes for correct averaging
                loss_gathered = accelerator.gather(loss.unsqueeze(0))
                total_loss += loss_gathered.mean().item() * batch["input_ids"].size(0)
                total_samples += batch["input_ids"].size(0)
    
    # Synchronize total_loss and total_samples across all ranks
    total_loss_tensor = torch.tensor(total_loss, device=model.device)
    total_samples_tensor = torch.tensor(total_samples, device=model.device)
    
    total_loss_tensor = accelerator.reduce(total_loss_tensor)
    total_samples_tensor = accelerator.reduce(total_samples_tensor)
    
    avg_loss = total_loss_tensor.item() / total_samples_tensor.item() if total_samples_tensor.item() > 0 else float('inf')
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    
    model.train()
    return {"eval_loss": avg_loss, "eval_perplexity": perplexity}


def manage_checkpoints(output_dir, checkpoint_tracker, max_checkpoints=5):
    """Keep only the top-k checkpoints based on validation loss"""
    if len(checkpoint_tracker) <= max_checkpoints:
        return
    
    sorted_checkpoints = sorted(checkpoint_tracker.items(), key=lambda x: x[1])
    
    for checkpoint_dir, _ in sorted_checkpoints[max_checkpoints:]:
        checkpoint_path = Path(output_dir) / checkpoint_dir
        if checkpoint_path.exists():
            shutil.rmtree(checkpoint_path)
            print(f"Removed checkpoint: {checkpoint_dir}")
    
    checkpoint_tracker.clear()
    for checkpoint_dir, val_loss in sorted_checkpoints[:max_checkpoints]:
        checkpoint_tracker[checkpoint_dir] = val_loss


def save_best_adapter(output_dir, best_adapter_dir, best_checkpoint_name, accelerator):
    """Copy the best adapter to a dedicated directory"""
    best_checkpoint_path = Path(output_dir) / best_checkpoint_name
    best_adapter_path = Path(best_adapter_dir)
    
    if accelerator.is_main_process:
        if best_adapter_path.exists():
            shutil.rmtree(best_adapter_path)
        
        shutil.copytree(best_checkpoint_path, best_adapter_path)
        accelerator.print(f"\n{'='*60}")
        accelerator.print(f"✅ Best adapter saved to: {best_adapter_dir}")
        accelerator.print(f"   Source checkpoint: {best_checkpoint_name}")
        accelerator.print(f"{'='*60}\n")


def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint directory"""
    checkpoint_dirs = []
    if os.path.exists(output_dir):
        for item in os.listdir(output_dir):
            if item.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, item)):
                try:
                    step = int(item.split("-")[1])
                    checkpoint_dirs.append((step, item))
                except (ValueError, IndexError):
                    continue
    
    if checkpoint_dirs:
        checkpoint_dirs.sort(reverse=True)
        return checkpoint_dirs[0][0], os.path.join(output_dir, checkpoint_dirs[0][1])
    return None, None


def save_training_state(output_dir, global_step, optimizer, scheduler, best_eval_loss, 
                        best_checkpoint_name, checkpoint_tracker, accelerator):
    """Save optimizer, scheduler, and training state"""
    if accelerator.is_main_process:
        state_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
        state_file = os.path.join(state_dir, "training_state.pt")
        
        state = {
            "global_step": global_step,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_eval_loss": best_eval_loss,
            "best_checkpoint_name": best_checkpoint_name,
            "checkpoint_tracker": checkpoint_tracker,
        }
        torch.save(state, state_file)


def load_training_state(checkpoint_dir, optimizer, scheduler, accelerator):
    """Load optimizer, scheduler, and training state"""
    state_file = os.path.join(checkpoint_dir, "training_state.pt")
    
    if not os.path.exists(state_file):
        return None
    
    accelerator.print(f"Loading training state from {state_file}")
    state = torch.load(state_file, map_location="cpu")
    
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    
    return {
        "global_step": state["global_step"],
        "best_eval_loss": state["best_eval_loss"],
        "best_checkpoint_name": state["best_checkpoint_name"],
        "checkpoint_tracker": state["checkpoint_tracker"],
    }


def get_lr_from_log(log_path, step_target):
    """Return the LR value closest to a specific step from the training log."""
    closest_lr = None
    closest_diff = float("inf")
    
    if not os.path.exists(log_path):
        return None
    
    with open(log_path, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                if "learning_rate" in entry and "step" in entry:
                    diff = abs(entry["step"] - step_target)
                    if diff < closest_diff:
                        closest_diff = diff
                        closest_lr = entry["learning_rate"]
            except Exception:
                continue
    
    return closest_lr


def restore_lr_from_logs(optimizer, scheduler, resume_step, train_log_path, 
                         initial_lr, total_updates, accelerator):
    """Restore learning rate and scheduler state from training logs"""
    log_lr = get_lr_from_log(train_log_path, resume_step)
    
    if log_lr is None:
        accelerator.print("⚠️  No LR found in logs; scheduler will start from beginning")
        return False
    
    # Restore the learning rate in optimizer
    for group in optimizer.param_groups:
        group["lr"] = log_lr
    
    accelerator.print(f"✅ Restored LR from logs: {log_lr:.6e} (at step {resume_step})")
    
    # The scheduler will be stepped naturally during training
    # We just need to set the current LR in the optimizer
    # The scheduler's step() will be called after each gradient update
    # So we don't need to manually advance it - just let it continue from current LR
    
    accelerator.print(f"✅ Scheduler will continue from current LR on next step")
    
    return True


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="GLM-4.5-Air QLoRA Training")
    parser.add_argument("--config", type=str, default="training_config.yaml", help="Path to config file")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from specific checkpoint directory")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Extract config sections
    paths = config['paths']
    training_cfg = config['training']
    opt_cfg = config['optimization']
    lora_cfg = config['lora']
    quant_cfg = config['quantization']
    log_cfg = config['logging']
    mem_cfg = config['memory']
    perf_cfg = config['performance']
    
    set_seed(training_cfg['seed'])
    import random
    import numpy as np
    random.seed(training_cfg['seed'])
    np.random.seed(training_cfg['seed'])

    accelerator = Accelerator(gradient_accumulation_steps=opt_cfg['grad_accum_steps'])

    if accelerator.is_main_process:
        os.makedirs(paths['output_dir'], exist_ok=True)
        os.makedirs(os.path.join(paths['output_dir'], "logs"), exist_ok=True)
        
        # Print comprehensive configuration
        accelerator.print("\n" + "="*70)
        accelerator.print("TRAINING CONFIGURATION")
        accelerator.print("="*70)
        
        accelerator.print("\n📁 PATHS:")
        accelerator.print(f"  Config file:      {args.config}")
        accelerator.print(f"  Model:            {paths['model_path']}")
        accelerator.print(f"  Dataset:          {paths['data_path']}")
        accelerator.print(f"  Output dir:       {paths['output_dir']}")
        accelerator.print(f"  Best adapter:     {paths['best_adapter_dir']}")
        
        accelerator.print("\n🎯 TRAINING:")
        accelerator.print(f"  Epochs:           {training_cfg['num_epochs']}")
        accelerator.print(f"  Seed:             {training_cfg['seed']}")
        accelerator.print(f"  Max seq length:   {training_cfg['max_seq_length']}")
        accelerator.print(f"  Train/val split:  {training_cfg['train_val_split']:.0%}/{1-training_cfg['train_val_split']:.0%}")
        
        accelerator.print("\n⚙️  OPTIMIZATION:")
        accelerator.print(f"  Learning rate:    {opt_cfg['learning_rate']:.2e}")
        accelerator.print(f"  Weight decay:     {opt_cfg['weight_decay']}")
        accelerator.print(f"  Max grad norm:    {opt_cfg['max_grad_norm']}")
        accelerator.print(f"  Warmup ratio:     {opt_cfg['warmup_ratio']:.1%} (hardcoded: 8%)")
        accelerator.print(f"  Betas:            {opt_cfg['betas']}")
        accelerator.print(f"  Epsilon:          {opt_cfg['eps']:.2e}")
        
        accelerator.print("\n📦 BATCH & ACCUMULATION:")
        accelerator.print(f"  Micro batch/GPU:  {opt_cfg['micro_batch_per_gpu']}")
        accelerator.print(f"  Grad accum steps: {opt_cfg['grad_accum_steps']}")
        accelerator.print(f"  Num GPUs:         {accelerator.num_processes}")
        effective_batch = opt_cfg['micro_batch_per_gpu'] * opt_cfg['grad_accum_steps'] * accelerator.num_processes
        accelerator.print(f"  Effective batch:  {effective_batch}")
        
        accelerator.print("\n🔧 LORA:")
        accelerator.print(f"  Rank (r):         {lora_cfg['r']}")
        accelerator.print(f"  Alpha:            {lora_cfg['lora_alpha']}")
        accelerator.print(f"  Dropout:          {lora_cfg['lora_dropout']}")
        accelerator.print(f"  Target modules:   {', '.join(lora_cfg['target_modules'])}")
        accelerator.print(f"  Bias:             {lora_cfg['bias']}")
        
        accelerator.print("\n🔢 QUANTIZATION:")
        accelerator.print(f"  Load in 4-bit:    {quant_cfg['load_in_4bit']}")
        accelerator.print(f"  Compute dtype:    {quant_cfg['bnb_4bit_compute_dtype']}")
        accelerator.print(f"  Double quant:     {quant_cfg['bnb_4bit_use_double_quant']}")
        accelerator.print(f"  Quant type:       {quant_cfg['bnb_4bit_quant_type']}")
        
        accelerator.print("\n📊 LOGGING & CHECKPOINTING:")
        accelerator.print(f"  Log interval:     Every {log_cfg['log_interval']} steps")
        accelerator.print(f"  Eval interval:    Every {log_cfg['eval_interval']} steps")
        accelerator.print(f"  Checkpoint int:   Every {log_cfg['checkpoint_interval']} steps")
        accelerator.print(f"  Max checkpoints:  {log_cfg['max_checkpoints']}")
        
        accelerator.print("\n💾 MEMORY:")
        accelerator.print(f"  CPU offload:      {mem_cfg['use_cpu_offload']}")
        accelerator.print(f"  DeepSpeed ZeRO3:  {mem_cfg['use_deepspeed_zero3']}")
        accelerator.print(f"  Grad checkpoint:  {mem_cfg['gradient_checkpointing']} (hardcoded: True)")
        accelerator.print(f"  Use cache:        {mem_cfg['use_cache']} (hardcoded: False)")
        
        accelerator.print("\n⚡ PERFORMANCE:")
        accelerator.print(f"  Allow TF32:       {perf_cfg['allow_tf32']}")
        accelerator.print(f"  Num workers:      {perf_cfg['num_workers']} (hardcoded: 4)")
        accelerator.print(f"  Pin memory:       {perf_cfg['pin_memory']} (hardcoded: True)")
        
        accelerator.print("\n" + "="*70 + "\n")

    if perf_cfg['allow_tf32']:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    accelerator.print("--- Loading Tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(
        paths['model_path'], padding_side="left", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        # Use max sequence length from YAML config (fallback to 4096 if missing)
        max_len = int(training_cfg.get("max_seq_length", 4096))
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_attention_mask=True,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    accelerator.print("--- Loading and Tokenizing Dataset ---")
    with accelerator.main_process_first():
        raw_dataset = load_dataset("json", data_files=paths['data_path'], split="train")
        
        split_dataset = raw_dataset.train_test_split(
            test_size=0.1, 
            seed=42
        )
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        
        accelerator.print(f"Train samples: {len(train_dataset)}")
        accelerator.print(f"Validation samples: {len(eval_dataset)}")
        
        tokenized_train = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names,
            num_proc=min(16, os.cpu_count() or 1),
        )
        
        tokenized_eval = eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=eval_dataset.column_names,
            num_proc=min(16, os.cpu_count() or 1),
        )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    accelerator.print("--- Loading 4‑bit Quantized 100B MoE Model (QLoRA) ---")
    model = AutoModelForCausalLM.from_pretrained(
        paths['model_path'],
        trust_remote_code=True,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
    )

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    if accelerator.is_main_process:
        inspect_model_modules(model, accelerator)

    # We'll create the PEFT model after checking for resume
    lora_config = get_lora_config(model, accelerator, config)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # For multi-GPU training, use num_workers=0 to avoid spawning too many processes
    # Each GPU process will handle its own data loading efficiently
    train_dataloader = DataLoader(
        tokenized_train,
        batch_size=opt_cfg['micro_batch_per_gpu'],
        collate_fn=data_collator,
        shuffle=True,
        num_workers=0,  # Avoid extra processes in multi-GPU setup
        pin_memory=True,
    )
    
    eval_dataloader = DataLoader(
        tokenized_eval,
        batch_size=opt_cfg['micro_batch_per_gpu'],
        collate_fn=data_collator,
        shuffle=False,
        num_workers=0,  # Avoid extra processes in multi-GPU setup
        pin_memory=True,
    )

    # Check for checkpoint resumption
    resume_checkpoint_dir = None
    resume_step = None
    
    if args.resume_from:
        # Resume from specific checkpoint
        resume_checkpoint_dir = args.resume_from
        if not os.path.exists(resume_checkpoint_dir):
            raise ValueError(f"Checkpoint directory not found: {resume_checkpoint_dir}")
        try:
            resume_step = int(os.path.basename(resume_checkpoint_dir).split("-")[1])
        except (ValueError, IndexError):
            raise ValueError(f"Invalid checkpoint directory name: {resume_checkpoint_dir}")
        accelerator.print(f"\n🔄 Resuming from specified checkpoint: {resume_checkpoint_dir}")
    elif args.resume:
        # Auto-detect latest checkpoint
        resume_step, resume_checkpoint_dir = find_latest_checkpoint(paths['output_dir'])
        if resume_checkpoint_dir:
            accelerator.print(f"\n🔄 Auto-detected checkpoint to resume from: {resume_checkpoint_dir}")
        else:
            accelerator.print("\n⚠️  No checkpoint found to resume from. Starting fresh training.")
    
    # Create PEFT model (either fresh or from checkpoint)
    if resume_checkpoint_dir and resume_step is not None:
        accelerator.print(f"Loading model from checkpoint at step {resume_step}...")
        
        # Load LoRA adapter using PeftModel.from_pretrained (proper way)
        from peft import PeftModel
        adapter_config_path = os.path.join(resume_checkpoint_dir, "adapter_config.json")
        
        if not os.path.exists(adapter_config_path):
            accelerator.print(f"⚠️  No adapter found in {resume_checkpoint_dir}")
            accelerator.print("   Creating fresh LoRA adapter instead")
            model = get_peft_model(model, lora_config)
        else:
            # Proper way: use PeftModel.from_pretrained for resume
            # CRITICAL: Set is_trainable=True to enable training mode
            model = PeftModel.from_pretrained(model, resume_checkpoint_dir, is_trainable=True)
            accelerator.print("✅ LoRA adapter loaded from checkpoint (trainable mode enabled)")
    else:
        # Fresh training: create new PEFT model
        model = get_peft_model(model, lora_config)
    
    if accelerator.is_main_process:
        model.print_trainable_parameters()
    
    # NOW create optimizer and scheduler AFTER PEFT model is ready
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, 
        lr=opt_cfg['learning_rate'], 
        weight_decay=opt_cfg['weight_decay'], 
        betas=(0.9, 0.95), 
        eps=1e-8
    )

    # Prepare model, optimizer, and dataloaders with accelerator
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # Create scheduler
    updates_per_epoch = math.ceil(len(train_dataloader) / opt_cfg['grad_accum_steps'])
    total_updates = updates_per_epoch * training_cfg['num_epochs']
    total_steps = len(train_dataloader) * training_cfg['num_epochs']
    warmup_updates = max(30, int(0.08 * total_updates))  # 8% warmup for MoE + 4-bit stability
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_updates, num_training_steps=total_updates
    )
    
    # Load training state (optimizer, scheduler, etc.) - must be outside main_process check
    if resume_checkpoint_dir and resume_step is not None:
        training_state = load_training_state(resume_checkpoint_dir, optimizer, scheduler, accelerator)
        
        if training_state:
            start_step = training_state["global_step"]
            best_eval_loss = training_state["best_eval_loss"]
            best_checkpoint_name = training_state["best_checkpoint_name"]
            checkpoint_tracker = training_state["checkpoint_tracker"]
            accelerator.print(f"✅ Training state loaded - resuming from step {start_step}")
            accelerator.print(f"   Best eval loss so far: {best_eval_loss:.4f}")
            accelerator.print(f"   Best checkpoint: {best_checkpoint_name}")
        else:
            accelerator.print("⚠️  No training_state.pt found - attempting to restore LR from logs")
            start_step = resume_step
            best_eval_loss = float('inf')
            best_checkpoint_name = None
            checkpoint_tracker = {}
            
            # Try to restore learning rate from training logs
            train_log_path_temp = os.path.join(paths['output_dir'], "logs", "training_log.jsonl")
            lr_restored = restore_lr_from_logs(
                optimizer, 
                scheduler, 
                resume_step, 
                train_log_path_temp,
                opt_cfg['learning_rate'],
                total_updates,
                accelerator
            )
            
            if not lr_restored:
                accelerator.print("⚠️  Optimizer and scheduler will start from initial state")
    else:
        start_step = 0
        best_eval_loss = float('inf')
        best_checkpoint_name = None
        checkpoint_tracker = {}
    
    start_time = time.time()
    
    train_log_path = os.path.join(paths['output_dir'], "logs", "training_log.jsonl")
    eval_log_path = os.path.join(paths['output_dir'], "logs", "eval_log.jsonl")
    expert_log_path = os.path.join(paths['output_dir'], "logs", "expert_usage_log.jsonl")
    error_log_path = os.path.join(paths['output_dir'], "logs", "error_log.jsonl")

    accelerator.print("--- Starting QLoRA Training ---")
    if start_step > 0:
        accelerator.print(f"🔄 RESUMING from step {start_step}")
    accelerator.print(f"World size: {accelerator.num_processes}")
    accelerator.print(f"Micro batch per GPU: {opt_cfg['micro_batch_per_gpu']}")
    accelerator.print(f"Gradient accumulation steps: {opt_cfg['grad_accum_steps']}")
    accelerator.print(f"Updates per epoch: {updates_per_epoch}")
    accelerator.print(f"Total steps: {total_steps}")
    accelerator.print(f"Total updates: {total_updates}")
    accelerator.print(f"Train samples: {len(tokenized_train)}")
    accelerator.print(f"Val samples: {len(tokenized_eval)}")
    
    # Initial evaluation (baseline before training) - skip if resuming
    if start_step == 0:
        accelerator.wait_for_everyone()  # Ensure all ranks are synchronized before eval
        accelerator.print("\n" + "="*60)
        accelerator.print("📊 Initial Evaluation (Baseline)")
        accelerator.print("="*60)
        initial_metrics = evaluate(model, eval_dataloader, accelerator)
        accelerator.wait_for_everyone()  # Synchronize after eval completes
        
        initial_loss = initial_metrics["eval_loss"]
        initial_ppl = initial_metrics["eval_perplexity"]
        accelerator.print(f"Initial Validation Loss: {initial_loss:.4f}")
        accelerator.print(f"Initial Perplexity: {initial_ppl:.2f}")
        accelerator.print("="*60 + "\n")
        
        # Log initial evaluation
        if accelerator.is_main_process:
            initial_eval_entry = {
                "step": 0,
                "eval_loss": initial_loss,
                "eval_perplexity": initial_ppl,
                "timestamp": time.time(),
                "note": "baseline_before_training"
            }
            with open(eval_log_path, "a") as f:
                f.write(json.dumps(initial_eval_entry) + "\n")
    else:
        accelerator.print("\n⏭️  Skipping initial evaluation (resuming from checkpoint)\n")

    global_step = start_step
    
    # Calculate starting epoch and step within epoch
    steps_per_epoch = len(train_dataloader)
    start_epoch = global_step // steps_per_epoch
    steps_to_skip_in_first_epoch = global_step % steps_per_epoch
    
    for epoch in range(start_epoch, training_cfg['num_epochs']):
        model.train()
        
        # Skip already completed steps in the first epoch when resuming
        dataloader_iterator = iter(train_dataloader)
        if epoch == start_epoch and steps_to_skip_in_first_epoch > 0:
            accelerator.print(f"Skipping {steps_to_skip_in_first_epoch} steps in epoch {epoch}...")
            for _ in range(steps_to_skip_in_first_epoch):
                try:
                    next(dataloader_iterator)
                except StopIteration:
                    break
        
        for step, batch in enumerate(dataloader_iterator):
            with accelerator.accumulate(model):
                outputs = model(**batch, output_router_logits=True)
                loss = outputs.loss

                if torch.isnan(loss):
                    error_msg = f"NaN loss encountered at step {global_step}, epoch {epoch}"
                    accelerator.print(f"❌ {error_msg}")
                    if accelerator.is_main_process:
                        log_error(
                            error_log_path,
                            error_type="NaNLoss",
                            error_message=error_msg,
                            step=global_step,
                            epoch=epoch,
                            traceback_str=None
                        )
                    raise ValueError(error_msg)

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    # Only clip, step the optimizer and step the scheduler when we actually sync gradients
                    grad_norm = accelerator.clip_grad_norm_(trainable_params, opt_cfg['max_grad_norm'])
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                else:
                    grad_norm = None

            # Logging
            if accelerator.is_main_process and (global_step % log_cfg['log_interval'] == 0):
                elapsed_time = time.time() - start_time
                # Calculate steps completed since start (or resume)
                steps_completed = global_step - start_step
                steps_per_sec = steps_completed / elapsed_time if elapsed_time > 0 else 0
                remaining_steps = total_steps - global_step
                eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                
                aux_loss = None
                for attr in ["router_aux_loss", "aux_loss", "load_balancing_loss"]:
                    if hasattr(outputs, attr) and getattr(outputs, attr) is not None:
                        aux_loss = getattr(outputs, attr)
                        break
                
                current_lr = scheduler.get_last_lr()[0]
                
                msg = (
                    f"Epoch {epoch} | Step {global_step}/{total_steps} ({100*global_step/total_steps:.1f}%) | "
                    f"Loss {loss.item():.4f}"
                )
                if aux_loss is not None:
                    msg += f" | Aux {aux_loss.item():.4f}"
                if grad_norm is not None:
                    msg += f" | GradNorm {grad_norm:.4f}"
                msg += f" | LR {current_lr:.2e} | {steps_per_sec:.2f} steps/s | ETA {format_time(eta_seconds)}"
                
                accelerator.print(msg)
                
                # Log to file - match the exact format from previous logs
                # Convert grad_norm to float if it's a Tensor
                grad_norm_value = float(grad_norm) if grad_norm is not None else None
                
                log_entry = {
                    "epoch": epoch,
                    "step": global_step,
                    "loss": loss.item(),
                    "aux_loss": aux_loss.item() if aux_loss is not None else None,
                    "grad_norm": grad_norm_value,
                    "learning_rate": current_lr,
                    "steps_per_sec": steps_per_sec,
                    "eta_seconds": eta_seconds,
                    "timestamp": time.time()
                }
                with open(train_log_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

            # Expert usage logging
            log_expert_usage(model, outputs, global_step, accelerator, expert_log_path)

            # Validation - skip if we just resumed from this exact step
            should_eval = (global_step > 0 and 
                          global_step % log_cfg['eval_interval'] == 0 and 
                          global_step != start_step)
            
            if should_eval:
                accelerator.wait_for_everyone()  # Sync before eval
                accelerator.print(f"\n{'='*60}")
                accelerator.print(f"Running validation at step {global_step}...")
                eval_metrics = evaluate(model, eval_dataloader, accelerator)
                accelerator.wait_for_everyone()  # Sync after eval
                
                eval_loss = eval_metrics["eval_loss"]
                eval_ppl = eval_metrics["eval_perplexity"]
                
                accelerator.print(f"Validation Loss: {eval_loss:.4f} | Perplexity: {eval_ppl:.2f}")
                accelerator.print(f"{'='*60}\n")
                
                # Log to file
                eval_log_entry = {
                    "step": global_step,
                    "eval_loss": eval_loss,
                    "eval_perplexity": eval_ppl,
                    "timestamp": time.time()
                }
                if accelerator.is_main_process:
                    with open(eval_log_path, "a") as f:
                        f.write(json.dumps(eval_log_entry) + "\n")
                
                # Track best model
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    best_checkpoint_name = f"checkpoint-{global_step}"
                    accelerator.print(f"🎯 New best validation loss: {best_eval_loss:.4f}")

            # Checkpointing - skip if we just resumed from this exact step
            should_checkpoint = (global_step > 0 and 
                               global_step % log_cfg['checkpoint_interval'] == 0 and 
                               global_step != start_step)
            
            if should_checkpoint:
                accelerator.wait_for_everyone()
                checkpoint_name = f"checkpoint-{global_step}"
                save_dir = os.path.join(paths['output_dir'], checkpoint_name)
                
                if accelerator.is_main_process:
                    accelerator.unwrap_model(model).save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)
                    
                    # Run validation for this checkpoint if not already done
                    if global_step % log_cfg['eval_interval'] != 0:
                        # Not an eval step, need to run eval for checkpoint
                        eval_metrics = evaluate(model, eval_dataloader, accelerator)
                        eval_loss = eval_metrics["eval_loss"]
                    else:
                        # We just ran eval in the validation block above
                        eval_loss = eval_metrics["eval_loss"]
                    
                    checkpoint_tracker[checkpoint_name] = eval_loss
                    
                    # Update best checkpoint if needed
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        best_checkpoint_name = checkpoint_name
                    
                    # Save training state (optimizer, scheduler, etc.) for resumption
                    save_training_state(
                        paths['output_dir'], 
                        global_step, 
                        optimizer, 
                        scheduler, 
                        best_eval_loss,
                        best_checkpoint_name, 
                        checkpoint_tracker, 
                        accelerator
                    )
                    
                    accelerator.print(f"✅ Checkpoint saved: {checkpoint_name} (val_loss: {eval_loss:.4f})")
                    
                    # Manage checkpoints
                    manage_checkpoints(paths['output_dir'], checkpoint_tracker, log_cfg['max_checkpoints'])

            global_step += 1

    # Final evaluation and save best adapter
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        accelerator.print(f"\n{'='*60}")
        accelerator.print("📊 Running Final Evaluation after training completion")
        accelerator.print("="*60)
        final_metrics = evaluate(model, eval_dataloader, accelerator)
        final_loss = final_metrics["eval_loss"]
        final_ppl = final_metrics["eval_perplexity"]
        accelerator.print(f"Final Validation Loss: {final_loss:.4f}")
        accelerator.print(f"Final Perplexity: {final_ppl:.2f}")
        accelerator.print("="*60 + "\n")

        # Log final evaluation
        final_eval_entry = {
            "step": global_step,
            "eval_loss": final_loss,
            "eval_perplexity": final_ppl,
            "timestamp": time.time(),
            "note": "final_evaluation"
        }
        with open(eval_log_path, "a") as f:
            f.write(json.dumps(final_eval_entry) + "\n")

        accelerator.print(f"\n{'='*60}")
        accelerator.print("Training completed!")
        accelerator.print(f"Total time: {format_time(time.time() - start_time)}")
        accelerator.print(f"Best validation loss: {best_eval_loss:.4f}")
        accelerator.print(f"Best checkpoint: {best_checkpoint_name}")
        accelerator.print(f"{'='*60}\n")
        
        # Save best adapter
        if best_checkpoint_name:
            save_best_adapter(paths['output_dir'], paths['best_adapter_dir'], best_checkpoint_name, accelerator)
        
        # Save final model state
        final_save_dir = os.path.join(paths['output_dir'], f"final-checkpoint-{global_step}")
        accelerator.unwrap_model(model).save_pretrained(final_save_dir)
        tokenizer.save_pretrained(final_save_dir)
        accelerator.print(f"✅ Final model state for step {global_step} saved to: {final_save_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Log the fatal error - use absolute path from script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        error_log_path = os.path.join(script_dir, "glm45-air-cpt-qlora", "logs", "error_log.jsonl")
        os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
        
        log_error(
            error_log_path,
            error_type=type(e).__name__,
            error_message=str(e),
            step=None,
            epoch=None,
            traceback_str=traceback.format_exc()
        )
        
        print(f"\n{'='*60}")
        print(f"❌ FATAL ERROR: Training crashed!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"Error logged to: {error_log_path}")
        print(f"{'='*60}\n")
        print(f"Full traceback:\n{traceback.format_exc()}")
        
        raise
