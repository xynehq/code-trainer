import os
import sys
import logging
import math
from pathlib import Path
import yaml
import torch
import warnings
import json
import signal
import atexit
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    TrainerCallback,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer
from datasets import load_dataset

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
warnings.filterwarnings("ignore", message="`torch_dtype` is deprecated")
warnings.filterwarnings("ignore", message=".*Gradient accumulation steps mismatch.*")
warnings.filterwarnings("ignore", message=".*tokenizer has new PAD/BOS/EOS tokens.*")

import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("accelerate.accelerator").setLevel(logging.ERROR)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from YAML file"""
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_training_config():
    """Get training configuration from config.yaml"""
    yaml_config = load_config()
    
    training_config = yaml_config.get('training', {})
    model_config = yaml_config.get('model', {})
    dataset_config = yaml_config.get('dataset', {})
    
    return {
        'base_model': model_config.get('name', 'Qwen/Qwen2.5-Coder-32B-Instruct'),
        'output_base_dir': './outputs',
        'run_name': f"hyperswitch-{model_config.get('name', 'qwen').split('/')[-1].lower()}-lora",
        
        'lora_r': training_config.get('lora', {}).get('r', 64),
        'lora_alpha': training_config.get('lora', {}).get('alpha', 128),
        'lora_dropout': training_config.get('lora', {}).get('dropout', 0.05),
        'lora_target_modules': training_config.get('lora', {}).get('target_modules', 
            ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']),
        
        'num_epochs': training_config.get('num_epochs', 5),
        'micro_batch_size': training_config.get('micro_batch_size', 2),
        'gradient_accumulation_steps': training_config.get('gradient_accumulation_steps', 8),
        'eval_batch_size': training_config.get('eval_batch_size', 2),
        'learning_rate': training_config.get('learning_rate', 0.00002),
        'lr_scheduler': training_config.get('lr_scheduler', 'cosine'),
        'warmup_ratio': training_config.get('warmup_ratio', 0.05),
        'weight_decay': training_config.get('weight_decay', 0.0),
        'max_grad_norm': training_config.get('max_grad_norm', 0.5),
        
        'bf16': training_config.get('bf16', True),
        'fp16': training_config.get('fp16', False),
        'tf32': training_config.get('tf32', True),
        
        'logging_steps': training_config.get('logging_steps', 10),
        'save_steps': training_config.get('save_steps', 50),
        'eval_steps': training_config.get('eval_steps', 25),
        
        'sequence_len': dataset_config.get('max_tokens', 4096),
        'sample_packing': training_config.get('sample_packing', False),
        
        'train_split': dataset_config.get('train_split', 0.90),
        'val_split': dataset_config.get('val_split', 0.10),
        'random_seed': dataset_config.get('random_seed', 42),
        
        'gradient_checkpointing': training_config.get('gradient_checkpointing', True),
        
        'seed': dataset_config.get('random_seed', 42),
    }


TRAINING_CONFIG = get_training_config()


def save_training_info(checkpoint_dir, config, train_dataset, eval_dataset, trainer, timestamp):
    """Save comprehensive training information to checkpoint directory"""
    import subprocess
    
    try:
        gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader']).decode().strip().split('\n')
        gpu_model = gpu_info[0] if gpu_info else "Unknown"
        num_gpus = len(gpu_info)
    except:
        gpu_model = "Unknown"
        num_gpus = torch.cuda.device_count()
    

    try:
        import deepspeed
        deepspeed_version = deepspeed.__version__
    except:
        deepspeed_version = "unknown"
    
    try:
        import flash_attn
        flash_attn_version = flash_attn.__version__
    except:
        flash_attn_version = "unknown"
    
    final_step = trainer.state.global_step
    final_epoch = trainer.state.epoch
    log_history = trainer.state.log_history
    
    final_train_loss = None
    final_eval_loss = None
    initial_loss = None
    final_accuracy = None
    initial_accuracy = None
    
    for entry in log_history:
        if 'loss' in entry and initial_loss is None:
            initial_loss = entry['loss']
            initial_accuracy = entry.get('mean_token_accuracy', None)
        if 'loss' in entry:
            final_train_loss = entry['loss']
            final_accuracy = entry.get('mean_token_accuracy', None)
        if 'eval_loss' in entry:
            final_eval_loss = entry['eval_loss']
    
    training_info = {
        "training_metadata": {
            "timestamp": timestamp,
            "training_date": datetime.now().strftime("%Y-%m-%d"),
            "training_time": datetime.now().strftime("%H:%M:%S"),
            "final_epoch": final_epoch,
            "total_steps": final_step,
            "status": "completed"
        },
        
        "model_config": {
            "base_model": config['base_model'],
            "model_type": "causal_lm",
            "architecture": "Qwen2ForCausalLM"
        },
        
        "lora_config": {
            "r": config['lora_r'],
            "lora_alpha": config['lora_alpha'],
            "lora_dropout": config['lora_dropout'],
            "target_modules": config['lora_target_modules'],
        },
        
        "training_config": {
            "num_epochs": config['num_epochs'],
            "per_device_train_batch_size": config['micro_batch_size'],
            "per_device_eval_batch_size": config['eval_batch_size'],
            "gradient_accumulation_steps": config['gradient_accumulation_steps'],
            "effective_batch_size": config['micro_batch_size'] * config['gradient_accumulation_steps'] * num_gpus,
            "learning_rate": config['learning_rate'],
            "lr_scheduler_type": config['lr_scheduler'],
            "warmup_ratio": config.get('warmup_ratio', 0.1),
            "weight_decay": config['weight_decay'],
            "max_grad_norm": config['max_grad_norm'],
            "bf16": config['bf16'],
            "gradient_checkpointing": config['gradient_checkpointing'],
            "optim": "adamw_torch",
            "logging_steps": config['logging_steps'],
            "save_steps": config['save_steps'],
            "eval_steps": config['eval_steps']
        },
        
        "dataset_info": {
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset),
            "max_seq_length": config['sequence_len'],
            "sample_packing": config['sample_packing']
        },
        
        "hardware_config": {
            "num_gpus": num_gpus,
            "gpu_model": gpu_model,
            "distributed_strategy": "DeepSpeed ZeRO-2",
            "flash_attention": flash_attn_version
        },
        
        "performance_metrics": {
            "final_train_loss": final_train_loss,
            "final_eval_loss": final_eval_loss,
            "final_train_perplexity": math.exp(final_train_loss) if final_train_loss is not None else None,
            "final_eval_perplexity": math.exp(final_eval_loss) if final_eval_loss is not None else None,
            "final_token_accuracy": final_accuracy,
            "initial_loss": initial_loss,
            "initial_perplexity": math.exp(initial_loss) if initial_loss is not None else None,
            "initial_accuracy": initial_accuracy
        },
        
        "framework_versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "trl": trl.__version__,
            "deepspeed": deepspeed_version,
            "flash_attn": flash_attn_version,
            "python": sys.version.split()[0]
        },
        
        "special_features": {
            "flash_attention_2": True,
            "gradient_checkpointing": config['gradient_checkpointing'],
            "bf16_training": config['bf16'],
            "sample_packing": config['sample_packing'],
            "deepspeed_zero2": True,
            "distributed_training": num_gpus > 1
        }
    }
    
    info_path = os.path.join(checkpoint_dir, 'training_info.json')
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2)
    
    logger.info(f"  ✓ Saved training info: {info_path}")
    return training_info


class PerplexityCallback(TrainerCallback):
    """Callback to log perplexity to TensorBoard"""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Calculate and log perplexity from loss"""
        if logs is not None and state.is_world_process_zero:
            import math
            from torch.utils.tensorboard import SummaryWriter
            
            if not hasattr(self, 'tb_writer'):
                self.tb_writer = SummaryWriter(log_dir=args.logging_dir)
            
            if 'loss' in logs:
                try:
                    train_ppl = math.exp(logs['loss'])
                    logs['train/perplexity'] = train_ppl
                    self.tb_writer.add_scalar('train/perplexity', train_ppl, state.global_step)
                except (ValueError, OverflowError):
                    logs['train/perplexity'] = float('inf')
            
            if 'eval_loss' in logs:
                try:
                    eval_ppl = math.exp(logs['eval_loss'])
                    logs['eval/perplexity'] = eval_ppl
                    self.tb_writer.add_scalar('eval/perplexity', eval_ppl, state.global_step)
                except (ValueError, OverflowError):
                    logs['eval/perplexity'] = float('inf')
            
            self.tb_writer.flush()
    
    def on_train_end(self, args, state, control, **kwargs):
        """Close tensorboard writer at end of training"""
        if hasattr(self, 'tb_writer'):
            self.tb_writer.close()


class TrainingInfoCallback(TrainerCallback):
    """Callback to save training_info.json after each checkpoint"""
    
    def __init__(self, config, train_dataset, eval_dataset, timestamp):
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.timestamp = timestamp
        self.trainer_ref = None
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Store trainer reference when training begins"""
        pass
    
    def on_save(self, args, state, control, model=None, **kwargs):
        """Called after saving a checkpoint"""
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        
        if checkpoint_dir.exists():
            try:
                logger.info(f"  💾 Saving training_info.json to {checkpoint_dir.name}...")
                
                class TrainerState:
                    def __init__(self, state):
                        self.state = state
                
                trainer_obj = TrainerState(state)
                
                save_training_info(
                    str(checkpoint_dir),
                    self.config,
                    self.train_dataset,
                    self.eval_dataset,
                    trainer_obj,
                    self.timestamp
                )
            except Exception as e:
                logger.warning(f"  ⚠️  Failed to save training_info.json: {e}")



def main():
    global TRAINING_CONFIG
    
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    if local_rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.environ['TRAINING_TIMESTAMP'] = timestamp
    else:
        import time
        for _ in range(30):  
            if 'TRAINING_TIMESTAMP' in os.environ:
                timestamp = os.environ['TRAINING_TIMESTAMP']
                break
            time.sleep(0.1)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    run_name = TRAINING_CONFIG['run_name']
    output_dir = os.path.join(TRAINING_CONFIG['output_base_dir'], f"{run_name}_{timestamp}")

    
    logger.info("=" * 80)
    logger.info("Hyperswitch CPT Training - Direct Implementation")
    logger.info("Using: Transformers + TRL + PEFT + DeepSpeed")
    logger.info("=" * 80)
    
    model_name = TRAINING_CONFIG['base_model']
    
    logger.info(f"\nModel: {model_name}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Training run: {timestamp}")
    
    logger.info("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"  Vocab size: {len(tokenizer)}")
    logger.info(f"  Pad token: {tokenizer.pad_token}")
    
    logger.info("\n[2/5] Loading datasets...")
    
    yaml_config = load_config()
    dataset_dir = yaml_config.get('dataset', {}).get('output_dir', 'dataset')
    dataset_file = f'{dataset_dir}/all_data.jsonl'
    
    all_data = load_dataset('json', data_files=dataset_file, split='train')
    logger.info(f"  Total samples: {len(all_data):,}")
    
    test_size = TRAINING_CONFIG['val_split']  
    seed = TRAINING_CONFIG['random_seed']
    
    split_dataset = all_data.train_test_split(test_size=test_size, seed=seed)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    logger.info(f"  Training samples ({TRAINING_CONFIG['train_split']*100:.0f}%): {len(train_dataset):,}")
    logger.info(f"  Validation samples ({TRAINING_CONFIG['val_split']*100:.0f}%): {len(eval_dataset):,}")
    
    logger.info("[3/5] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        TRAINING_CONFIG['base_model'],
        dtype=torch.bfloat16,  
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  
    )
    model.config.use_cache = False
    
    logger.info(f"  Model loaded: {model.dtype}")
    logger.info(f"  Parameters: {model.num_parameters() / 1e9:.2f}B")
    
    logger.info("\n[4/5] Setting up LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=TRAINING_CONFIG['lora_r'],
        lora_alpha=TRAINING_CONFIG['lora_alpha'],
        lora_dropout=TRAINING_CONFIG['lora_dropout'],
        target_modules=TRAINING_CONFIG['lora_target_modules'],
        bias="none",
        inference_mode=False,
    )
    
    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info(f"  LoRA rank: {lora_config.r}, alpha: {lora_config.lora_alpha}")
    logger.info(f"  Target modules: {len(lora_config.target_modules)}")
    logger.info(f"  Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    logger.info("\n[5/5] Setting up training...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        
        num_train_epochs=TRAINING_CONFIG['num_epochs'],
        max_steps=-1,  
        
        per_device_train_batch_size=TRAINING_CONFIG['micro_batch_size'],
        per_device_eval_batch_size=TRAINING_CONFIG['eval_batch_size'],
        gradient_accumulation_steps=TRAINING_CONFIG['gradient_accumulation_steps'],
        
        learning_rate=TRAINING_CONFIG['learning_rate'],
        lr_scheduler_type=TRAINING_CONFIG['lr_scheduler'],
        warmup_ratio=TRAINING_CONFIG['warmup_ratio'],
        weight_decay=TRAINING_CONFIG['weight_decay'],
        max_grad_norm=TRAINING_CONFIG['max_grad_norm'],
        
        bf16=TRAINING_CONFIG['bf16'],
        fp16=TRAINING_CONFIG['fp16'],
        tf32=TRAINING_CONFIG['tf32'],
        
        logging_steps=TRAINING_CONFIG['logging_steps'],
        logging_dir=f"{output_dir}/logs",
        report_to=["tensorboard"],
        
        save_strategy="steps",
        save_steps=TRAINING_CONFIG['save_steps'],
        save_total_limit=3,
        
        eval_strategy="steps",
        eval_steps=TRAINING_CONFIG['eval_steps'],
        
        gradient_checkpointing=TRAINING_CONFIG['gradient_checkpointing'],
        
        deepspeed="deepspeed_configs/zero2.json",
        
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        seed=TRAINING_CONFIG['seed'],
    )
    
    logger.info(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size}")
    logger.info(f"  Learning rate: {training_args.learning_rate}")
    logger.info(f"  Epochs: {training_args.num_train_epochs}")
    logger.info(f"  Save every: {training_args.save_steps} steps")
    logger.info(f"  Eval every: {training_args.eval_steps} steps")
    
    perplexity_callback = PerplexityCallback()
    training_info_callback = TrainingInfoCallback(
        config=TRAINING_CONFIG,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        timestamp=timestamp
    )
    
    logger.info("\n[READY] Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,  
        formatting_func=lambda x: x["text"],  
        callbacks=[perplexity_callback, training_info_callback],  
    )
    
    def signal_handler(signum, frame):
        logger.info("\n⚠️  Training interrupted! Saving training info before exit...")
        try:
            checkpoints = sorted(Path(output_dir).glob("checkpoint-*"), 
                               key=lambda x: int(x.name.split('-')[1]))
            if checkpoints:
                latest_checkpoint = checkpoints[-1]
                logger.info(f"Saving to latest checkpoint: {latest_checkpoint}")
                save_training_info(str(latest_checkpoint), TRAINING_CONFIG, 
                                 train_dataset, eval_dataset, trainer, timestamp)
            else:
                logger.info(f"Saving to output dir: {output_dir}")
                save_training_info(output_dir, TRAINING_CONFIG, 
                                 train_dataset, eval_dataset, trainer, timestamp)
            logger.info("✓ Training info saved successfully")
        except Exception as e:
            logger.error(f"Failed to save training info: {e}")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("\n" + "=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80 + "\n")
    
    trainer.train()

    logger.info("\n" + "=" * 80)
    logger.info("Training complete! Saving final model...")
    logger.info("=" * 80)
    
    final_model_dir = os.path.join(output_dir, "final_model")
    logger.info(f"\nSaving final model to: {final_model_dir}")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    save_training_info(final_model_dir, TRAINING_CONFIG, train_dataset, eval_dataset, trainer, timestamp)
    
    logger.info("\n✓ Final model saved!")
    logger.info(f"  Location: {final_model_dir}")
    
    adapter_path = os.path.join(final_model_dir, 'adapter_model.safetensors')
    if os.path.exists(adapter_path):
        adapter_size = os.path.getsize(adapter_path) / (1024**3)
        logger.info(f"  LoRA adapters: {adapter_size:.2f} GB")
    else:
        logger.info(f"  LoRA adapters: saved")
    
    logger.info("\n✓ Saving training info for all checkpoints...")
    for checkpoint in sorted(Path(output_dir).glob("checkpoint-*")):
        save_training_info(str(checkpoint), TRAINING_CONFIG, train_dataset, eval_dataset, trainer, timestamp)
        logger.info(f"  • {checkpoint.name}")
    
    logger.info(f"\n✓ Model saved to: {output_dir}")
    logger.info("✓ Training complete!")


if __name__ == "__main__":
    main()
