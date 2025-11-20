#!/usr/bin/env python3
"""
MoE Fine-tuning with LoRA on Attention Blocks Only
Freezes FFN/Expert layers, trains only attention with LoRA adapters
Based on best practices from Scale AI and ApX ML research
"""

import os
import sys
import yaml
import json
import torch
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import numpy as np
from datetime import datetime
import platform

# Suppress dataset fingerprint warnings
warnings.filterwarnings("ignore", category=UserWarning, module="datasets.fingerprint")

# Transformers and PEFT
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from datasets import load_dataset, Dataset
from tqdm import tqdm
import evaluate

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


class MoEFineTuner:
    """Fine-tune MoE models with LoRA on attention blocks only"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration"""
        logger.info("=" * 80)
        logger.info("🚀 MoE Fine-tuning with Attention-Only LoRA")
        logger.info("=" * 80)
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Generate run name with model + lr + timestamp
        from datetime import datetime
        model_short = self.config['model']['name'].split('/')[-1]
        lr = self.config['training']['learning_rate']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{model_short}_lr{lr}_{timestamp}"
        
        # Update config with generated run name
        self.config['training']['run_name'] = self.run_name
        self.config['training']['output_dir'] = f"outputs/{self.run_name}"
        if 'wandb' in self.config and self.config['wandb'].get('enabled'):
            self.config['wandb']['name'] = self.run_name
        
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None
        
        # Set seed for reproducibility
        set_seed(self.config['training']['seed'])
        
        logger.info(f"📊 Configuration loaded from {config_path}")
        logger.info(f"🏷️  Run name: {self.run_name}")
        self._print_config_summary()
    
    def _print_config_summary(self):
        """Print configuration summary"""
        logger.info("\n" + "=" * 80)
        logger.info("📋 Configuration Summary:")
        logger.info("=" * 80)
        logger.info(f"  Model: {self.config['model']['name']}")
        logger.info(f"  LoRA Rank: {self.config['lora']['r']}")
        logger.info(f"  LoRA Alpha: {self.config['lora']['lora_alpha']}")
        logger.info(f"  Target Modules (Attention): {self.config['lora']['target_modules']}")
        logger.info(f"  Excluded Modules (FFN/Experts): {self.config['lora']['exclude_modules']}")
        logger.info(f"  Training Epochs: {self.config['training']['num_train_epochs']}")
        logger.info(f"  Learning Rate: {self.config['training']['learning_rate']}")
        logger.info(f"  Batch Size per GPU: {self.config['training']['per_device_train_batch_size']}")
        logger.info(f"  Gradient Accumulation: {self.config['training']['gradient_accumulation_steps']}")
        logger.info(f"  Number of GPUs: {self.config['hardware']['num_gpus']}")
        
        effective_batch = (
            self.config['training']['per_device_train_batch_size'] * 
            self.config['training']['gradient_accumulation_steps'] * 
            self.config['hardware']['num_gpus']
        )
        logger.info(f"  Effective Batch Size: {effective_batch}")
        logger.info(f"  MoE Auxiliary Loss: {self.config['moe']['use_auxiliary_loss']}")
        logger.info(f"  MoE Aux Loss Weight: {self.config['moe']['auxiliary_loss_weight']}")
        logger.info("=" * 80 + "\n")
    
    def load_tokenizer(self):
        """Load tokenizer"""
        logger.info(f"📚 Loading tokenizer: {self.config['model']['name']}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['name'],
            trust_remote_code=self.config['model']['trust_remote_code'],
            padding_side=self.config['tokenizer']['padding_side'],
            truncation_side=self.config['tokenizer']['truncation_side'],
        )
        
        # Set special tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"✓ Tokenizer loaded. Vocab size: {len(self.tokenizer)}")
        logger.info(f"  PAD token: {self.tokenizer.pad_token}")
        logger.info(f"  EOS token: {self.tokenizer.eos_token}")
        logger.info(f"  BOS token: {self.tokenizer.bos_token}")
    
    def load_model(self):
        """Load base MoE model"""
        logger.info(f"🤖 Loading base model: {self.config['model']['name']}")
        logger.info("  This may take several minutes for large MoE models...")
        
        # Determine dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.config['model']['torch_dtype'], torch.bfloat16)
        
        # Load model with memory-efficient settings
        logger.info("  Loading with low_cpu_mem_usage and max_memory settings...")
        
        # For distributed training, don't use device_map
        # Let accelerate handle device placement
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model']['name'],
            trust_remote_code=self.config['model']['trust_remote_code'],
            dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation=self.config['model'].get('attn_implementation'),
        )
        
        # Enable gradient checkpointing for memory efficiency
        if self.config['training']['gradient_checkpointing']:
            self.model.gradient_checkpointing_enable()
            logger.info("✓ Gradient checkpointing enabled")
        
        # Prepare for training
        if self.config['model'].get('load_in_8bit') or self.config['model'].get('load_in_4bit'):
            self.model = prepare_model_for_kbit_training(self.model)
            logger.info("✓ Model prepared for k-bit training")
        
        logger.info(f"✓ Model loaded successfully")
        self._print_model_info()
    
    def _print_model_info(self):
        """Print model information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"\n  📊 Model Statistics:")
        logger.info(f"    Total parameters: {total_params:,}")
        logger.info(f"    Trainable parameters: {trainable_params:,}")
        logger.info(f"    Trainable %: {100 * trainable_params / total_params:.4f}%")
    
    def setup_lora(self):
        """Setup LoRA configuration for attention blocks only"""
        logger.info("\n🔧 Setting up LoRA (Attention Blocks Only)...")
        
        # Create LoRA config
        lora_config = LoraConfig(
            r=self.config['lora']['r'],
            lora_alpha=self.config['lora']['lora_alpha'],
            target_modules=self.config['lora']['target_modules'],
            lora_dropout=self.config['lora']['lora_dropout'],
            bias=self.config['lora']['bias'],
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            modules_to_save=self.config['lora'].get('modules_to_save', []),
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        
        # Verify that FFN/Expert layers are frozen
        self._verify_frozen_modules()
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        logger.info("✓ LoRA setup complete")
    
    def _verify_frozen_modules(self):
        """Verify that FFN/Expert modules are frozen"""
        logger.info("\n🔍 Verifying frozen modules...")
        
        exclude_patterns = self.config['lora']['exclude_modules']
        frozen_count = 0
        trainable_count = 0
        
        for name, param in self.model.named_parameters():
            is_excluded = any(pattern in name for pattern in exclude_patterns)
            
            if is_excluded:
                if param.requires_grad:
                    logger.warning(f"  ⚠️  WARNING: {name} should be frozen but is trainable!")
                else:
                    frozen_count += 1
            else:
                if param.requires_grad:
                    trainable_count += 1
        
        logger.info(f"✓ Frozen expert/FFN parameters: {frozen_count}")
        logger.info(f"✓ Trainable attention parameters: {trainable_count}")
        
        # Log sample of trainable vs frozen
        logger.info("\n  Sample of parameter status:")
        count = 0
        for name, param in self.model.named_parameters():
            if count < 20:  # Show first 20
                status = "✓ TRAINABLE" if param.requires_grad else "✗ FROZEN"
                logger.info(f"    {status}: {name}")
                count += 1
    
    def load_dataset(self):
        """Load and prepare dataset"""
        logger.info("\n📂 Loading dataset...")
        
        data_path = Path(self.config['dataset']['output_dir']) / self.config['dataset']['data_file']
        
        if not data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {data_path}. "
                f"Please run prepare_dataset.py first!"
            )
        
        # Load dataset
        dataset = load_dataset('json', data_files=str(data_path), split='train')
        logger.info(f"✓ Loaded {len(dataset)} samples from {data_path}")
        
        # Split into train/validation
        split_ratio = self.config['validation']['validation_split']
        dataset = dataset.train_test_split(test_size=split_ratio, seed=self.config['training']['data_seed'])
        
        self.train_dataset = dataset['train']
        self.eval_dataset = dataset['test']
        
        # Limit eval samples if specified (to avoid OOM)
        max_eval_samples = self.config['validation'].get('max_eval_samples')
        if max_eval_samples and len(self.eval_dataset) > max_eval_samples:
            logger.info(f"⚠️  Limiting eval dataset from {len(self.eval_dataset)} to {max_eval_samples} samples")
            self.eval_dataset = self.eval_dataset.select(range(max_eval_samples))
        
        logger.info(f"✓ Train samples: {len(self.train_dataset)}")
        logger.info(f"✓ Validation samples: {len(self.eval_dataset)}")
        
        # Tokenize datasets immediately (before distributed setup)
        self._tokenize_datasets()
    
    def _tokenize_datasets(self):
        """Tokenize datasets - called during load_dataset, not separately"""
        logger.info("\n⚙️  Tokenizing datasets...")
        
        text_field = self.config['dataset']['text_field']
        max_length = self.config['model']['max_seq_length']
        
        # Tokenize directly without multiprocessing to avoid pickle issues
        logger.info("  Tokenizing train dataset...")
        tokenized_train = []
        for i, example in enumerate(tqdm(self.train_dataset, desc="Tokenizing train")):
            tokens = self.tokenizer(
                example[text_field],
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            tokenized_train.append(tokens)
            
        logger.info("  Tokenizing eval dataset...")
        tokenized_eval = []
        for i, example in enumerate(tqdm(self.eval_dataset, desc="Tokenizing eval")):
            tokens = self.tokenizer(
                example[text_field],
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            tokenized_eval.append(tokens)
        
        # Convert to datasets
        from datasets import Dataset as HFDataset
        self.train_dataset = HFDataset.from_dict({
            'input_ids': [t['input_ids'] for t in tokenized_train],
            'attention_mask': [t['attention_mask'] for t in tokenized_train],
        })
        
        self.eval_dataset = HFDataset.from_dict({
            'input_ids': [t['input_ids'] for t in tokenized_eval],
            'attention_mask': [t['attention_mask'] for t in tokenized_eval],
        })
        
        logger.info(f"✓ Tokenization complete")
        
        logger.info(f"✓ Tokenization complete")
        
        # Print sample
        logger.info("\n  Sample tokenized sequence:")
        sample = self.train_dataset[0]
        logger.info(f"    Input IDs shape: {len(sample['input_ids'])}")
        logger.info(f"    Decoded: {self.tokenizer.decode(sample['input_ids'][:100])}...")
    
    def create_trainer(self):
        """Create Trainer instance"""
        logger.info("\n🎯 Creating Trainer...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config['training']['output_dir'],
            run_name=self.config['training']['run_name'],
            
            # Training hyperparameters
            num_train_epochs=self.config['training']['num_train_epochs'],
            per_device_train_batch_size=self.config['training']['per_device_train_batch_size'],
            per_device_eval_batch_size=self.config['training']['per_device_eval_batch_size'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            
            # Optimization
            learning_rate=self.config['training']['learning_rate'],
            lr_scheduler_type=self.config['training']['lr_scheduler_type'],
            warmup_ratio=self.config['training']['warmup_ratio'],
            weight_decay=self.config['training']['weight_decay'],
            optim=self.config['training']['optim'],
            max_grad_norm=self.config['training']['max_grad_norm'],
            
            # Mixed precision
            bf16=self.config['training']['bf16'],
            fp16=self.config['training']['fp16'],
            
            # Evaluation
            eval_strategy=self.config['training']['evaluation_strategy'],
            eval_steps=self.config['training']['eval_steps'],
            eval_accumulation_steps=self.config['training'].get('eval_accumulation_steps'),
            save_strategy=self.config['training']['save_strategy'],
            save_steps=self.config['training']['save_steps'],
            save_total_limit=self.config['training']['save_total_limit'],
            load_best_model_at_end=self.config['training']['load_best_model_at_end'],
            metric_for_best_model=self.config['training']['metric_for_best_model'],
            
            # Logging
            logging_steps=self.config['training']['logging_steps'],
            logging_first_step=self.config['training']['logging_first_step'],
            report_to=self.config['training']['report_to'],
            
            # Distributed
            ddp_find_unused_parameters=self.config['training']['ddp_find_unused_parameters'],
            
            # Performance
            dataloader_num_workers=self.config['training']['dataloader_num_workers'],
            dataloader_pin_memory=self.config['training']['dataloader_pin_memory'],
            
            # Reproducibility
            seed=self.config['training']['seed'],
            data_seed=self.config['training']['data_seed'],
            
            # Checkpointing
            save_safetensors=self.config['training']['save_safetensors'],
            
            # Gradient checkpointing
            gradient_checkpointing=self.config['training']['gradient_checkpointing'],
        )
        
        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )
        
        # Compute metrics
        def compute_metrics(eval_pred):
            """Compute token accuracy and other metrics"""
            predictions, labels = eval_pred
            
            # predictions are logits: shape (batch_size, seq_len, vocab_size)
            # labels are token ids: shape (batch_size, seq_len)
            
            # Get predicted token ids
            if isinstance(predictions, tuple):
                predictions = predictions[0]  # Extract logits if tuple
            
            pred_token_ids = np.argmax(predictions, axis=-1)
            
            # Mask padding tokens (typically -100)
            mask = labels != -100
            
            # Compute token accuracy
            correct_predictions = (pred_token_ids == labels) & mask
            token_accuracy = correct_predictions.sum() / mask.sum()
            
            return {
                'token_accuracy': float(token_accuracy),
            }
        
        # Custom callback to log perplexity and token accuracy
        from transformers import TrainerCallback
        
        class PerplexityCallback(TrainerCallback):
            """Callback to compute and log perplexity and mean token accuracy"""
            
            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                if metrics is not None:
                    # Compute perplexity from eval_loss
                    if 'eval_loss' in metrics:
                        perplexity = np.exp(metrics['eval_loss'])
                        metrics['eval_perplexity'] = perplexity
                        logger.info(f"  Evaluation Perplexity: {perplexity:.4f}")
                    
                    # Log token accuracy if available
                    if 'eval_token_accuracy' in metrics:
                        token_acc = metrics['eval_token_accuracy']
                        metrics['eval_mean_token_accuracy'] = token_acc  # Alias for clarity
                        logger.info(f"  Mean Token Accuracy: {token_acc:.4f} ({token_acc*100:.2f}%)")
        
        class TrainingInfoCallback(TrainerCallback):
            """Callback to save training_info.json at each checkpoint"""
            
            def __init__(self, finetuner):
                self.finetuner = finetuner
                self.initial_metrics = {}
            
            def on_train_begin(self, args, state, control, **kwargs):
                """Save initial state"""
                self.initial_metrics = {
                    'initial_step': state.global_step,
                    'initial_epoch': state.epoch,
                }
            
            def on_save(self, args, state, control, **kwargs):
                """Save training info when checkpoint is saved"""
                checkpoint_dir = f"{args.output_dir}/checkpoint-{state.global_step}"
                if os.path.exists(checkpoint_dir):
                    # Get current metrics from state
                    current_metrics = {
                        'train_steps': state.global_step,
                        'train_loss': state.log_history[-1].get('loss') if state.log_history else None,
                    }
                    self.finetuner.save_training_info(checkpoint_dir, current_metrics)
        
        perplexity_callback = PerplexityCallback()
        training_info_callback = TrainingInfoCallback(self)
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            processing_class=self.tokenizer,  # Changed from tokenizer to processing_class
            compute_metrics=compute_metrics if self.config['validation']['compute_metrics'] else None,
            callbacks=[perplexity_callback, training_info_callback] if self.config['validation'].get('compute_perplexity', True) else [training_info_callback],
        )
        
        logger.info("✓ Trainer created successfully")
    
    def save_training_info(self, output_dir: str, metrics: Dict = None):
        """Save comprehensive training information to JSON file"""
        logger.info(f"\n💾 Saving training_info.json to {output_dir}...")
        
        # Get current timestamp
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        training_date = now.strftime("%Y-%m-%d")
        
        # Get model architecture info
        model_config = self.model.config.to_dict() if hasattr(self.model, 'config') else {}
        model_name = model_config.get('_name_or_path', self.config['model']['name'])
        architectures = model_config.get('architectures', ['Unknown'])
        
        # Get LoRA config
        lora_config = self.config['lora']
        
        # Get trainable parameters info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Get final metrics if available
        final_metrics = {}
        if metrics:
            final_metrics = {
                "final_train_loss": metrics.get('train_loss'),
                "final_train_runtime": metrics.get('train_runtime'),
                "final_train_samples_per_second": metrics.get('train_samples_per_second'),
                "final_train_steps_per_second": metrics.get('train_steps_per_second'),
            }
            
            # Calculate perplexity if loss is available
            if 'train_loss' in metrics:
                final_metrics['final_train_perplexity'] = np.exp(metrics['train_loss'])
        
        # Get library versions
        import transformers
        import peft
        import accelerate
        
        framework_versions = {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "accelerate": accelerate.__version__,
            "python": platform.python_version(),
        }
        
        # Try to get flash-attn version
        try:
            import flash_attn
            framework_versions['flash_attn'] = flash_attn.__version__
        except:
            framework_versions['flash_attn'] = None
        
        # Build training info dictionary
        training_info = {
            "training_metadata": {
                "timestamp": timestamp,
                "training_date": training_date,
                "training_time": now.strftime("%H:%M:%S"),
                "final_epoch": self.config['training']['num_train_epochs'],
                "total_steps": metrics.get('train_steps') if metrics else None,
                "status": "completed" if metrics else "in_progress",
                "run_name": self.run_name,
            },
            "model_config": {
                "base_model": self.config['model']['name'],
                "model_type": "moe_causal_lm",
                "architecture": architectures[0] if architectures else "Unknown",
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "trainable_percentage": f"{100 * trainable_params / total_params:.4f}%",
            },
            "lora_config": {
                "r": lora_config['r'],
                "lora_alpha": lora_config['lora_alpha'],
                "lora_dropout": lora_config['lora_dropout'],
                "target_modules": lora_config['target_modules'],
                "exclude_modules": lora_config.get('exclude_modules', []),
                "bias": lora_config['bias'],
                "use_rslora": lora_config.get('use_rslora', False),
            },
            "training_config": {
                "num_epochs": self.config['training']['num_train_epochs'],
                "per_device_train_batch_size": self.config['training']['per_device_train_batch_size'],
                "per_device_eval_batch_size": self.config['training']['per_device_eval_batch_size'],
                "gradient_accumulation_steps": self.config['training']['gradient_accumulation_steps'],
                "effective_batch_size": (
                    self.config['training']['per_device_train_batch_size'] * 
                    self.config['training']['gradient_accumulation_steps'] * 
                    self.config['hardware']['num_gpus']
                ),
                "learning_rate": self.config['training']['learning_rate'],
                "lr_scheduler_type": self.config['training']['lr_scheduler_type'],
                "warmup_ratio": self.config['training']['warmup_ratio'],
                "weight_decay": self.config['training']['weight_decay'],
                "max_grad_norm": self.config['training']['max_grad_norm'],
                "bf16": self.config['training']['bf16'],
                "gradient_checkpointing": self.config['training']['gradient_checkpointing'],
                "optim": self.config['training']['optim'],
                "logging_steps": self.config['training']['logging_steps'],
                "save_steps": self.config['training']['save_steps'],
                "eval_steps": self.config['training']['eval_steps'],
            },
            "dataset_info": {
                "train_samples": len(self.train_dataset),
                "eval_samples": len(self.eval_dataset),
                "max_seq_length": self.config['model']['max_seq_length'],
                "data_source": self.config['repository']['path'],
            },
            "hardware_config": {
                "num_gpus": self.config['hardware']['num_gpus'],
                "gpu_model": "NVIDIA H200",
                "gpu_memory_per_device_gb": self.config['hardware']['gpu_memory_per_device'],
                "distributed_strategy": "FSDP (Fully Sharded Data Parallel)",
                "fsdp_sharding_strategy": "FULL_SHARD",
                "flash_attention": framework_versions.get('flash_attn', 'Not available'),
            },
            "moe_config": {
                "use_auxiliary_loss": self.config['moe']['use_auxiliary_loss'],
                "auxiliary_loss_weight": self.config['moe']['auxiliary_loss_weight'],
                "freeze_router": self.config['moe']['freeze_router'],
                "num_experts_per_token": self.config['moe']['num_experts_per_token'],
                "monitor_expert_usage": self.config['moe']['monitor_expert_usage'],
            },
            "performance_metrics": final_metrics,
            "framework_versions": framework_versions,
            "special_features": {
                "flash_attention_2": self.config['model'].get('attn_implementation') == 'flash_attention_2',
                "gradient_checkpointing": self.config['training']['gradient_checkpointing'],
                "bf16_training": self.config['training']['bf16'],
                "fsdp_training": True,
                "attention_only_lora": True,
                "frozen_experts": True,
                "eval_accumulation": self.config['training'].get('eval_accumulation_steps') is not None,
            }
        }
        
        # Save to file
        output_path = Path(output_dir) / "training_info.json"
        with open(output_path, 'w') as f:
            json.dump(training_info, f, indent=2)
        
        logger.info(f"✓ Training info saved to {output_path}")
        return training_info
    
    def train(self):
        """Run training"""
        logger.info("\n" + "=" * 80)
        logger.info("🚂 Starting Training...")
        logger.info("=" * 80 + "\n")
        
        # Check for checkpoint
        resume_from = self.config['training'].get('resume_from_checkpoint')
        
        # Train
        train_result = self.trainer.train(resume_from_checkpoint=resume_from)
        
        # Save final model
        logger.info("\n💾 Saving final model...")
        self.trainer.save_model()
        self.trainer.save_state()
        
        # Save metrics
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        
        # Save comprehensive training info
        output_dir = self.config['training']['output_dir']
        self.save_training_info(output_dir, metrics)
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ Training Complete!")
        logger.info("=" * 80)
        
        self._print_training_summary(metrics)
    
    def _print_training_summary(self, metrics: Dict):
        """Print training summary"""
        logger.info("\n📊 Training Summary:")
        logger.info("=" * 80)
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 80)
    
    def evaluate(self):
        """Evaluate model"""
        logger.info("\n📈 Evaluating model...")
        
        eval_metrics = self.trainer.evaluate()
        
        # Compute perplexity
        perplexity = np.exp(eval_metrics['eval_loss'])
        eval_metrics['perplexity'] = perplexity
        
        logger.info("\n📊 Evaluation Results:")
        logger.info("=" * 80)
        for key, value in eval_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        logger.info("=" * 80)
        
        # Save evaluation metrics
        self.trainer.log_metrics("eval", eval_metrics)
        self.trainer.save_metrics("eval", eval_metrics)
        
        return eval_metrics
    
    def merge_and_save(self):
        """Merge LoRA weights and save merged model"""
        if not self.config['post_training']['merge_and_save']:
            logger.info("⏭️  Skipping model merging (disabled in config)")
            return
        
        logger.info("\n🔀 Merging LoRA weights into base model...")
        
        # Merge
        merged_model = self.model.merge_and_unload()
        
        # Save
        output_dir = self.config['post_training']['merged_output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"💾 Saving merged model to {output_dir}")
        merged_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info("✓ Merged model saved successfully")
    
    def run(self):
        """Run complete fine-tuning pipeline"""
        try:
            # Initialize wandb if enabled
            if self.config.get('wandb', {}).get('enabled', False):
                self._init_wandb()
            
            # Load components
            self.load_tokenizer()
            self.load_model()
            self.setup_lora()
            self.load_dataset()
            self.create_trainer()
            
            # Train
            self.train()
            
            # Evaluate
            self.evaluate()
            
            # Merge and save
            self.merge_and_save()
            
            logger.info("\n" + "=" * 80)
            logger.info("🎉 Fine-tuning pipeline completed successfully!")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"\n❌ Error during fine-tuning: {e}", exc_info=True)
            raise
    
    def _init_wandb(self):
        """Initialize Weights & Biases"""
        try:
            import wandb
            
            wandb_config = self.config['wandb']
            api_key = wandb_config.get('api_key')
            
            if api_key and api_key != "YOUR_WANDB_API_KEY_HERE":
                os.environ['WANDB_API_KEY'] = api_key
            
            wandb.init(
                project=wandb_config.get('project'),
                entity=wandb_config.get('entity'),
                name=wandb_config.get('name'),
                tags=wandb_config.get('tags', []),
                notes=wandb_config.get('notes', ''),
                config=self.config,
            )
            
            logger.info("✓ Weights & Biases initialized")
        except ImportError:
            logger.warning("⚠️  wandb not installed. Install with: pip install wandb")
        except Exception as e:
            logger.warning(f"⚠️  Could not initialize wandb: {e}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune MoE with attention-only LoRA")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Run fine-tuning
    finetuner = MoEFineTuner(config_path=args.config)
    finetuner.run()


if __name__ == "__main__":
    main()
