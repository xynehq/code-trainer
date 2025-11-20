# MoE Fine-tuning Training Guide

Complete guide for fine-tuning Mixtral MoE models on Hyperswitch codebase.

## 📚 Table of Contents

1. [Overview](#overview)
2. [Training Parameters Explained](#training-parameters-explained)
3. [Step-by-Step Training](#step-by-step-training)
4. [Hyperparameter Tuning](#hyperparameter-tuning)
5. [Troubleshooting](#troubleshooting)

---

## Overview

This project fine-tunes Mixtral 8x7B (or 8x22B) using:
- **LoRA on Attention Blocks Only**: q_proj, k_proj, v_proj, o_proj
- **Frozen FFN/Experts**: All MoE layers remain unchanged
- **4x H200 GPUs**: Distributed training with gradient accumulation
- **BF16 Precision**: For H200 optimal performance

**Key Benefits:**
- Only ~0.1% of parameters trained (memory efficient)
- Preserves expert specialization from pre-training
- Adapts attention routing to Rust/Hyperswitch domain
- No catastrophic forgetting

---

## Training Parameters Explained

### Learning Rate Parameters

```yaml
learning_rate: 2.0e-4          # Base learning rate
min_learning_rate: 1.0e-6      # Minimum LR (for cosine decay)
lr_scheduler_type: "cosine"    # Scheduler type
warmup_ratio: 0.03             # 3% warmup
```

**Explanation:**
- **learning_rate**: Higher for LoRA (1e-4 to 3e-4) vs full fine-tuning (1e-5 to 5e-5)
- **min_learning_rate**: Prevents LR from going to zero in cosine schedule
- **warmup_ratio**: Gradually increases LR from 0 to learning_rate over first 3% of training
- **lr_scheduler_type**: 
  - `cosine`: Smooth decay (recommended)
  - `linear`: Linear decay
  - `constant`: No decay
  - `cosine_with_restarts`: Periodic restarts

**When to adjust:**
- Loss exploding → Reduce learning_rate to 1e-4
- Loss plateaus early → Increase to 3e-4
- Unstable training → Increase warmup_ratio to 0.05-0.1

### Batch Size and Accumulation

```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
# Effective batch = 1 * 16 * 4 GPUs = 64
```

**Explanation:**
- **per_device_train_batch_size**: Actual batch per GPU (limited by memory)
- **gradient_accumulation_steps**: Accumulate gradients over N steps before update
- **Effective batch size** = per_device × accumulation × num_gpus

**When to adjust:**
- OOM error → Keep per_device at 1, increase accumulation
- Underutilized GPU → Increase per_device_train_batch_size
- Noisy gradients → Increase effective batch size
- Faster iteration → Decrease accumulation (trades off batch size)

### Regularization

```yaml
weight_decay: 0.01             # L2 regularization
label_smoothing_factor: 0.0    # Label smoothing
dropout: 0.05                  # Dropout in LoRA
```

**Explanation:**
- **weight_decay**: Prevents overfitting by penalizing large weights
- **label_smoothing**: Softens hard labels (usually 0 for LM)
- **dropout**: Random neuron dropout in LoRA layers

**When to adjust:**
- Overfitting (train loss << val loss) → Increase weight_decay to 0.1
- Underfitting → Decrease to 0.001 or 0
- Model too confident → Try label_smoothing 0.1

### Optimizer Settings

```yaml
optim: "adamw_torch_fused"
adam_beta1: 0.9                # Momentum
adam_beta2: 0.999              # RMSprop-like component
adam_epsilon: 1.0e-8           # Numerical stability
max_grad_norm: 1.0             # Gradient clipping
```

**Explanation:**
- **optim**: Optimizer algorithm
  - `adamw_torch_fused`: Fastest (fused CUDA kernels)
  - `adamw_torch`: Standard PyTorch AdamW
  - `adafactor`: Memory-efficient (no momentum)
  - `sgd`: Simple SGD (rarely used for LLMs)
- **adam_beta1**: Exponential decay for first moment (momentum)
- **adam_beta2**: Exponential decay for second moment (adaptive LR)
- **max_grad_norm**: Clips gradients to prevent exploding gradients

**When to adjust:**
- Unstable training → Reduce max_grad_norm to 0.5
- Memory issues → Use adafactor (saves ~50% optimizer memory)
- Slow convergence → Increase beta1 to 0.95

### MoE Specific

```yaml
moe:
  use_auxiliary_loss: true
  auxiliary_loss_weight: 0.001
  freeze_router: false
```

**Explanation:**
- **auxiliary_loss**: Load balancing loss to prevent expert collapse
- **auxiliary_loss_weight**: Weight of aux loss (0.001-0.01)
- **freeze_router**: If true, only train attention (not router)

**When to adjust:**
- Expert collapse (all tokens → 1 expert) → Increase weight to 0.01
- Router needs adaptation → Keep freeze_router: false
- Pure attention training → Set freeze_router: true

### Inference Parameters

```yaml
inference:
  temperature: 0.7             # Randomness
  top_k: 50                    # Top-k sampling
  top_p: 0.95                  # Nucleus sampling
  repetition_penalty: 1.0      # Repetition penalty
```

**Explanation:**
- **temperature**: 
  - 0.0-0.5: More deterministic, focused
  - 0.7-0.9: Balanced creativity
  - 1.0+: More random, creative
- **top_k**: Only sample from top K tokens
- **top_p**: Sample from smallest set with cumulative probability ≥ p
- **repetition_penalty**: 
  - 1.0: No penalty
  - 1.2: Mild penalty
  - 1.5+: Strong penalty

**When to adjust:**
- Code generation → Lower temperature (0.2-0.5)
- Creative writing → Higher temperature (0.8-1.0)
- Repetitive output → Increase repetition_penalty to 1.2

---

## Step-by-Step Training

### 1. Environment Setup

```bash
# Activate virtual environment
source .venv/bin/activate

# Verify GPUs
nvidia-smi

# Check installations
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

### 2. Prepare Dataset

```bash
# Generate dataset from Hyperswitch
python prepare_dataset.py

# Check dataset
ls -lh data/all_data.jsonl
wc -l data/all_data.jsonl  # Count samples
```

Expected output:
```
✓ Found 8000+ Rust files
✓ Created 15000+ file-level samples
✓ Created 25000+ granular samples
✓ Total: 40000+ samples
```

### 3. Configure Training

Edit `config.yaml` based on your needs:

**Quick test (1 epoch, small dataset):**
```yaml
training:
  num_train_epochs: 1
  max_steps: 100  # Only 100 steps for testing
```

**Full training:**
```yaml
training:
  num_train_epochs: 3
  max_steps: -1  # Use epochs
```

**Memory-constrained:**
```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 32  # Increase accumulation
  gradient_checkpointing: true
```

### 4. Launch Training

```bash
# Launch distributed training
bash train.sh

# Or with custom config
python train_moe.py --config my_config.yaml
```

**Monitor training:**
```bash
# In another terminal
tensorboard --logdir outputs/moe-hyperswitch-attn-lora

# Watch GPU usage
watch -n 1 nvidia-smi
```

### 5. Evaluate Model

```bash
# After training completes
python evaluate_moe.py --max-samples 100

# With custom adapter path
python evaluate_moe.py --adapter-path outputs/moe-hyperswitch-attn-lora/checkpoint-500
```

---

## Hyperparameter Tuning

### Recommended Starting Points

**Conservative (safe, slower):**
```yaml
learning_rate: 1.0e-4
weight_decay: 0.1
gradient_accumulation_steps: 32
num_train_epochs: 5
```

**Aggressive (faster, may be unstable):**
```yaml
learning_rate: 3.0e-4
weight_decay: 0.001
gradient_accumulation_steps: 8
num_train_epochs: 2
```

**Balanced (recommended):**
```yaml
learning_rate: 2.0e-4
weight_decay: 0.01
gradient_accumulation_steps: 16
num_train_epochs: 3
```

### Tuning Process

1. **Baseline Run**: Use default config, train for 1 epoch
2. **Learning Rate**: Try [1e-4, 2e-4, 3e-4], pick best validation loss
3. **Batch Size**: Increase if GPU underutilized
4. **Regularization**: Add weight_decay if overfitting
5. **Epochs**: Train longer if still improving

### Grid Search Example

```bash
# Create config variants
for lr in 1e-4 2e-4 3e-4; do
  for wd in 0.01 0.1; do
    sed "s/learning_rate: .*/learning_rate: $lr/" config.yaml > config_lr${lr}_wd${wd}.yaml
    sed -i "s/weight_decay: .*/weight_decay: $wd/" config_lr${lr}_wd${wd}.yaml
    python train_moe.py --config config_lr${lr}_wd${wd}.yaml
  done
done
```

---

## Troubleshooting

### Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions (in order of preference):**

1. **Reduce batch size:**
```yaml
per_device_train_batch_size: 1  # Already at minimum
```

2. **Increase gradient accumulation:**
```yaml
gradient_accumulation_steps: 32  # From 16
```

3. **Enable gradient checkpointing:**
```yaml
gradient_checkpointing: true
```

4. **Reduce sequence length:**
```yaml
max_seq_length: 2048  # From 4096
```

5. **Use CPU offload:**
```yaml
advanced:
  use_cpu_offload: true
```

### Loss Not Decreasing

**Symptoms:**
- Loss stays constant or increases
- Validation loss same as initial

**Solutions:**

1. **Check learning rate:**
```yaml
learning_rate: 2.0e-4  # Try 1e-4 or 3e-4
```

2. **Verify trainable parameters:**
```bash
# Check logs for:
# "trainable params: XXX || all params: YYY || trainable%: Z.ZZ"
```

3. **Increase warmup:**
```yaml
warmup_ratio: 0.1  # From 0.03
```

4. **Check data:**
```python
# Verify dataset loading
python -c "from datasets import load_dataset; ds = load_dataset('json', data_files='data/all_data.jsonl'); print(ds)"
```

### Expert Collapse

**Symptoms:**
- Logs show: "Expert utilization: [1.0, 0.0, 0.0, ...]"
- All tokens routed to single expert

**Solutions:**

1. **Increase auxiliary loss:**
```yaml
moe:
  auxiliary_loss_weight: 0.01  # From 0.001
```

2. **Check router training:**
```yaml
moe:
  freeze_router: false  # Ensure router trains
```

3. **Monitor expert usage:**
```yaml
moe:
  monitor_expert_usage: true
  log_expert_metrics: true
```

### Slow Training

**Symptoms:**
- < 0.5 samples/sec
- Low GPU utilization (< 70%)

**Solutions:**

1. **Enable Flash Attention:**
```bash
pip install flash-attn --no-build-isolation
```
```yaml
model:
  attn_implementation: "flash_attention_2"
```

2. **Increase data workers:**
```yaml
training:
  dataloader_num_workers: 8  # From 4
```

3. **Enable torch compile:**
```yaml
advanced:
  torch_compile: true
```

4. **Reduce logging:**
```yaml
training:
  logging_steps: 50  # From 10
```

### Validation Loss Increasing

**Symptoms:**
- Training loss decreasing
- Validation loss increasing (overfitting)

**Solutions:**

1. **Add regularization:**
```yaml
training:
  weight_decay: 0.1  # From 0.01
```

2. **Early stopping:**
```yaml
advanced:
  early_stopping_patience: 3
  early_stopping_threshold: 0.001
```

3. **Reduce epochs:**
```yaml
training:
  num_train_epochs: 2  # From 3
```

4. **Increase validation split:**
```yaml
validation:
  validation_split: 0.1  # From 0.05
```

---

## Monitoring Checklist

### During Training

- [ ] GPU utilization > 70% (watch -n 1 nvidia-smi)
- [ ] Loss decreasing consistently
- [ ] Validation loss tracking training loss
- [ ] No OOM errors
- [ ] Expert utilization balanced (if logging)
- [ ] Gradient norms reasonable (< 10)
- [ ] Learning rate following schedule

### After Training

- [ ] Final perplexity < initial perplexity
- [ ] Checkpoints saved successfully
- [ ] Validation metrics logged
- [ ] Generated samples look reasonable
- [ ] Model size as expected (~16GB for LoRA adapters)

---

## Advanced Tips

### 1. Finding Optimal Batch Size

```python
# Add to config
training:
  auto_find_batch_size: true
```

### 2. Gradient Accumulation Optimization

```python
# Dynamic accumulation based on sequence length
training:
  gradient_accumulation_steps_auto: true
```

### 3. Mixed Precision Training

```yaml
# Already enabled for H200
training:
  bf16: true  # Use BF16 on H200/A100
  fp16: false # Use FP16 on V100/older
```

### 4. Distributed Training Optimization

```yaml
training:
  ddp_backend: "nccl"
  ddp_timeout: 3600  # Increase for large models
```

### 5. Checkpoint Management

```yaml
training:
  save_steps: 100
  save_total_limit: 3  # Keep only 3 best
  load_best_model_at_end: true
  metric_for_best_model: "eval_loss"
```

---

## Expected Training Time

**Mixtral 8x7B on 4x H200:**
- Dataset: 40,000 samples
- Batch size: 64 (effective)
- Epochs: 3
- **Estimated time**: 12-18 hours

**Breakdown:**
- Samples per epoch: 40,000
- Steps per epoch: 40,000 / 64 = 625
- Total steps: 625 * 3 = 1,875
- Time per step: ~30-45 seconds
- Total: ~15-23 hours

**To reduce training time:**
1. Reduce num_train_epochs to 2
2. Increase batch size (if memory allows)
3. Enable torch.compile
4. Reduce validation frequency

---

## Next Steps After Training

1. **Evaluate thoroughly:**
   ```bash
   python evaluate_moe.py --max-samples 500
   ```

2. **Test on real prompts:**
   ```python
   # Create test_prompts.txt with Rust code prompts
   # Run inference script
   ```

3. **Merge weights (optional):**
   ```yaml
   post_training:
     merge_and_save: true
   ```

4. **Deploy model:**
   - Push to HuggingFace Hub
   - Create inference endpoint
   - Integrate with IDE

5. **Further fine-tuning:**
   - If results good but not perfect, train for 1-2 more epochs
   - Try different LoRA rank (r=32 instead of 16)
   - Experiment with learning rate

---

**Happy Training! 🚀**

For questions or issues, check the troubleshooting section or review training logs.
