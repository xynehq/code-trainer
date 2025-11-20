# Quick Reference: Key Training Parameters

## 🎯 Most Important Parameters

### Learning Rate (learning_rate)
```yaml
learning_rate: 2.0e-4  # Recommended for LoRA
```
- **Too high (>3e-4)**: Loss explodes, NaN values
- **Too low (<5e-5)**: Very slow learning, may not converge
- **Sweet spot**: 1e-4 to 3e-4 for LoRA

### Warmup Ratio (warmup_ratio)
```yaml
warmup_ratio: 0.03  # 3% of training
```
- **Purpose**: Prevent instability at start
- **Typical range**: 0.01-0.1
- **Rule**: Larger models → more warmup

### Weight Decay (weight_decay)
```yaml
weight_decay: 0.01  # L2 regularization
```
- **Overfitting?** → Increase to 0.1
- **Underfitting?** → Decrease to 0.001
- **Default**: 0.01

### Batch Size (effective)
```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
# Effective = 1 * 16 * 4 GPUs = 64
```
- **Larger batch**: More stable, slower iteration
- **Smaller batch**: Noisier gradients, faster iteration
- **Sweet spot**: 32-128 for LLMs

### LoRA Rank (r)
```yaml
lora:
  r: 16  # Rank
  lora_alpha: 32  # Usually 2*r
```
- **r=8**: Fastest, least capacity
- **r=16**: Balanced (recommended)
- **r=32**: More capacity, slower
- **r=64**: High capacity, much slower

### MoE Auxiliary Loss
```yaml
moe:
  auxiliary_loss_weight: 0.001
```
- **Expert collapse?** → Increase to 0.01
- **Over-balanced?** → Decrease to 0.0001
- **Default**: 0.001 (10x less than pre-training)

## 📊 Quick Diagnostics

### Training Loss Not Decreasing
1. ✅ Check: `learning_rate: 2.0e-4`
2. ✅ Check: `warmup_ratio: 0.1` (increase warmup)
3. ✅ Check logs for trainable parameters
4. ✅ Verify dataset loaded correctly

### Out of Memory (OOM)
1. ✅ Set: `gradient_accumulation_steps: 32`
2. ✅ Ensure: `gradient_checkpointing: true`
3. ✅ Reduce: `max_seq_length: 2048`
4. ✅ Keep: `per_device_train_batch_size: 1`

### Overfitting (val_loss > train_loss)
1. ✅ Increase: `weight_decay: 0.1`
2. ✅ Reduce: `num_train_epochs: 2`
3. ✅ Add: `label_smoothing_factor: 0.1`
4. ✅ Increase: `validation_split: 0.1`

### Expert Collapse
1. ✅ Increase: `auxiliary_loss_weight: 0.01`
2. ✅ Ensure: `freeze_router: false`
3. ✅ Enable: `monitor_expert_usage: true`
4. ✅ Check logs for expert utilization

### Slow Training
1. ✅ Enable: `attn_implementation: "flash_attention_2"`
2. ✅ Increase: `dataloader_num_workers: 8`
3. ✅ Enable: `torch_compile: true`
4. ✅ Reduce: `logging_steps: 50`

## 🔧 Common Configuration Presets

### Quick Test (Fast)
```yaml
training:
  max_steps: 100  # Just 100 steps
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  evaluation_strategy: "steps"
  eval_steps: 50
  save_steps: 50
```

### Memory Constrained
```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 32
  gradient_checkpointing: true
model:
  max_seq_length: 2048
advanced:
  use_cpu_offload: true
```

### Maximum Quality
```yaml
training:
  num_train_epochs: 5
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 16
  learning_rate: 1.0e-4
  weight_decay: 0.1
lora:
  r: 32
  lora_alpha: 64
```

### Maximum Speed
```yaml
training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 3.0e-4
  logging_steps: 50
  eval_steps: 500
model:
  attn_implementation: "flash_attention_2"
advanced:
  torch_compile: true
```

## 🚀 Commands Cheat Sheet

```bash
# Setup
source .venv/bin/activate
bash setup.sh

# Prepare data
python prepare_dataset.py

# Train
bash train.sh

# Train with custom config
python train_moe.py --config my_config.yaml

# Monitor
tensorboard --logdir outputs/moe-hyperswitch-attn-lora
watch -n 1 nvidia-smi

# Evaluate
python evaluate_moe.py --max-samples 100

# Resume training
# Edit config.yaml:
# resume_from_checkpoint: "outputs/moe-hyperswitch-attn-lora/checkpoint-500"
bash train.sh
```

## 📈 Expected Metrics

### Good Training Run
- **Train Loss**: Decreasing smoothly from ~3.0 to ~1.5
- **Val Loss**: Following train loss (gap < 0.5)
- **Perplexity**: Improving from ~15 to ~5
- **GPU Util**: 70-95%
- **Time/Step**: 30-45 seconds

### Warning Signs
- ❌ Loss = NaN (reduce learning_rate)
- ❌ Loss not changing (check data/LR)
- ❌ Val loss >> train loss (overfitting)
- ❌ GPU util < 50% (increase batch/workers)
- ❌ All tokens → 1 expert (increase aux loss)

## 🎓 Parameter Relationships

```
Effective Batch = per_device_batch × accumulation × num_gpus
                = 1 × 16 × 4 = 64

Total Steps = (num_samples / effective_batch) × epochs
            = (40000 / 64) × 3 = 1875

Warmup Steps = total_steps × warmup_ratio
             = 1875 × 0.03 = 56

Training Time = steps × time_per_step
              = 1875 × 35s ≈ 18 hours
```

## 💡 Pro Tips

1. **Start conservative**: Use default config first
2. **One change at a time**: Easier to debug
3. **Monitor continuously**: Use tensorboard
4. **Save checkpoints**: Don't lose progress
5. **Log everything**: Helps debugging
6. **Test on small data**: Verify pipeline works
7. **Compare to baseline**: Always evaluate
8. **Document changes**: Track what works

---

**Need help?** Check TRAINING_GUIDE.md for detailed explanations!
