# MoE Fine-tuning for Hyperswitch

Fine-tune Mixtral MoE models on Hyperswitch codebase using LoRA on attention blocks only (FFN/Experts frozen).

## 🎯 Overview

This project implements parameter-efficient fine-tuning of Mixture-of-Experts (MoE) models using:
- **LoRA (Low-Rank Adaptation)** on attention blocks only
- **Frozen FFN/Expert layers** to preserve general knowledge
- **4x NVIDIA H200 GPUs** with distributed training
- **Hyperswitch Rust codebase** as training data

Based on research from:
- [Scale AI: Fine-Tuning Mixture of Experts PEFT](https://scale.com/blog/fine-tuning-mixture-of-experts-peft)
- [ApX ML: Fine-tuning Pretrained MoE](https://apxml.com/courses/mixture-of-experts-advanced-implementation/chapter-3-training-large-scale-moes/fine-tuning-pretrained-moe)

## 📋 Requirements

- 4x NVIDIA H200 GPUs (141GB VRAM each)
- Python 3.10+
- CUDA 12.1+
- ~200GB disk space

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Run setup script
bash setup.sh

# Or manually:
source .venv/bin/activate
```

### 2. Prepare Dataset

```bash
# Generate dataset from Hyperswitch codebase
python prepare_dataset.py
```

This will:
- Extract Rust code from `hyperswitch/` repository
- Create file-level chunks and granular samples (functions, structs, traits)
- Save to `data/all_data.jsonl`

### 3. Configure Training

Edit `config.yaml` to customize:
- Model name (default: `mistralai/Mixtral-8x7B-v0.1`)
- LoRA parameters (rank, alpha, dropout)
- Training hyperparameters (learning rate, batch size, epochs)
- Dataset paths and preprocessing settings

### 4. Start Training

```bash
# Launch distributed training on 4 GPUs
bash train.sh
```

Training will:
- Load Mixtral model with BF16 precision
- Apply LoRA adapters to attention blocks only
- Freeze all FFN/Expert layers
- Train with gradient checkpointing and accumulation
- Save checkpoints every 100 steps
- Log to TensorBoard

### 5. Evaluate Model

```bash
# Compare base vs fine-tuned model
python evaluate_moe.py --max-samples 100
```

## 📂 Project Structure

```
MoE/
├── config.yaml              # Configuration file
├── prepare_dataset.py       # Dataset preparation script
├── train_moe.py            # Main training script
├── evaluate_moe.py         # Evaluation script
├── train.sh                # Training launcher
├── setup.sh                # Environment setup
├── README.md               # This file
├── hyperswitch/            # Cloned Hyperswitch repo
├── data/                   # Dataset directory
│   └── all_data.jsonl     # Generated dataset
├── outputs/                # Training outputs
│   └── moe-hyperswitch-attn-lora/
│       ├── checkpoint-*/   # Model checkpoints
│       └── runs/          # TensorBoard logs
└── .venv/                  # Virtual environment
```

## ⚙️ Configuration

### Key Settings in `config.yaml`

**Model:**
```yaml
model:
  name: "mistralai/Mixtral-8x7B-v0.1"
  torch_dtype: "bfloat16"
  max_seq_length: 4096
  attn_implementation: "flash_attention_2"
```

**LoRA (Attention Only):**
```yaml
lora:
  r: 16                    # LoRA rank
  lora_alpha: 32           # Scaling factor
  target_modules:          # ONLY attention layers
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
  exclude_modules:         # FFN/Experts FROZEN
    - "block_sparse_moe"
    - "w1"
    - "w2"
    - "w3"
    - "gate"
```

**Training:**
```yaml
training:
  num_train_epochs: 3
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16  # Effective batch = 64
  learning_rate: 2.0e-4           # Higher for LoRA
  gradient_checkpointing: true
```

**MoE Specific:**
```yaml
moe:
  use_auxiliary_loss: true
  auxiliary_loss_weight: 0.001    # Reduced from pre-training
  freeze_router: false            # Allow adaptation
```

## 🎓 Training Strategy

### Why Attention-Only LoRA?

1. **Preserve Expert Knowledge**: Freezing FFN/Expert layers maintains the specialized knowledge learned during pre-training
2. **Memory Efficient**: Only ~0.1% of parameters are trainable
3. **Domain Adaptation**: Attention layers learn to route to appropriate experts for Rust/Hyperswitch domain
4. **Prevent Catastrophic Forgetting**: Base capabilities remain intact

### Training Process

1. **Load Base Model**: Mixtral 8x7B with BF16 precision
2. **Apply LoRA**: Add low-rank adapters to attention blocks only
3. **Freeze Experts**: All FFN/MoE layers remain frozen
4. **Train**: Optimize only LoRA parameters with auxiliary loss
5. **Merge**: Optionally merge LoRA weights back into base model

### Memory Usage (per H200 GPU)

- Model: ~90GB (Mixtral 8x7B in BF16)
- Gradients: ~1GB (LoRA parameters only)
- Optimizer: ~2GB (AdamW state)
- Activations: ~20GB (with gradient checkpointing)
- **Total**: ~115GB per GPU (well within 141GB)

## 📊 Monitoring

### TensorBoard

```bash
tensorboard --logdir outputs/moe-hyperswitch-attn-lora
```

Metrics tracked:
- Training loss
- Evaluation loss
- Learning rate
- Gradient norms
- Expert utilization (MoE specific)

### Weights & Biases (Optional)

Enable in `config.yaml`:
```yaml
wandb:
  enabled: true
  project: "moe-hyperswitch-finetuning"
```

## 🧪 Evaluation

The evaluation script compares:
- **Perplexity**: Lower is better
- **Loss**: Cross-entropy loss on validation set
- **Generation Quality**: Side-by-side comparison of generated code

Sample output:
```
Base Model Metrics:
  Loss: 2.3451
  Perplexity: 10.432

Fine-tuned Model Metrics:
  Loss: 1.8923
  Perplexity: 6.638

Improvement:
  Loss improvement: +19.32%
  Perplexity improvement: +36.38%
```

## 🔧 Troubleshooting

### Out of Memory (OOM)

- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps`
- Enable `gradient_checkpointing`
- Reduce `max_seq_length`

### Slow Training

- Enable Flash Attention 2: `pip install flash-attn`
- Use `torch_compile: true` (PyTorch 2.0+)
- Increase `dataloader_num_workers`
- Check GPU utilization: `nvidia-smi dmon`

### Expert Collapse

If all tokens route to single expert:
- Increase `auxiliary_loss_weight` (e.g., 0.01)
- Monitor expert usage in logs
- Check `monitor_expert_usage: true`

## 📚 References

- [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [Pushing Mixture of Experts to the Limit](https://arxiv.org/abs/2309.05444)
- [Switch Transformers](https://arxiv.org/abs/2101.03961)

## 📝 License

This project follows the licenses of its dependencies:
- Mixtral: Apache 2.0
- Transformers: Apache 2.0
- PEFT: Apache 2.0
- Hyperswitch: Apache 2.0

## 🤝 Contributing

Contributions welcome! Please:
1. Test on your setup
2. Update documentation
3. Follow existing code style

## 📧 Support

For issues or questions:
1. Check troubleshooting section
2. Review configuration carefully
3. Monitor training logs
4. Check GPU memory usage

---

**Happy Fine-tuning! 🚀**
