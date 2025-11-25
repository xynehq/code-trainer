# Qwen2.5-Coder-7B Fine-Tuning on TPU

Fine-tune Qwen2.5-Coder-7B-Instruct on Hyperswitch Rust codebase using LoRA on Google Cloud TPU v6e.

## 📋 Overview

This project fine-tunes the Qwen2.5-Coder-7B-Instruct model on the Hyperswitch Rust repository using:
- **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning
- **8 x TPU v6e cores** for distributed training
- **Data Parallel** training strategy
- **Weights & Biases** for experiment tracking
- **BFloat16** precision for memory efficiency

## 🔧 Hardware Requirements

- **TPU**: 8 x v6e cores (32GB HBM each)
- **Memory**: ~256GB total HBM across all cores
- **Storage**: ~60GB for model, dataset, and checkpoints

## 📦 Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd TPU_FineTuning
```

### 2. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Clone Hyperswitch Repository

```bash
git clone https://github.com/juspay/hyperswitch.git
```

## 🚀 Quick Start

### 1. Prepare Dataset

```bash
python prepare_dataset.py
```

This will:
- Parse the Hyperswitch Rust codebase
- Extract code samples (full files, functions, structs, traits)
- Generate training data in JSONL format
- Output: `dataset/all_data.jsonl` (~17K samples, 7.1M tokens)

### 2. Configure Training

Edit `config.yaml` to customize:
- Model settings (LoRA rank, alpha, dropout)
- Training hyperparameters (learning rate, batch size, epochs)
- TPU configuration
- Weights & Biases settings

### 3. Run Training

```bash
bash run_training.sh
```

Training will:
- Load model across 8 TPU cores
- Apply LoRA adapters (trainable params: ~2% of total)
- Train with gradient accumulation and checkpointing
- Log metrics to Weights & Biases
- Save checkpoints every 50 steps (keep last 3)

## 📊 Training Configuration

### Current Settings

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Model** | Qwen2.5-Coder-7B-Instruct | Base model |
| **LoRA Rank** | 16 | Low-rank dimension |
| **LoRA Alpha** | 32 | Scaling factor |
| **Learning Rate** | 1e-4 | Initial LR |
| **Batch Size** | 1 | Per-device batch size |
| **Gradient Accumulation** | 4 | Accumulation steps |
| **Effective Batch Size** | 32 | 1 × 4 × 8 devices |
| **Sequence Length** | 1024 | Max tokens per sample |
| **Epochs** | 5 | Training epochs |
| **Total Steps** | 2,405 | Total optimizer steps |

### Memory Optimization

- **BFloat16 precision**: Reduces model size by 50%
- **Gradient checkpointing**: Reduces activation memory
- **LoRA adapters**: Only 2% of parameters trainable
- **No KV cache**: Disabled during training

## 📁 Project Structure

```
TPU_FineTuning/
├── config.yaml              # Main configuration file
├── train_tpu.py             # Training script
├── prepare_dataset.py       # Dataset preparation
├── run_training.sh          # Training launcher
├── tpu_utils.py             # Utility functions
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── dataset/                 # Training data
│   └── all_data.jsonl       # Generated dataset
├── checkpoints/             # Model checkpoints
│   ├── checkpoint-step_50/  # Intermediate checkpoints
│   ├── checkpoint-step_100/
│   ├── checkpoint-step_150/
│   └── checkpoint-final/    # Final model
├── wandb/                   # Weights & Biases logs
└── hyperswitch/             # Source code repository
```

## 💾 Checkpoint Structure

Each checkpoint contains:

```
checkpoint-{name}/
├── adapter_config.json          # LoRA configuration
├── adapter_model.safetensors    # LoRA adapter weights
├── tokenizer.json               # Tokenizer vocabulary
├── tokenizer_config.json        # Tokenizer settings
├── special_tokens_map.json      # Special tokens
├── training_info.json           # Training metadata
└── training_config.yaml         # Full training config
```

## 📈 Monitoring

### Weights & Biases

Training metrics are logged to Weights & Biases:
- Training loss and perplexity
- Learning rate schedule
- Validation metrics
- System metrics (memory, throughput)

Access your run at:
```
https://wandb.ai/<entity>/hyperswitch-cpt
```

### Terminal Output

Real-time progress bar shows:
- Current epoch and step
- Training loss and perplexity
- Learning rate
- Iteration speed

Example:
```
Epoch 1/5: 27%|████▍ | 131/481 [05:59<12:09, 2.46it/s, loss=11.45, ppl=93730, lr=4.17e-05, global_step=131/2405]
```

## 🔍 Key Features

### LoRA Fine-Tuning
- **Efficient**: Only 2% of parameters trainable (~140M params)
- **Fast**: Trains 10x faster than full fine-tuning
- **Portable**: Adapters are small (~500MB vs 14GB full model)

### Data Parallel Training
- **Simple**: Full model replica on each TPU
- **Stable**: No complex sharding logic
- **Compatible**: Works with PEFT library

### Automatic Checkpointing
- Saves every 50 steps
- Keeps only last 3 checkpoints
- Final checkpoint always saved
- Automatic cleanup of old checkpoints

## 🛠️ Advanced Usage

### Resume from Checkpoint

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

# Load LoRA adapters
model = PeftModel.from_pretrained(model, "checkpoints/checkpoint-final")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("checkpoints/checkpoint-final")
```

### Custom Dataset

Edit `prepare_dataset.py` to use your own codebase:

```python
# Update repository path in config.yaml
repository:
  path: "/path/to/your/repo"
  language: "rust"  # or python, javascript, etc.
```

### Adjust Hyperparameters

Edit `config.yaml`:

```yaml
training:
  num_epochs: 10              # More epochs
  learning_rate: 0.0002       # Higher learning rate
  gradient_accumulation_steps: 8  # Larger effective batch
  
  lora:
    r: 32                     # Higher rank = more capacity
    alpha: 64                 # Typically 2*r
```

## 📊 Expected Performance

### Training Time
- **Per epoch**: ~2-3 hours on 8 x TPU v6e
- **Total (5 epochs)**: ~10-15 hours

### Metrics
- **Initial loss**: ~11-12
- **Final loss**: ~2-3 (target)
- **Perplexity**: Should decrease from ~90K to ~10-20

### Memory Usage
- **Per TPU core**: ~28-30GB / 32GB
- **Model size**: ~14GB (bfloat16)
- **Gradients + activations**: ~14-16GB

## 🐛 Troubleshooting

### TPU Not Detected

```bash
# Check TPU availability
python -c "import torch_xla.runtime as xr; print('TPU cores:', xr.world_size())"

# Clear locked TPU
pkill -9 python
```

### Out of Memory

Reduce memory usage in `config.yaml`:
```yaml
dataset:
  max_tokens: 512             # Reduce sequence length

training:
  micro_batch_size: 1         # Already minimum
  gradient_accumulation_steps: 2  # Reduce if needed
```

### Slow Training

Check XLA compilation:
- First epoch is slow (XLA compiles graphs)
- Subsequent epochs should be faster
- Monitor `it/s` in progress bar

## 📝 Citation

```bibtex
@software{qwen25coder,
  title = {Qwen2.5-Coder: Technical Report},
  author = {Qwen Team},
  year = {2024},
  url = {https://github.com/QwenLM/Qwen2.5-Coder}
}
```

## 📄 License

This project follows the license of the base model (Apache 2.0).

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Last Updated**: November 25, 2025
