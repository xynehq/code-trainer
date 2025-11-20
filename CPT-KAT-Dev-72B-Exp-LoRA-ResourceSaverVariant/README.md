# Hyperswitch CPT Training - KAT-Dev-72B-Exp-LoRA-ResourceSaver

Memory-optimized implementation using Transformers + TRL + PEFT + DeepSpeed ZeRO-3 with CPU offloading.

## Overview

- **Model**: Kwaipilot/KAT-Dev-72B-Exp (72B experimental instruct model)
- **Method**: LoRA (Low-Rank Adaptation, r=64, alpha=128, Full LoRA - all attention & FFN layers)
- **Hardware**: 4 x H200 GPUs
- **Dataset**: Hyperswitch Rust codebase (~16,731 samples)
- **Training**: Causal Language Modeling (next-token prediction)
- **Memory Strategy**: DeepSpeed ZeRO-3 + CPU Offloading + CPU Activation Checkpointing

## Key Differences vs Standard CPT-KAT-Dev-72B-Exp-LoRA

This variant uses **aggressive memory optimization** techniques to reduce GPU VRAM usage:

### Memory Optimization Features
1. **DeepSpeed ZeRO-3** (vs ZeRO-2 in standard variant)
   - Model parameters sharded across all GPUs
   - Optimizer states sharded across all GPUs
   - Gradients sharded across all GPUs
   
2. **CPU Offloading**
   - Optimizer states offloaded to CPU RAM (~60GB per GPU)
   - Pin memory enabled for faster CPU↔GPU transfers
   
3. **CPU Activation Checkpointing**
   - Activations stored in CPU RAM instead of GPU (~30GB per GPU)
   - Reduces GPU memory by ~40GB per GPU
   
4. **Reduced Batch Size**
   - micro_batch_size: 1 (vs 2 in standard variant)
   - gradient_accumulation_steps: 16 (vs 8)
   - Same effective batch size maintained

### Trade-offs
- ✅ **25GB less GPU VRAM per GPU** (~65GB vs ~90GB)
- ✅ **Can train on GPUs with less memory**
- ✅ **More stable training with lower memory pressure**
- ❌ **~50% slower** (~4-5 hours vs ~2.5-3 hours)
- ❌ **Requires significant CPU RAM** (360GB total)

### When to Use This Variant
- Limited GPU VRAM (e.g., training on A100 40GB or similar)
- Memory instability issues with standard variant
- Want maximum memory safety margin
- Training time is not critical
- Have sufficient CPU RAM available

## Quick Setup After Cloning

### 1. Clone This Repository
```bash
git clone <your-repo-url>
cd CPT_Kwai
```

### 2. Setup Python Environment
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Clone Hyperswitch Repository
```bash
# Clone Hyperswitch codebase (required for dataset generation)
git clone https://github.com/juspay/hyperswitch.git
cd hyperswitch
git checkout main  # or specific commit
cd ..
```

### 4. Prepare Dataset
```bash
# Generate training dataset from Hyperswitch code
python3 prepare_dataset.py

# This creates dataset/all_data.jsonl (~16,731 samples)
```

### 5. Start Training
```bash
# Make script executable
chmod +x train_direct.sh

# Start training
./train_direct.sh
```

### 6. Monitor Training (Optional)
```bash
# Weights & Biases (Primary)
# Training metrics are automatically logged to wandb
# Access your dashboard at: https://wandb.ai/<your-username>/hyperswitch-cpt

# Or monitor console logs
tail -f train_log.txt

# Or watch GPU usage
watch -n 1 nvidia-smi
```

## Configuration

### Model Configuration (`config.yaml`)
- Model: `Kwaipilot/KAT-Dev-72B-Exp` (72B experimental instruct)
- Repository: Path to Hyperswitch codebase
- Dataset: Output directory and token limits
- Training: Epochs, batch sizes, learning rate

### Training Hyperparameters
- **LoRA**: r=64, alpha=128, dropout=0.05 (Full LoRA - all 7 layers)
- **Target modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Batch size**: 1 micro × 16 grad_accum × 4 GPUs = 64 effective
- **Learning rate**: 5e-5 with cosine schedule, warmup=0.03
- **Epochs**: 3
- **Context length**: 8,192 tokens
- **Sample packing**: Disabled
- **Optimizer**: AdamW with weight_decay=0.1

### DeepSpeed Configuration - Memory Optimized
- **zero3_h200.json** (Primary): ZeRO stage 3 with aggressive memory saving
  - Model + Optimizer + Gradient sharding across all GPUs
  - **CPU Offloading**: Optimizer states offloaded to CPU RAM
  - **CPU Activation Checkpointing**: Activations stored in CPU RAM
  - Reduced max live parameters: 3e8 (vs 3e9 in standard variant)
  - Pin memory enabled for faster CPU-GPU transfers
- **zero2.json/zero2_h200.json**: ZeRO stage 2 (fallback options)
- **zero3.json**: ZeRO stage 3 without CPU offloading (fallback)
- BF16 precision
- Flash Attention 2 enabled

## File Structure

```
CPT_Kwai/
├── config.yaml                  # Main configuration
├── train_direct.py              # Training script (Transformers + TRL + PEFT)
├── train_direct.sh              # Training launcher with monitoring
├── prepare_dataset.py           # Dataset preparation from Hyperswitch
├── inference.py                 # Model comparison tool
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── deepspeed_configs/           # DeepSpeed configurations
│   ├── zero2.json              # ZeRO stage 2
│   ├── zero2_h200.json         # ZeRO stage 2 (H200 optimized)
│   └── zero3.json              # ZeRO stage 3
├── dataset/                     # Training data (generated, ~36MB)
│   ├── all_data.jsonl          # 16,731 combined samples
│   └── dataset_stats.json      # Statistics
├── hyperswitch/                 # Cloned repo (not pushed to git)
└── outputs/                     # Training outputs (not pushed to git)
    └── hyperswitch-kat-dev-lora_*/
        ├── logs/               # Wandb logs
        ├── checkpoint-*/       # Saved checkpoints
        └── final_model/        # Final trained model
```

## What Gets Pushed to GitHub

**Essential files (~144KB total):**
- ✅ `config.yaml`, `requirements.txt`
- ✅ `prepare_dataset.py`, `train_direct.py`, `train_direct.sh`
- ✅ `inference.py`
- ✅ `deepspeed_configs/` (zero2.json, zero2_h200.json, zero3.json)
- ✅ `README.md`

**NOT pushed (auto-generated or too large):**
- ❌ `.venv/` (8.4GB) - Recreate with pip
- ❌ `outputs/` (17GB) - Training results
- ❌ `.cache/` (71GB) - HuggingFace cache
- ❌ `hyperswitch/` (325MB) - Clone separately
- ❌ `dataset/` (36MB) - Regenerate with script
- ❌ `__pycache__/`, `*.log` - Runtime artifacts

## Training Details

### Dataset Composition (16,731 total samples)
- **File-level**: 2,120 samples (complete files or chunks)
- **Granular**: 14,611 samples
  - Functions: 4,121 (public function signatures)
  - Structs: 5,710 (public struct definitions)
  - Traits: 223 (trait definitions)
  - Impls: 4,296 (implementation blocks)
  - Modules: 261 (module structure)

### Memory Usage (per GPU) - With ZeRO-3 + CPU Offloading
- Model: ~35GB (72B params in BF16, sharded across 4 GPUs via ZeRO-3)
- LoRA adapters: ~0.7GB (r=64, full LoRA - 7 modules, sharded)
- Optimizer: ~18GB GPU + ~60GB CPU (offloaded to CPU RAM)
- Activations: ~10GB GPU + ~30GB CPU (CPU checkpointing enabled)
- **GPU VRAM: ~65GB / 141GB per H200** (46% utilization - Very comfortable!)
- **System RAM: ~90GB offloaded per GPU** (360GB total CPU RAM usage)

### Expected Training Time
- Total steps: ~1,410 (3 epochs)
- ~10-12 seconds per step (CPU offload overhead, but more stable)
- **Total: ~4-5 hours** on 4x H200 (Slower than standard variant, but uses 25GB less VRAM per GPU!)

### Training Results (Example Run)
- **Initial loss**: TBD → **Final loss**: TBD 
- **Initial perplexity**: TBD → **Final perplexity**: TBD
- **Token accuracy**: TBD → TBD
- **Eval loss**: TBD (experimental 72B memory-optimized run)

### Checkpoints
- Saved every 50 steps
- ~1.2GB per checkpoint (LoRA adapters only)
- Format: SafeTensors
- Final model includes training metadata

## What is CPT?

**Continued Pre-Training (CPT)** adapts a pre-trained model to a specific code domain by continuing the causal language modeling (next-token prediction) on domain-specific code.

### How It Works
The model learns by predicting the next token given previous tokens:
```rust
pub fn create_payment(        ← Model sees this
    request: PaymentRequest   ← Predicts this
) -> Result<Payment, Error> { ← Then this
```

### What The Model Learns
Through next-token prediction, the model learns:
1. **Syntax & Grammar** - Rust language rules
2. **Code Patterns** - `.await?`, error handling chains
3. **API Knowledge** - Function signatures, parameter types
4. **Naming Conventions** - Variable/function naming patterns
5. **Architecture** - Module structure, imports
6. **Domain Logic** - Payment processing concepts
7. **Error Handling** - Hyperswitch error patterns
8. **Documentation** - Doc comment styles

### Why Use Base vs Instruct Model?
We use **Kwaipilot/KAT-Dev** (32B instruct) because:
- Already optimized for code understanding
- Strong base performance on code tasks (62.4% SWE-Bench)
- Based on Qwen2.5 with agentic RL training
- Good balance of size vs performance

### Training Approach
- **Method**: LoRA (efficient fine-tuning)
- **Data**: 16,731 samples from Hyperswitch codebase
- **Goal**: Teach model Hyperswitch-specific patterns
- **Result**: Better at explaining/understanding Hyperswitch code

## Technology Stack

- **Transformers**: Model loading and tokenization
- **TRL (Transformer Reinforcement Learning)**: SFTTrainer with sample packing
- **PEFT**: LoRA implementation
- **DeepSpeed**: Multi-GPU training with ZeRO-2
- **Flash Attention 2**: Memory-efficient attention

## After Training

### Run Inference Comparison
```bash
python3 inference.py

# Compare base model vs fine-tuned on various prompts
# - Code explanations
# - Function completion
# - Module declarations
```

### Test the Fine-tuned Model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Kwaipilot/KAT-Dev",
    dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Kwaipilot/KAT-Dev")

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model, 
    "outputs/hyperswitch-kat-dev-lora_*/final_model"
)

# Generate code
prompt = "// Hyperswitch payment processing\npub fn validate_payment_method"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.2)
print(tokenizer.decode(outputs[0]))
```

### Merge Adapter (Optional)
```python
from peft import PeftModel

# Load and merge
model = PeftModel.from_pretrained(
    base_model, 
    "outputs/hyperswitch-kat-dev-lora_*/final_model"
)
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("hyperswitch-kat-dev-merged")
tokenizer.save_pretrained("hyperswitch-kat-dev-merged")
```

## Evaluation Results

### Strengths (Fine-tuned Model)
✅ **Code Explanation** - Excellent at explaining Hyperswitch code
✅ **Domain Knowledge** - Understands payment processing concepts
✅ **Architecture** - Knows module structure and organization
✅ **Faster** - More concise responses than base model
✅ **Context Aware** - Recognizes Hyperswitch patterns

### Weaknesses (Fine-tuned Model)
❌ **Module Declarations** - Confuses `mod` vs `use` statements
❌ **Macro Completion** - Can struggle with complex macros
❌ **Precise Syntax** - Over-confident on edge cases

### Recommended Use Cases
- ✅ Code reviews and explanations
- ✅ Documentation generation
- ✅ Understanding existing code
- ✅ Architectural discussions
- ✅ Onboarding new developers

### Not Recommended For
- ❌ Blind code generation without review
- ❌ Module structure creation
- ❌ Syntax-critical tasks without validation

### Optimal Settings
- **Temperature**: 0.2-0.3 for code tasks
- **Temperature**: 0.5-0.7 for explanations
- **Max tokens**: 512-1024 for most tasks

## Troubleshooting

### OOM (Out of Memory) Error
Reduce batch size in `config.yaml`:
```yaml
training:
  micro_batch_size: 1        # Was 2 (72B model needs very small batches)
  gradient_accumulation_steps: 16  # Was 8 (compensate with more accumulation)
```

### Loss is NaN
Check learning rate in `config.yaml`:
```yaml
training:
  learning_rate: 0.00003     # Try lower (was 5e-5, 72B models need lower LR)
```

### Slow Training
- Check GPU utilization: `nvidia-smi`
- Verify Flash Attention is loaded (check logs for "Using Flash Attention 2")
- Check if data loading is bottleneck (should be fast)

### Flash Attention Issues
If Flash Attention fails to load, edit `train_direct.py`:
```python
attn_implementation="eager"  # Instead of "flash_attention_2"
```

### Dataset Not Found
Ensure you ran `prepare_dataset.py` and `dataset/all_data.jsonl` exists:
```bash
python3 prepare_dataset.py
ls -lh dataset/all_data.jsonl
```

### Wandb Not Logging Metrics
Check if wandb is properly configured:
```bash
# Verify API key is set
echo $WANDB_API_KEY

# Or check config.yaml
cat config.yaml | grep -A 5 "wandb:"

# Test wandb login
wandb login
```

### Module Import Errors
Reinstall requirements:
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

## References

- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)