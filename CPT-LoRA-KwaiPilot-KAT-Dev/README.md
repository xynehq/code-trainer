# Hyperswitch CPT Training - KAT-Dev-32B

Clean, direct implementation using Transformers + TRL + PEFT + DeepSpeed.

## Overview

- **Model**: Kwaipilot/KAT-Dev (32B instruct model)
- **Method**: LoRA (Low-Rank Adaptation, r=64, alpha=128)
- **Hardware**: 2x H200 GPUs
- **Dataset**: Hyperswitch Rust codebase (~16,731 samples)
- **Training**: Causal Language Modeling (next-token prediction)

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
# In a separate terminal - TensorBoard
cd /workspace/RL_Code/CPT_Kwai
source .venv/bin/activate
nohup tensorboard --logdir outputs --port 6006 --bind_all --reload_interval 5 > tensorboard.log 2>&1 &

# Access at http://localhost:6006

# Or monitor console logs
tail -f train_log.txt

# Or watch GPU usage
watch -n 1 nvidia-smi
```

## Configuration

### Model Configuration (`config.yaml`)
- Model: `Kwaipilot/KAT-Dev` (32B instruct)
- Repository: Path to Hyperswitch codebase
- Dataset: Output directory and token limits
- Training: Epochs, batch sizes, learning rate

### Training Hyperparameters
- **LoRA**: r=64, alpha=128, dropout=0.05
- **Batch size**: 2 micro × 8 grad_accum × 2 GPUs = 32 effective
- **Learning rate**: 5e-5 with cosine schedule, warmup=0.03
- **Epochs**: 5
- **Context length**: 8,192 tokens
- **Sample packing**: Disabled
- **Optimizer**: AdamW with weight_decay=0.1

### DeepSpeed Configuration (`deepspeed_configs/zero2.json`)
- ZeRO stage 2 (optimizer state sharding)
- BF16 precision
- No CPU offloading
- Flash Attention 2 enabled

## File Structure

```
CPT_Kwai/
├── config.yaml                  # Main configuration
├── train_direct.py              # Training script (Transformers + TRL + PEFT)
├── train_direct.sh              # Training launcher with monitoring
├── prepare_dataset.py           # Dataset preparation from Hyperswitch
├── inference.py                 # Model comparison tool
├── inference_with_instructions.py
├── requirements.txt             # Python dependencies
├── evaluation_questions_cpt.json
├── deepspeed_configs/           # DeepSpeed configurations
│   └── zero2.json
├── dataset/                     # Training data (generated, ~36MB)
│   ├── all_data.jsonl          # 16,731 combined samples
│   └── dataset_stats.json      # Statistics
├── hyperswitch/                 # Cloned repo (not pushed to git)
└── outputs/                     # Training outputs (not pushed to git)
    └── hyperswitch-kat-dev-lora_*/
        ├── logs/               # TensorBoard logs
        ├── checkpoint-*/       # Saved checkpoints
        └── final_model/        # Final trained model
```

## What Gets Pushed to GitHub

**Essential files (~144KB total):**
- ✅ `config.yaml`, `requirements.txt`
- ✅ `prepare_dataset.py`, `train_direct.py`, `train_direct.sh`
- ✅ `inference.py`, `inference_with_instructions.py`
- ✅ `deepspeed_configs/`, `evaluation_questions_cpt.json`
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

### Memory Usage (per GPU)
- Model: ~62GB (32B params in BF16)
- LoRA adapters: ~1.2GB (r=64)
- Optimizer: ~31GB (ZeRO-2 split)
- Activations: ~24GB (with gradient checkpointing)
- **Total: ~78GB / 141GB available per H200**

### Expected Training Time
- Total steps: 2,355 (5 epochs)
- ~5-6 seconds per step
- **Total: ~3.5-4 hours** on 2x H200

### Training Results (Example Run)
- **Initial loss**: 1.63 → **Final loss**: 0.48 (71% reduction)
- **Initial perplexity**: 5.12 → **Final perplexity**: 1.59
- **Token accuracy**: 61% → 89%
- **Eval loss**: 0.46 (no overfitting)

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
  micro_batch_size: 1        # Was 2
  gradient_accumulation_steps: 16  # Was 8
```

### Loss is NaN
Check learning rate in `config.yaml`:
```yaml
training:
  learning_rate: 0.00003     # Try lower (was 5e-5)
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

### TensorBoard Not Showing Metrics
Make sure TensorBoard is running and pointing to correct directory:
```bash
# Stop existing instances
pkill -f "tensorboard.*port 6006"

# Start fresh
cd /workspace/RL_Code/CPT_Kwai
source .venv/bin/activate
tensorboard --logdir outputs --port 6006 --bind_all
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
