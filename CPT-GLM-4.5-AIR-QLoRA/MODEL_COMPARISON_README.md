# Model Comparison Tool - Documentation

Compare your fine-tuned GLM-4.5-Air checkpoints with the base model to detect overfitting and measure adaptation quality.

---

## Quick Start

### 1. Activate Virtual Environment
```bash
source /workspace/Avinash/ashenv/bin/activate
cd /workspace/Avinash/code-trainer/CPT-GLM-4.5-AIR-QLoRA
```

### 2. Run Comparison (Easy Way)
```bash
bash compare_checkpoint.sh
```

This will compare `checkpoint-1000` with the base model using default test prompts.

---

## Usage Examples

### Compare Specific Checkpoint
```bash
# Compare checkpoint-2000
bash compare_checkpoint.sh glm45-air-cpt-qlora/checkpoint-2000

# Compare and save results to custom file
bash compare_checkpoint.sh glm45-air-cpt-qlora/checkpoint-3000 results_3000.json
```

### Using Python Script Directly

#### Compare with Custom Prompts
```bash
python3 compare_models.py \
    --checkpoint glm45-air-cpt-qlora/checkpoint-1000 \
    --prompts "Explain payment gateways" "Write a sorting algorithm"
```

#### Use Prompts from File
```bash
python3 compare_models.py \
    --checkpoint glm45-air-cpt-qlora/checkpoint-1000 \
    --prompts_file test_prompts.json \
    --save_results my_comparison.json
```

#### Compare Multiple Checkpoints
```bash
# Loop through all checkpoints
for checkpoint in glm45-air-cpt-qlora/checkpoint-*; do
    echo "Testing $checkpoint"
    python3 compare_models.py \
        --checkpoint "$checkpoint" \
        --prompts_file test_prompts.json \
        --save_results "results_$(basename $checkpoint).json"
done
```

#### Test Base Model Only
```bash
python3 compare_models.py --base_only --prompts "Hello, how are you?"
```

---

## Command Line Options

### `compare_models.py` Arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--base_model` | Path to base GLM-4.5-Air model | `/workspace/Avinash/models/GLM-4.5-Air` |
| `--checkpoint` | Path to checkpoint directory | `glm45-air-cpt-qlora/checkpoint-1000` |
| `--prompts` | Space-separated prompts to test | Default test prompts |
| `--prompts_file` | JSON file with prompts array | `test_prompts.json` |
| `--max_tokens` | Maximum tokens to generate | `256` |
| `--save_results` | Path to save JSON results | None (print only) |
| `--base_only` | Test only base model | `False` |

---

## Understanding the Output

### Console Output

For each prompt, you'll see:

```
💬 Prompt: [Your question]
======================================================================

🔵 BASE MODEL Response:
----------------------------------------------------------------------
[Base model's answer]
⏱️  Generation time: X.XXs

🟢 FINE-TUNED MODEL Response:
----------------------------------------------------------------------
[Fine-tuned model's answer]
⏱️  Generation time: X.XXs

📊 Comparison Metrics:
----------------------------------------------------------------------
Response Length:
  Base:       XXX chars
  Fine-tuned: XXX chars
  Difference: XX chars (X.X%)

Word-level Similarity:
  Jaccard Index: XX.X%
  Overlap: XX/XX unique words

Generation Speed:
  Base:       X.XXs
  Fine-tuned: X.XXs
  Speedup:    +X.X%
```

### Overall Summary

After all prompts, you'll see:

```
📈 OVERALL SUMMARY
======================================================================

Average Metrics (across N prompts):
  Jaccard Similarity: XX.X%
  Length Difference:  XX.X%
  Speed Change:       +X.X%

🔍 Overfitting Assessment:
  [Interpretation of results]
```

### Overfitting Interpretation

- **Very high similarity (>80%)**: Minimal adaptation - model barely changed
- **Moderate similarity (50-80%)**: ✅ Good adaptation - healthy fine-tuning
- **Low similarity (20-50%)**: Significant deviation - model specialized heavily
- **Very low similarity (<20%)**: ❌ Possible overfitting - model memorized training data

---

## Metrics Explained

### 1. Jaccard Index (Word-level Similarity)
- Measures overlap between unique words in responses
- Formula: `(common words) / (total unique words) × 100`
- **Higher** = More similar responses
- **Lower** = More different responses

### 2. Response Length Difference
- Compares character count between responses
- Shows if fine-tuned model is more/less verbose
- Calculated as percentage of base model length

### 3. Generation Speed
- Time taken to generate response
- Positive speedup = Fine-tuned model is faster
- Negative speedup = Fine-tuned model is slower

---

## Creating Custom Test Prompts

Edit `test_prompts.json`:

```json
{
  "prompts": [
    "Your custom question 1",
    "Your custom question 2",
    "Domain-specific query relevant to your training data"
  ]
}
```

**Tips for Good Test Prompts:**
- Include prompts from your training domain
- Include general knowledge prompts (to test overfitting)
- Mix easy and complex questions
- Test edge cases and unusual queries

---

## Saved Results Format

The `--save_results` option creates a JSON file:

```json
{
  "timestamp": "2025-11-18T05:00:00",
  "base_model": "/workspace/Avinash/models/GLM-4.5-Air",
  "checkpoint": "glm45-air-cpt-qlora/checkpoint-1000",
  "results": [
    {
      "prompt": "Question here",
      "base_response": "Base model answer",
      "ft_response": "Fine-tuned model answer",
      "base_time": 2.5,
      "ft_time": 2.3,
      "base_length": 150,
      "ft_length": 145,
      "length_diff_pct": 3.3,
      "jaccard_similarity": 65.2,
      "speedup_pct": 8.0
    }
  ]
}
```

View results with `jq`:
```bash
cat comparison_results.json | jq .
```

---

## Workflow Recommendations

### During Training

1. **After each checkpoint (1000, 2000, 3000...):**
   ```bash
   bash compare_checkpoint.sh glm45-air-cpt-qlora/checkpoint-XXXX checkpoint_XXXX_comparison.json
   ```

2. **Monitor metrics:**
   - Watch for declining Jaccard similarity (specialization)
   - Check if responses stay coherent
   - Verify domain knowledge improves

### After Training Complete

1. **Compare all checkpoints:**
   ```bash
   for checkpoint in glm45-air-cpt-qlora/checkpoint-*; do
       bash compare_checkpoint.sh "$checkpoint" "$(basename $checkpoint)_results.json"
   done
   ```

2. **Analyze progression:**
   - See how adaptation evolved across training
   - Identify optimal checkpoint (balance between specialization and generalization)
   - Detect if overfitting occurred in later checkpoints

3. **Select best checkpoint:**
   - Use metrics + manual review
   - Consider using earlier checkpoint if later ones overfit
   - Test on held-out validation prompts

---

## Troubleshooting

### Out of Memory
- Both models load into GPU memory simultaneously
- **Solution:** Use smaller `--max_tokens` or run on machine with more VRAM
- **Alternative:** Compare checkpoints one at a time

### Slow Generation
- 4-bit quantized inference is slower than training
- **Expected:** 2-10 seconds per prompt depending on length
- **Normal for 107B parameter model**

### Different Responses Each Time
- Models use sampling by default (`temperature=0.7`)
- For deterministic output, modify script to use `temperature=0` and `do_sample=False`

---

## Advanced Usage

### Batch Compare All Checkpoints
```bash
#!/bin/bash
for checkpoint in glm45-air-cpt-qlora/checkpoint-*; do
    step=$(basename $checkpoint | cut -d'-' -f2)
    echo "=== Checkpoint $step ==="
    python3 compare_models.py \
        --checkpoint "$checkpoint" \
        --prompts_file test_prompts.json \
        --save_results "step_${step}_comparison.json"
done
```

### Domain-Specific Testing
Create `hyperswitch_prompts.json` with payment-specific questions:
```json
{
  "prompts": [
    "How does payment tokenization work?",
    "Explain PCI DSS compliance requirements",
    "What is 3D Secure authentication?"
  ]
}
```

Then run:
```bash
python3 compare_models.py \
    --checkpoint glm45-air-cpt-qlora/checkpoint-1000 \
    --prompts_file hyperswitch_prompts.json
```

---

## Next Steps

1. **Resume Training:**
   ```bash
   ./launch_training.sh
   ```

2. **Monitor with Dashboard:**
   ```bash
   ./launch_dashboard.sh
   ```

3. **Compare Checkpoints Periodically:**
   - After each checkpoint saves
   - To track adaptation progress
   - To detect overfitting early

4. **Final Evaluation:**
   - Compare best checkpoint with base
   - Test on real use cases
   - Deploy if performance improves

---

## Summary

This tool helps you:
- ✅ Detect overfitting in checkpoints
- ✅ Measure how much your model adapted
- ✅ Choose the best checkpoint for deployment
- ✅ Understand trade-offs between specialization and generalization

**Remember:** The goal is not maximum deviation, but **optimal adaptation** to your domain while maintaining general capabilities!
