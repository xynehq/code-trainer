# GLM-4.5-Air QLoRA Training Kit

End-to-end toolkit for adapting the 100B+ parameter GLM-4.5-Air mixture-of-experts model with QLoRA, curated launch scripts, and a real-time dashboard. This folder is self-contained: download the base checkpoint, edit one YAML config, start the multi-GPU training loop, and monitor progress from any browser.

---

## Repository Layout

```
code-trainer/CPT-GLM-4.5-AIR-QLoRA/
├── download_model.py             # Pulls zai-org/GLM-4.6 into /workspace/Avinash/models
├── training_config.yaml          # Single source of truth for paths + hyperparameters
├── train_qlora.py                # Accelerate-powered trainer with logging/resume logic
├── launch_training.sh            # 4×H200 friendly wrapper around accelerate launch
├── training_dashboard.py         # Flask backend that reads the emitted JSON logs
├── launch_dashboard.sh           # Convenience launcher (gunicorn if available)
├── templates/dashboard_enhanced.html
└── DASHBOARD_ENHANCED_README.md  # Standalone dashboard tour
```

---

## Prerequisites

- **Hardware**: 4×H200 (140 GB) or comparable GPUs; adjust `CUDA_VISIBLE_DEVICES` if you scale up/down (`launch_training.sh`:12–23).
- **Software**: Python ≥3.10 with CUDA-enabled PyTorch, `accelerate`, `transformers`, `datasets`, `bitsandbytes`, `peft`, `flask`, `matplotlib`, `numpy`, and `huggingface_hub`.
- **Accounts**: Hugging Face access to `zai-org/GLM-4.5-Air`. Run `huggingface-cli login` before downloading.
- **Storage**: ≥200 GB free space for model weights + checkpoints.

You can keep dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U accelerate transformers datasets bitsandbytes peft flask matplotlib numpy huggingface_hub
```

Configure Accelerate for your topology once:

```bash
accelerate config  # choose multi-GPU, DeepSpeed ZeRO-3, etc.
```

---

## Workflow at a Glance

1. **Download the base model**
   ```bash
   cd /workspace/Avinash/code-trainer/CPT-GLM-4.5-AIR-QLoRA
   python3 download_model.py
   ```
   `download_model.py` pulls `zai-org/GLM-4.5-Air` with `snapshot_download`, resumes partial transfers, verifies key files, and reports the disk footprint (`download_model.py`:7-68).

2. **Prepare your dataset**
   - Expect a JSONL file with a top-level `text` field per row.
   - Update `paths.data_path` if your dataset lives elsewhere (`training_config.yaml`:5-17).

3. **Tweak training_config.yaml**
   - `paths`: source / output directories and best-adapter target.
   - `training`: epochs, seed, max seq length, and train/validation split.
   - `optimization`: micro batch, grad accumulation, LR schedule, Adam betas/eps.
   - `lora`: rank, alpha, dropout, and target modules for Q/K/V/O projections.
   - `quantization`: BitsAndBytes NF4 4-bit knobs.
   - `logging` / `memory` / `performance`: cadence for logs/evals/checkpoints plus toggles such as TF32 and pinning (`training_config.yaml`:18-62).

4. **Launch training**
   ```bash
   chmod +x launch_training.sh
   ./launch_training.sh    # auto-detects checkpoints and resumes if present
   ```
   The launcher seeds temp/cache locations, exports NCCL + CUDA env vars, validates the model/dataset paths, prints `nvidia-smi`, and finally calls `accelerate launch` with four processes and your config file (`launch_training.sh`:7-139). Set `RESUME_FLAG` manually via `--resume_from /path/to/checkpoint-XXXX` if you want a specific checkpoint.

   Manual invocation (for notebooks or SLURM jobs):
   ```bash
   accelerate launch \
     --num_processes 4 \
     train_qlora.py \
     --config training_config.yaml \
     --resume  # optional
   ```

5. **Monitor in real time**
   ```bash
   chmod +x launch_dashboard.sh
   ./launch_dashboard.sh
   # Browse http://localhost:5000 or http://<server-ip>:5000
   ```
   The dashboard reads JSONL logs only—no impact on training—and serves the enhanced UI defined under `templates/` (`launch_dashboard.sh`:3-43, `training_dashboard.py`:1-200).

---

## Training Pipeline Details

### Data ingestion & preprocessing
- `train_qlora.py` uses `datasets.load_dataset` with your JSONL file, performs a deterministic 90/10 split, and tokenizes with left-padding up to 4096 tokens (`train_qlora.py`:420-455).
- Training/validation DataLoaders pin memory, shuffle the training split, and default to worker count 0 to avoid over-spawning per process (`train_qlora.py`:488-512).

### Model loading & adaptation
- Loads GLM-4.5-Air with BitsAndBytes NF4 4-bit quantization plus double quantization for QLoRA efficiency (`train_qlora.py`:456-470).
- Applies `prepare_model_for_kbit_training` and disables `use_cache` so gradient checkpointing works with multiple GPUs (`train_qlora.py`:471-481).
- Builds a LoRA config that targets Q/K/V/O projections across all transformer layers, honoring the YAML values (`train_qlora.py`:126-143).

### Optimization loop
- Accelerator handles gradient accumulation and distributed sync; effective batch = micro batch × grad_accum × number of processes (printed at startup).
- The training loop logs loss, LR, steps/sec, ETA, gradient norm, auxiliary loss (if provided by the model), and router stats for the MoE experts (`train_qlora.py`:512-760). Expert router usage snapshots are appended every 100 steps (`train_qlora.py`:61-124).
- Validation runs every `logging.eval_interval` steps using a no-grad helper (`train_qlora.py`:153-171) and writes to `glm45-air-cpt-qlora/logs/eval_log.jsonl`.
- Checkpoints drop every `logging.checkpoint_interval` steps. Optimizer/scheduler states are saved to `training_state.pt`, the best checkpoint is tracked by validation loss, and only the top-K checkpoints are retained (`train_qlora.py`:173-261, 610-742).
- At the end of training, the best adapter is copied to `paths.best_adapter_dir` and a final checkpoint bundle is stored under `final-checkpoint-<global_step>` (`train_qlora.py`:191-204, 742-812).

### Resumption & safety
- `--resume` auto-discovers the latest `checkpoint-*` folder; `--resume_from` accepts a path. When resuming, the optimizer LR is restored by parsing historical logs so the scheduler keeps continuity (`train_qlora.py`:225-305, 520-612).
- Fatal exceptions bubble up through the `main()` guard, log a structured entry to `glm45-air-cpt-qlora/logs/error_log.jsonl`, and re-raise for visibility (`train_qlora.py`:816-859).

---

## Outputs & Artifacts

| Path | Description |
| --- | --- |
| `glm45-air-cpt-qlora*/checkpoint-*/` | Serialized adapters + tokenizer for each checkpoint. |
| `best-glm45-adapter/` | Copy of the best-performing checkpoint (overwritten every improvement). |
| `glm45-air-cpt-qlora/logs/training_log.jsonl` | Per-step training metrics consumed by the dashboard. |
| `glm45-air-cpt-qlora/logs/eval_log.jsonl` | Validation loss & perplexity snapshots. |
| `glm45-air-cpt-qlora/logs/expert_usage_log.jsonl` | Router statistics for MoE experts. |
| `glm45-air-cpt-qlora/logs/error_log.jsonl` | Fatal errors captured by `log_error`. |

> **Tip:** `training_config.yaml` currently points at `glm45-air-cpt-qlora-2` while the launcher defaults to `glm45-air-cpt-qlora`. Align these if you want the dashboard and auto-resume logic to pick up the same directory.

---

## Dashboard Highlights

- **Backend (`training_dashboard.py`)**: Flask app that loads only the most recent run, renders plots via Matplotlib's Agg backend, computes moving averages, decorates plots with checkpoint markers, and exposes JSON endpoints for stats, plots, checkpoints, events, GPU telemetry, and errors (`training_dashboard.py`:1-400).
- **Frontend (`templates/dashboard_enhanced.html`)**: Responsive dashboard with progress bars, stat cards, validation highlights, event timeline, checkpoint table, GPU cards, and auto-refresh behavior (HTML/CSS/JS only—no build step).
- **Launch flow**: `launch_dashboard.sh` ensures Flask is available, prefers gunicorn, and prints both localhost and LAN URLs for quick sharing.
- **Further reading**: `DASHBOARD_ENHANCED_README.md` dives into every widget and API if you need to extend or embed it elsewhere.

---

## Dataset Notes

- Expected schema: each line in your JSONL file must have at least a `text` key. Add any preprocessing (filters, deduplication, safety scrubs) before pointing the config at it.
- The script tokenizes with `padding="max_length"` and `truncation=True`, so extremely long samples will be clipped at `training.max_seq_length`.
- If you need instruction-style formatting, implement it upstream (e.g., templating) so the JSON already reflects the desired prompt/response pairs.

---

## Troubleshooting & Tips

- **Model download fails**: ensure your Hugging Face token has access, verify disk space, and rerun `python download_model.py` (it resumes automatically).
- **OOM during warm-up**: lower `optimization.micro_batch_per_gpu` or `lora.r` in the YAML; the launcher already enables gradient checkpointing and NF4 quantization.
- **Mismatched paths**: keep `paths.output_dir`, `CHECKPOINT_DIR` in `launch_training.sh`, and the dashboard constants in sync so resume + monitoring work out of the box.
- **Dashboard shows no data**: confirm `glm45-air-cpt-qlora/logs/training_log.jsonl` exists and that your web browser can reach port 5000; the `/health` endpoint is a quick ping.
- **Exporting adapters**: copy `best-glm45-adapter` into your inference repo or push to the Hub; it contains only the PEFT weights, so pair it with the original GLM base model.

---

## Next Steps

- Integrate the `launch_training.sh` command into your job scheduler (SLURM, RunPod, Modal, etc.).
- Extend `training_dashboard.py` if you need alerts (Slack/Webhook) or additional metrics—it's already modular via the `/api/*` endpoints.
- Use the produced adapters for evaluation scripts under `Avinash/GLM-4.5-Air-HS` or deploy them with your inference stack.

Happy training!
