# Fine-Tuning Guide

Train a small instruct model (default: **Qwen2.5-1.5B-Instruct**) to emit the agent’s JSON schema from `build_prompt(record)` inputs.

Two supported paths:

| Path | Best for |
|------|----------|
| **[Local GPU (Windows)](#local-gpu-windows)** | GTX 1660 Ti / 6 GB VRAM dev machine |
| **[Google Colab](#google-colab)** | Faster training, production uploads to Hugging Face |

Production deployment (HF Inference Endpoint → Vercel): **[HOSTING.md](HOSTING.md)**

---

## Prerequisites

### Dataset

Labeled JSONL with hold-out `expected` fields (`data/test_cases.jsonl`, `data/sample.jsonl`).

Export chat JSONL for training:

```bash
python scripts/export_finetune_dataset.py \
  --input data/test_cases.jsonl data/sample.jsonl \
  --output data/finetune/train.jsonl \
  --format openai
```

Optional synthetic expansion:

```bash
python scripts/generate_training_data.py --output-dir data/generated
```

### GPU venv (local training)

Use a **separate** venv from the main app (`.venv-finetune`):

```powershell
cd realpage-message-agent
py -3.14 -m venv .venv-finetune
.\.venv-finetune\Scripts\Activate.ps1

# PyTorch with CUDA (Python 3.14: use cu130, not cu124)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

pip install -r requirements-finetune.txt
pip install -r requirements.txt
```

**Driver requirement:** PyTorch cu130 needs a recent NVIDIA driver (610+). Update drivers if `torch.cuda.is_available()` is `False`.

**Windows note:** Use PowerShell backticks for line continuation, not `\`.

---

## Local GPU (Windows)

### 1. Train LoRA

```powershell
cd realpage-message-agent
.\.venv-finetune\Scripts\Activate.ps1

python scripts/train_lora.py `
  --train-file data/finetune/train.jsonl `
  --output-dir models/realpage-message-agent-v1
```

Defaults: fp16 LoRA, 5 epochs, batch 1 × grad accum 8, `max_length=1536`, merged weights saved to `models/realpage-message-agent-v1/`.

**6 GB VRAM:** Run **one** of `train_lora.py` or `serve_local_model.py` at a time.

### 2. Verify CUDA

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 3. Serve locally (OpenAI-compatible)

```powershell
# Terminal 1
python scripts/serve_local_model.py --model-dir models/realpage-message-agent-v1
```

Server: `http://127.0.0.1:8000/v1`

If port 8000 is in use:

```powershell
Stop-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess -Force
# or
python scripts/serve_local_model.py --port 8001
```

### 4. Configure `.env`

```env
LLM_PROVIDER=local
LOCAL_BASE_URL=http://127.0.0.1:8000/v1
LOCAL_MODEL=realpage-message-agent-v1
LOCAL_API_KEY=local
LOCAL_JSON_MODE=true
LOCAL_MAX_TOKENS=512
```

### 5. Evaluate

```powershell
# Terminal 2 (server must be running)
python scripts/eval_lora.py --input data/test_cases.jsonl --output outputs/lora_eval.jsonl
```

Writes `outputs/lora_eval.summary.json` with per-`task_id` pass/fail vs gold labels.

**Baseline expectation:** ~22 labeled examples often yields **2/20** strict passes on `test_cases.jsonl` (suppress cases learn first). Add more labeled data before expecting production-quality eval scores.

Pipeline-only check without GPU:

```powershell
python scripts/eval_lora.py --mock
```

---

## Google Colab

1. Export `data/finetune/train.jsonl` (see above).
2. Open `notebooks/train_lora_colab.ipynb` (T4+ runtime).
3. Upload training JSONL when prompted; run through merge + smoke test.
4. Download merged weights → `models/realpage-message-agent-v1/`.
5. Verify: `python scripts/verify_model_export.py --model-dir models/realpage-message-agent-v1`
6. Upload to Hub: `python scripts/upload_model_hub.py ...` — see **[HOSTING.md](HOSTING.md)**.

---

## Scripts reference

| Script | Purpose |
|--------|---------|
| `scripts/export_finetune_dataset.py` | Labeled JSONL → chat training JSONL |
| `scripts/generate_training_data.py` | Synthetic labeled records |
| `scripts/train_lora.py` | Local GPU LoRA + merge |
| `scripts/serve_local_model.py` | Dev OpenAI-compatible server |
| `scripts/eval_lora.py` | Labeled eval + summary JSON |
| `scripts/verify_model_export.py` | Pre-upload weight check |
| `scripts/upload_model_hub.py` | Push merged model to Hub |
| `scripts/test_remote_model.py` | Single-record HF endpoint smoke test |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `pip install torch` fails on cu124 + Python 3.14 | Use `--index-url https://download.pytorch.org/whl/cu130` |
| PowerShell `Directory '\\' is not installable` | Don't use `\` line continuation; use one line or `` ` `` |
| NVIDIA installer fails (disk full) | Free **10+ GB** on `C:` before retrying driver update |
| `cuda available False` after driver install | Reboot; verify with `nvidia-smi` (610+) |
| Port 8000 already in use | Kill old `serve_local_model.py` process |
| `Activate.ps1` not found | `cd realpage-message-agent` first (not parent `RealPage` folder) |
| Eval very slow | ~30–60 s/record on GTX 1660 Ti; normal |
| Low eval pass rate | Add labeled examples; re-export; retrain |

---

## What not to commit

Model weights (`models/`, `*.zip`) and secrets (`.env`) stay local. See `.gitignore`.
