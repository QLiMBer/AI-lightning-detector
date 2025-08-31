Setup — MiniCPM‑V 4.5 (Local)

Note: Developer reference for the underlying model. Typical users can skip this section; use the project README (Quick Start) and User Guide.

- Python: 3.10+ recommended
- PyTorch: CUDA build matching your driver (RTX 3090; BF16 and/or FP16)
- Transformers: 4.47–<5 (required for Qwen3Config used by upstream MiniCPM‑V code)
- Key Python deps: `transformers`, `torch`, `accelerate` (optional), `decord` (video), `Pillow`, `numpy`, `scipy`
- Optional: `flash-attn` if you want FlashAttention2; else use `attn_implementation='sdpa'`

Install (baseline)

- Ensure a working CUDA PyTorch:
  - Visit https://pytorch.org/get-started/locally/ for the exact pip command for your CUDA.
-- Install libs:
  - `pip install 'transformers>=4.47,<5' pillow numpy scipy decord`
  - Optional performance: `pip install flash-attn --no-build-isolation` (depends on your env)

Using uv (recommended)

- Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Project‑managed flow (lockfile):
  - `uv lock` then `uv sync` (installs base deps into `.venv/`).
- Install PyTorch with CUDA via uv (idiomatic):
  - `uv pip install --prefix .venv --torch-backend cu124 torch torchvision torchaudio`
  - Verify: `source .venv/bin/activate && python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"`
  - If your runtime differs, swap `cu124` for `cu121`, `cu126`, etc.

Example for your machine

- Your `nvidia-smi` shows CUDA Version 12.8; CUDA 12.4 wheels are compatible. Use the `--torch-backend cu124` command above.

Model loading (GPU BF16)

```python
import torch
from transformers import AutoModel, AutoTokenizer

model_id = 'openbmb/MiniCPM-V-4_5'
model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
    attn_implementation='sdpa',  # or 'flash_attention_2' if installed
    dtype=torch.bfloat16,
).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
```

Notes

- If BF16 causes issues in your environment, try `dtype=torch.float16`.
- For large/long videos, use 3D‑Resampler (temporal packing) and/or limit sampled frames to avoid OOM.
- The full model uses ~18 GB VRAM; 3090 (24 GB) is suitable.

Troubleshooting first run

- Error `cannot import name 'Qwen3Config'`: your Transformers is too old. Upgrade to `transformers>=4.47,<5` and re‑sync (uv: `uv lock && uv sync`).
- Stuck during initial model load or silent stalls: avoid suspended background runs holding HF cache locks. Use `jobs -l` then `kill -TERM %<id>`; remove stale locks under cache if needed.
- Force a fresh download in a local cache and skip preload:
  - `HF_HOME=.hf-cache TRANSFORMERS_CACHE=.hf-cache/hub lightning-detector scan --input videos --output reports --fps 1 --max-frames 16 --max-slice-nums 1 --attn sdpa --dtype float16 --no-preload-model`
  - You should see Hugging Face download progress and CLI "Processing … / Finished …" logs.
