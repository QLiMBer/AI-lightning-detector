Setup — MiniCPM‑V 4.5 (Local)

- Python: 3.10+ recommended
- PyTorch: CUDA build matching your driver (RTX 3090; BF16 and/or FP16)
- Transformers: 4.44.2 (per upstream note for compatibility)
- Key Python deps: `transformers`, `torch`, `accelerate` (optional), `decord` (video), `Pillow`, `numpy`, `scipy`
- Optional: `flash-attn` if you want FlashAttention2; else use `attn_implementation='sdpa'`

Install (baseline)

- Ensure a working CUDA PyTorch:
  - Visit https://pytorch.org/get-started/locally/ for the exact pip command for your CUDA.
- Install libs:
  - `pip install transformers==4.44.2 pillow numpy scipy decord`
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
    torch_dtype=torch.bfloat16,
).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
```

Notes

- If BF16 causes issues in your environment, try `torch_dtype=torch.float16`.
- For large/long videos, use 3D‑Resampler (temporal packing) and/or limit sampled frames to avoid OOM.
- The full model uses ~18 GB VRAM; 3090 (24 GB) is suitable.
