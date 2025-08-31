Session Restart Checklist

Use this to quickly resume work in a fresh session.

1) Activate the environment

- `source .venv/bin/activate`

3) Recreate the Python environment with uv

- Install uv (once per machine):
  - `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Ensure `~/.local/bin` is on your PATH.
- Sync base dependencies from the lockfile into a venv:
  - `uv lock`
  - `uv sync`
- Install PyTorch with CUDA using uv’s backend selector (CUDA 12.4 recommended for your 12.8 driver):
  - `uv pip install --prefix .venv --torch-backend cu124 torch torchvision torchaudio`
- Activate and verify GPU:
  - `source .venv/bin/activate`
  - `python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"`
  - Expected: `True 12.4`

2) Quick sanity checks

- Inspect videos:
  - `python scripts/inspect_videos.py`
  - Confirms fps/resolution/duration are read correctly.
 - Optional: try the minimal model smoke test: `python scripts/smoke_chat.py --attn sdpa --dtype float16`

Troubleshooting first run (model downloads)

- If you see `cannot import name 'Qwen3Config'`: upgrade to `transformers>=4.47,<5` and re‑sync.
- To force a clean download and avoid global cache issues, run the CLI with a repo‑local cache and skip preload:
  - `HF_HOME=.hf-cache TRANSFORMERS_CACHE=.hf-cache/hub lightning-detector scan --input videos --output reports --fps 1 --max-frames 16 --max-slice-nums 1 --attn sdpa --dtype float16 --no-preload-model`
- Avoid suspending runs (Ctrl‑Z). If you did, clean them up: `jobs -l` then `kill -TERM %<id>`.

3) Folder expectations

- Input videos: `videos/`
- Outputs (once implemented): `reports/` (JSON + text summaries)

4) Run the detector

- Example: `lightning-detector scan --input videos --output reports --fps 1 --max-frames 16 --max-slice-nums 1 --attn sdpa --dtype float16 --no-preload-model`
