Docs Index

- Quick Start: minimal working setup and run
  - `uv lock && uv sync`
  - `uv pip install --prefix .venv --torch-backend cu124 torch torchvision torchaudio`
  - `uv pip install --prefix .venv accelerate`
  - `source .venv/bin/activate`
  - `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  - Run: `lightning-detector scan --input videos --output reports --fps 1 --max-frames 8 --max-slice-nums 1 --attn sdpa --dtype float16 --no-preload-model`
  - Optional local cache: prefix with `HF_HOME=.hf-cache TRANSFORMERS_CACHE=.hf-cache/hub`

- MiniCPM‑V 4.5 Overview: `docs/minicpm-v/overview.md`
- Setup (Python/CUDA/Deps): `docs/minicpm-v/setup.md`
- Video Inference Usage: `docs/minicpm-v/usage-video.md`
- Deployment Options: `docs/minicpm-v/deployment.md`
- CLI Reference: `docs/cli.md`
- Troubleshooting & First‑Run Notes: `docs/troubleshooting.md`
- References: `docs/references.md`

Utilities

- Inspect video metadata: `scripts/inspect_videos.py` (prints fps, resolution, frames, duration for all `.mp4` in `videos/`)
- Session Restart Checklist: `docs/restart.md`
- Minimal model smoke test: `scripts/smoke_chat.py`
