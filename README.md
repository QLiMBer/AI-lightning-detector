# Lightning Detector (MiniCPM‑V 4.5)

Local CLI to detect lightning flashes in `.mp4` videos using MiniCPM‑V 4.5.

## Quick Start

- Create/activate env
  - `uv lock && uv sync`
  - `uv pip install --prefix .venv --torch-backend cu124 torch torchvision torchaudio`
  - `uv pip install --prefix .venv accelerate`
  - `source .venv/bin/activate`
  - Optional allocator: `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Editable install (for local changes):
  - `uv pip install --prefix .venv -e .`
- Put one or more `.mp4` files into `videos/`
- Run a minimal scan:
  - `lightning-detector scan --input videos --output reports --fps 1 --max-frames 8 --max-slice-nums 1 --attn sdpa --dtype float16 --no-preload-model`

Outputs are written to `reports/` (`<name>.json`, `<name>.txt`, `index.txt`).

## CLI Reference

- See `docs/cli.md` for explanations of `--fps`, `--packing`, `--max-frames`, `--max-slice-nums`, `--attn`, `--dtype`, etc.

## Troubleshooting

- First‑run tips, cache/locks, and CUDA placement guidance: `docs/troubleshooting.md`

## Development Notes

- Source: `lightning_detector/`
- Scripts: `scripts/` (e.g., `scripts/smoke_chat.py`, `scripts/inspect_videos.py`)
- Docs index: `docs/README.md`
