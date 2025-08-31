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
- Run a minimal scan (defaults tuned for common cases):
  - `lightning-detector scan`
  - Common tweaks:
    - Increase coverage: `--fps 3`
    - Control runtime/VRAM on long clips: `--packing 3 --max-slice-nums 2`
    - Cap frames (0 = unlimited): `--max-frames 64`

 Outputs are written to `reports/` (`<name>.json`, `<name>.txt`, `index.txt`).

## Daily Usage (after reboot)

- Activate the environment (each new shell):
  - `source .venv/bin/activate`
- Run the detector:
  - `lightning-detector scan` (add flags as needed)
- Without activating the venv, you can still run via full path:
  - `.venv/bin/lightning-detector scan`

## About `uv pip install --prefix .venv -e .`

- Purpose: installs this project into the virtualenv in “editable” mode so that the `lightning-detector` console command is created under `.venv/bin`, while your code edits are picked up immediately without reinstalling.
- When to run:
  - First-time setup (after creating `.venv`).
  - After recreating or deleting `.venv`.
  - After dependency changes (`pyproject.toml` or `uv.lock`) — typically run `uv sync` first, then re-run the editable install if the entry point is missing.
  - Not needed for ordinary code edits; editable mode reflects changes automatically.
- If run without `source .venv/bin/activate`:
  - It still installs into `.venv` because of `--prefix .venv`.
  - The console script lands in `.venv/bin/`. Use `.venv/bin/lightning-detector` or activate the venv to get it on `PATH`.

## Next Steps

- User Guide (recipes, outputs, tips): `docs/user-guide.md`
- CLI Reference (all options): `docs/cli.md`
- Troubleshooting: `docs/troubleshooting.md`

## Troubleshooting

- First‑run tips, cache/locks, and CUDA placement guidance: `docs/troubleshooting.md`

## Development Notes

- Source: `lightning_detector/`
- Scripts: `scripts/` (e.g., `scripts/smoke_chat.py`, `scripts/inspect_videos.py`)
- Docs index: `docs/README.md`
