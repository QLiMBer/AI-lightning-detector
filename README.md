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

## Stable vs Latest (Reproducibility)

Two practical workflows, pick what you need today:

- Stable (pinned snapshot; reliable for day‑to‑day work):
  - Find a known‑good revision (commit SHA) in your HF cache: `ls ~/.cache/huggingface/hub/models--openbmb--MiniCPM-V-4_5/snapshots`
  - Force offline to avoid surprise updates and run the pinned snapshot:
    - `export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1`
    - `lightning-detector scan --model-revision <commit-sha> --attn sdpa --dtype float16`

- Latest (track upstream changes; expect occasional breakage):
  - Do NOT set the offline env vars. Omit `--model-revision` so Transformers pulls the newest remote code.
  - If a breaking change appears (e.g., new flash‑attn requirement), switch back to the Stable workflow above.

Minimal reproducible run (stable example)

- `export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1`
- `lightning-detector scan --model-revision a8dd5db4715809f904dbf39c2a98a6112033f0f1 --attn sdpa --dtype float16 --fps 2 --packing 0`

Notes

- Some upstream revisions hard‑require FlashAttention2. If you see errors about `flash_attn` even with `--attn sdpa`, use the Stable workflow (pin + offline) or install a matching `flash-attn` wheel for your Torch/CUDA.
- Certain snapshots need a custom Processor; the CLI now constructs and passes it automatically. If you’re on an older checkout, pull latest from this repo.

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
- Flash‑attn vs SDPA, processor mismatches, and reproducibility: `docs/troubleshooting.md` (sections: Upstream revision requires flash‑attn; Processor/AutoProcessor mismatch)

## Contributing Workflow (Branches)

- `main`: stable for users; docs and code tested against a pinned MiniCPM‑V revision noted in commits/PRs. Prefer this branch for demos and day‑to‑day runs.
- `edge` (or `dev`): tracks latest upstream model code; used to validate changes against the newest snapshots. Expect occasional breakage and quick fixes.

Recommended practice

- When developing features: iterate on `edge` against the latest model; if issues arise (e.g., flash‑attn hard‑requirement), fall back to `main` and continue using the pinned snapshot.
- When cutting a stable point: merge fixes to `main` and update docs with the verified `--model-revision` SHA.

See `docs/roadmap.md` for current obstacles and planned work to support the latest snapshots by default.

## Development Notes

- Source: `lightning_detector/`
- Scripts: `scripts/` (e.g., `scripts/smoke_chat.py`, `scripts/inspect_videos.py`)
- Docs index: `docs/README.md`
