# User Guide

This guide covers day‑to‑day usage, common recipes, outputs, and practical tips.

## Daily Usage

- Activate the virtualenv in each new shell:
  - `source .venv/bin/activate`
- Run a scan (defaults tuned for common cases):
  - `lightning-detector scan`
- Without activation, use the full path:
  - `.venv/bin/lightning-detector scan`

## Common Recipes

- More coverage (slightly slower):
  - `lightning-detector scan --fps 3`
- Long clips on limited VRAM: temporal packing + slice images:
  - `lightning-detector scan --packing 3 --max-slice-nums 2`
- Cap runtime on very long videos (0 = unlimited):
  - `lightning-detector scan --max-frames 64`
- Stable shapes to avoid size mismatches (default = 448):
  - `lightning-detector scan --image-size 448`
- Reproducibility: pin model code revision from Hugging Face:
  - `lightning-detector scan --model-revision <commit-or-tag>`

See `docs/cli.md` for the full list of options and defaults.

## Outputs

- Reports directory (default `reports/`):
  - `<video>.json`: parsed detections with metadata (video path, fps_sampled, method, model).
  - `<video>.txt`: raw model output (for debugging).
  - `index.txt`: one line per JSON file with detection counts.
  - `results.txt`: aggregated summaries written per run.

## Project Folders

- `videos/`: input `.mp4` files you provide (gitignored).
- `reports/`: per‑video outputs (gitignored).
- `lightning_detector/`: CLI, video I/O, and model integration.
- `scripts/`: utilities like `inspect_videos.py` and `smoke_chat.py`.

## Performance & VRAM Tips

- Start with defaults: `--fps 2`, `--packing 0`, `--max-frames 0` (unlimited), `--max-slice-nums 1`, `--image-size 448`.
- Increase `--fps` to improve recall; consider `--packing` for multi‑minute clips.
- If you hit CUDA OOM, try `--max-slice-nums 2` and/or reduce `--fps`.
- Attention backend: prefer `--attn sdpa`. Use `flash_attention_2` only if you installed a matching wheel.
- Precision: `--dtype float16` is fast and broadly compatible; `bfloat16` for newer GPUs; `float32` for debugging.

## Restart Tips

- After reboot or a new terminal session, reactivate the env:
  - `source .venv/bin/activate`
- If the `lightning-detector` command is missing, reinstall the console script in editable mode:
  - `uv pip install --prefix .venv -e .`
- To use a local model cache (optional), prefix commands:
  - `HF_HOME=.hf-cache TRANSFORMERS_CACHE=.hf-cache/hub lightning-detector scan`
- If the first run stalls (cache locks), check and clear stale processes/locks (see `docs/troubleshooting.md`).

