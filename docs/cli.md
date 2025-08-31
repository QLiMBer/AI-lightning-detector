# CLI Reference

## Command

- `lightning-detector scan`: scan a directory of `.mp4` files and write per‑video reports.

## Options

- `--input DIR`: directory with `.mp4` files to analyze. Default: `videos/`.
- `--output DIR`: where outputs are written. Produces `<name>.json`, `<name>.txt`, and `index.txt`. Default: `reports/`.
- `--fps INT`: frames per second to sample before packing. Lower = faster/less coverage; higher = more coverage/VRAM. Example: `--fps 1`.
- `--packing INT`: temporal packing level for the 3D‑Resampler. `0` disables; `1–6` packs consecutive frames to increase temporal coverage with similar cost. Example: `--packing 3`.
- `--max-frames INT`: hard cap on frames sent to the model per video (after sampling/packing). Example: `--max-frames 32`.
- `--max-slice-nums INT`: split hi‑res frames into slices to reduce VRAM use. Increase if you hit OOM (e.g., `2` or `3`).
- `--attn {sdpa,flash_attention_2,eager}`: attention backend. `sdpa` is default/stable. Use `flash_attention_2` only with a matching wheel.
- `--dtype {bfloat16,float16,float32}`: compute dtype. Prefer `bfloat16` on Ampere+; use `float16` for widest compatibility; `float32` for debugging.
- `--thinking`: enable deeper reasoning mode (higher latency).
- `--no-preload-model`: skip upfront model load; the model initializes lazily before the first video. Useful to surface download/progress.

## Examples

- Minimal, fast smoke:
  - `lightning-detector scan --input videos --output reports --fps 1 --max-frames 8 --max-slice-nums 1 --attn sdpa --dtype float16 --no-preload-model`
- Longer clips, more coverage:
  - `lightning-detector scan --fps 3 --packing 3 --max-frames 48 --max-slice-nums 2`
