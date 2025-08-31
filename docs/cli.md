# CLI Reference

## Command

- `lightning-detector scan`: scan a directory of `.mp4` files and write per‑video reports.

## Options (common first)

- `--input DIR`: directory with `.mp4` files to analyze. Default: `videos/`.
- `--output DIR`: where outputs are written. Produces `<name>.json`, `<name>.txt`, and `index.txt`. Default: `reports/`.
- `--fps INT` (default: 2): frames per second to sample before packing. Lower = faster/less coverage; higher = more coverage/VRAM.
- `--packing INT` (default: 0): temporal packing level for 3D‑Resampler. `0` disables; `1–6` packs consecutive frames to increase coverage.
- `--max-frames INT` (default: 32): hard cap on frames sent to the model per video (after sampling/packing).
- `--max-slice-nums INT` (default: 1): split hi‑res frames into slices to reduce VRAM use. Increase if you hit OOM (e.g., `2` or `3`).
- `--attn {sdpa,flash_attention_2,eager}` (default: sdpa): attention backend. Use `flash_attention_2` only with a matching wheel.
- `--dtype {bfloat16,float16,float32}` (default: float16): compute dtype. `float16` is broadly compatible; `bfloat16` on newer GPUs; `float32` for debugging.
- `--thinking`: enable deeper reasoning mode (higher latency).
- `--no-preload-model`: skip upfront model load; the model initializes lazily before the first video. Useful to surface download/progress.

## Examples

- Minimal, fast smoke (no flags needed):
  - `lightning-detector scan`
- Slightly more coverage:
  - `lightning-detector scan --fps 3 --max-frames 48`
- Longer clips, more coverage:
  - `lightning-detector scan --fps 3 --packing 3 --max-frames 48 --max-slice-nums 2`
