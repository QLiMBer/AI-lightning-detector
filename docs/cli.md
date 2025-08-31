# CLI Reference

## Command

- `lightning-detector scan`: scan a directory of `.mp4` files and write per‑video reports.

## Options (common first)

- `--input DIR`: directory with `.mp4` files to analyze. Default: `videos/`.
- `--output DIR`: where outputs are written. Produces `<name>.json`, `<name>.txt`, and `index.txt`. Default: `reports/`.
- `--fps INT` (default: 2): frames per second to sample before packing. Lower = faster/less coverage; higher = more coverage/VRAM.
- `--packing INT` (default: 0): temporal packing level for 3D‑Resampler. `0` disables; `1–6` packs consecutive frames to increase coverage.
- `--max-frames INT` (default: 0): hard cap on frames sent to the model per video (after sampling/packing). `0` means unlimited (scan the whole video at the chosen `--fps`). For long videos, consider keeping a cap or enabling `--packing` to control time/VRAM.
- `--max-slice-nums INT` (default: 1): split hi‑res frames into slices to reduce VRAM use. Increase if you hit OOM (e.g., `2` or `3`).
- `--attn {sdpa,flash_attention_2,eager}` (default: sdpa): attention backend. `sdpa` uses PyTorch’s scaled dot‑product attention (stable, built‑in). `flash_attention_2` uses custom CUDA kernels for speed but requires an exact‑match wheel for your Torch/CUDA. `eager` is a fallback (slower).
- `--dtype {bfloat16,float16,float32}` (default: float16): numeric precision for compute. `float16` (FP16) is compatible on most GPUs and fast. `bfloat16` (BF16) can be more numerically stable on newer GPUs (e.g., Ampere+ with BF16) with similar speed. `float32` is highest precision but slowest and uses most VRAM — helpful for debugging.
- `--thinking`: enable deeper reasoning mode (higher latency).
- `--no-preload-model`: skip upfront model load; the model initializes lazily before the first video. Useful to surface download/progress.

## Examples

- Minimal, fast smoke (no flags needed):
  - `lightning-detector scan`
- Slightly more coverage:
  - `lightning-detector scan --fps 3 --max-frames 48`
- Longer clips, more coverage:
  - `lightning-detector scan --fps 3 --packing 3 --max-frames 48 --max-slice-nums 2`
