# CLI Reference

See also: `docs/user-guide.md` for day‑to‑day recipes and outputs.

## Command

- `lightning-detector scan`: scan a directory of `.mp4` files and write per‑video reports.

Note (temporary): until an `edge` branch is maintained against the latest upstream model, prefer pinning a known‑good model revision for stability, e.g.:

- `lightning-detector scan --model-revision a8dd5db4715809f904dbf39c2a98a6112033f0f1`

## Options (common first)

- `--input DIR`: directory with `.mp4` files to analyze. Default: `videos/`.
- `--output DIR`: where outputs are written. Produces `<name>.json`, `<name>.txt`, and `index.txt`. Default: `reports/`.
- `--fps INT` (default: 2): frames per second to sample before packing. Lower = faster/less coverage; higher = more coverage/VRAM.
- `--packing INT` (default: 0): temporal packing level for 3D‑Resampler. `0` disables; `1–6` packs consecutive frames to increase coverage.
- `--max-frames INT` (default: 0): hard cap on frames sent to the model per video (after sampling/packing). `0` means unlimited (scan the whole video at the chosen `--fps`). For long videos, consider keeping a cap or enabling `--packing` to control time/VRAM.
- `--max-slice-nums INT` (default: 1): split hi‑res frames into slices to reduce VRAM use. Increase if you hit OOM (e.g., `2` or `3`).
- `--attn {sdpa,flash_attention_2,eager}` (default: sdpa): attention backend. `sdpa` uses PyTorch’s scaled dot‑product attention (stable, built‑in). `flash_attention_2` uses custom CUDA kernels for speed but requires an exact‑match wheel for your Torch/CUDA. `eager` is a fallback (slower).
- `--model-revision STR` (optional, default: unset): pin the Hugging Face repo revision (commit SHA/tag/branch) for `openbmb/MiniCPM-V-4_5`. Pinning avoids surprise code updates and makes runs reproducible.
- `--dtype {bfloat16,float16,float32}` (default: float16): numeric precision for compute. `float16` (FP16) is compatible on most GPUs and fast. `bfloat16` (BF16) can be more numerically stable on newer GPUs (e.g., Ampere+ with BF16) with similar speed. `float32` is highest precision but slowest and uses most VRAM — helpful for debugging.
- `--image-size INT` (default: 448): resize frames to a fixed square size before inference to ensure consistent token shapes. 448 works well with the model’s 14‑pixel patch size (32×32 patches).
- `--no-resize`: disable resizing. Use only if you’re certain all frames produce identical token shapes; otherwise you may hit size mismatch errors.
- `--batch-size INT` (default: 16): microbatch size used by the CLI’s robust fallback when a tensor size mismatch occurs. Lower this if the fallback still fails; each batch is retried or split into single‑frame calls as needed.

## Tuning Guide (what the flags mean for detection)

- Coverage (`--fps`): higher fps samples more moments from the video, improving chances to catch brief lightning flashes, at the cost of time and VRAM. Start at 2–3; raise to 5 for thorough scans.
- Long videos (`--packing`): packing groups nearby frames so you keep temporal coverage without linearly increasing cost. Try `--packing 3` for multi‑minute clips.
- Runtime bounds (`--max-frames`): keep as `0` (unlimited) for full coverage. Set a number only to bound runtime on very long inputs.
- Memory control (`--max-slice-nums`): increase to 2–3 if you see CUDA OOM with high‑res frames; it slices images before encoding.
- Precision (`--dtype`): FP16 is fast and widely compatible; BF16 may be more stable on newer GPUs; FP32 is slow/high‑VRAM and mostly for debugging.
- Attention backend (`--attn`): `sdpa` is stable and built‑in. `flash_attention_2` can be faster but needs a matching wheel; if unsure, stick to `sdpa`.
- Reproducibility (`--model-revision`): pin a specific commit or tag once you confirm a working setup to prevent model code updates mid‑project.
- Input normalization (`--image-size`): resizing to 448×448 ensures all frames produce the same patch grid, avoiding tensor size mismatches.
- Robust fallback (`--batch-size`): on tensor size mismatch, the CLI automatically retries by microbatching and, if needed, single‑frame calls. Tune the microbatch size via `--batch-size`.
- `--thinking`: enable deeper reasoning mode (higher latency).
- `--no-preload-model`: skip upfront model load; the model initializes lazily before the first video. Useful to surface download/progress.

## Quick Recipes

- Faster scan with decent coverage: `--fps 3`
- Long clips on limited VRAM: `--packing 3 --max-slice-nums 2`
- Cap runtime on long inputs: `--max-frames 64`

For more scenarios and explanations, see `docs/user-guide.md`.

## Examples (pinned for stability)

- Minimal, fast smoke:
  - `lightning-detector scan --model-revision a8dd5db4715809f904dbf39c2a98a6112033f0f1`
- Slightly more coverage:
  - `lightning-detector scan --model-revision a8dd5db4715809f904dbf39c2a98a6112033f0f1 --fps 3 --max-frames 48`
- Longer clips, more coverage:
  - `lightning-detector scan --model-revision a8dd5db4715809f904dbf39c2a98a6112033f0f1 --fps 3 --packing 3 --max-frames 48 --max-slice-nums 2`

## Generated Help (sync via scripts/sync_cli_help.py)

<!-- BEGIN: GENERATED SCAN HELP -->

```
usage: lightning-detector scan [-h] [--input INPUT] [--output OUTPUT]
                               [--fps FPS] [--packing PACKING]
                               [--max-frames MAX_FRAMES]
                               [--max-slice-nums MAX_SLICE_NUMS]
                               [--model-revision MODEL_REVISION]
                               [--attn {sdpa,flash_attention_2,eager}]
                               [--dtype {bfloat16,float16,float32}]
                               [--thinking] [--no-preload-model]
                               [--image-size IMAGE_SIZE] [--no-resize]
                               [--batch-size BATCH_SIZE] [--no-color]
                               [--quiet]

options:
  -h, --help            show this help message and exit
  --packing PACKING     Temporal packing level for 3D-Resampler: 0 disables
                        packing; 1–6 pack consecutive frames to increase
                        coverage at similar cost (default: 0)

I/O:
  --input INPUT         Input directory with .mp4 files (default: videos)
  --output OUTPUT       Output directory for <name>.json, <name>.txt and
                        index.txt (default: reports)

Common tweaks:
  --fps FPS             Frames-per-second to sample before packing (lower =
                        faster; higher = more coverage) (default: 2)
  --max-frames MAX_FRAMES
                        Cap frames per video after sampling/packing (0 =
                        unlimited; use with care for long videos) (default: 0)
  --max-slice-nums MAX_SLICE_NUMS
                        Split hi-res frames into slices to reduce VRAM
                        (increase if OOM) (default: 1)

Model:
  --model-revision MODEL_REVISION
                        Pin the model repo revision (commit SHA, tag, or
                        branch). Improves reproducibility and avoids surprise
                        code updates from Hugging Face. (default: )

Advanced:
  --attn {sdpa,flash_attention_2,eager}
                        Attention backend; prefer sdpa unless you installed a
                        matching flash-attn wheel (default: sdpa)
  --dtype {bfloat16,float16,float32}
                        Computation dtype; float16 is broadly compatible;
                        bfloat16 on newer GPUs; float32 for debugging
                        (default: float16)
  --thinking            Enable deeper reasoning mode (higher latency)
                        (default: False)
  --no-preload-model    Skip upfront model load; initialize lazily before
                        first video (default: False)
  --image-size IMAGE_SIZE
                        Resize frames to this square size before inference
                        (multiple of 14 recommended) (default: 448)
  --no-resize           Do not resize frames; may cause tensor size mismatch
                        on some videos (default: False)
  --batch-size BATCH_SIZE
                        Microbatch size for robust fallback when size
                        mismatches occur (default: 16)
  --no-color            Disable ANSI colors in console output (default: False)
  --quiet               Reduce library noise (transformers logs, deprecation
                        warnings) (default: False)
```

<!-- END: GENERATED SCAN HELP -->
