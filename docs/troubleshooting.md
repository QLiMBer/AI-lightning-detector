Troubleshooting & First‑Run Notes

Quick sanity checks

- GPU available: `python -c "import torch; print(torch.cuda.is_available())"`
- Versions: `python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"`
- Minimal model call: `python scripts/smoke_chat.py --attn sdpa --dtype float16`

Common issues and fixes

- ImportError `cannot import name 'Qwen3Config' from 'transformers'`
  - Cause: Transformers < 4.47. Fix: upgrade to `transformers>=4.47,<5` and re‑sync (uv: `uv lock && uv sync`).

- FlashAttention errors
  - Keep `--attn sdpa` unless your `flash-attn` wheel matches your exact Torch+CUDA. On mismatch, prefer SDPA.

- Stuck at first model load / no GPU activity
  - Avoid suspended runs holding cache locks (Ctrl‑Z). List and kill: `jobs -l` then `kill -TERM %<id>` (or `pkill -f lightning-detector`).
  - Force a clean local cache and skip preload to surface progress per video:
    - `HF_HOME=.hf-cache TRANSFORMERS_CACHE=.hf-cache/hub HF_HUB_ENABLE_HF_TRANSFER=1 lightning-detector scan --input videos --output reports --fps 1 --max-frames 16 --max-slice-nums 1 --attn sdpa --dtype float16 --no-preload-model`
  - Remove stale locks if any: `find .hf-cache -name '*.lock' -delete` (ensure no runs are active).

- CUDA OOM or very slow runs
  - Lower `--fps`, lower `--max-frames`, increase `--max-slice-nums`, or enable `--packing` for long videos.

Observability tips

- Watch GPU: `watch -n1 nvidia-smi` (expect non‑zero util during chat)
- CLI prints per‑video progress: `Processing … | frames=… | method=…` and `Finished … in Xs`

