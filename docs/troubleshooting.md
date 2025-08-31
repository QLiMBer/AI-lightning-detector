Troubleshooting & First‑Run Notes

Scope: issues and fixes. For setup and daily usage, see repository `README.md` (Quick Start) and `docs/user-guide.md`.

Quick sanity checks

- GPU available: `python -c "import torch; print(torch.cuda.is_available())"`
- Versions: `python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"`
- Minimal model call: `python scripts/smoke_chat.py --attn sdpa --dtype float16`

Common issues and fixes

- ImportError `cannot import name 'Qwen3Config' from 'transformers'`
  - Cause: Transformers < 4.47. Fix: upgrade to `transformers>=4.47,<5` and re‑sync (uv: `uv lock && uv sync`).

- FlashAttention errors
  - Keep `--attn sdpa` unless your `flash-attn` wheel matches your exact Torch+CUDA. On mismatch, prefer SDPA.

- Upstream revision suddenly requires flash-attn (even with `--attn sdpa`)
  - Symptom: `This modeling file requires the following packages ... flash_attn` or CLI note `flash_attn_required`.
  - Why: the model’s remote Python code changed; some revisions import FlashAttention2 unconditionally.
  - Quick fixes:
    - Pin a known‑good model revision and run offline to avoid updating code:
      - `export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1`
      - `lightning-detector scan --model-revision <commit-sha> --attn sdpa --dtype float16`
    - If you must use flash‑attn, install a matching wheel (fragile):
      - `uv pip install --prefix .venv --no-build-isolation flash-attn`
      - If wheels aren’t available, a source build needs CUDA toolkit + `CUDA_HOME` and `nvcc` on PATH (heavier and error‑prone on WSL2).
  - Prevent: always pass `--model-revision` for reproducibility; consider `TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1` during PoCs.

- "Why did it download Python files?" (model code updates)
  - This model requires `trust_remote_code=True` and ships custom Python modules (e.g., `modeling_minicpmv.py`, `resampler.py`). On first run—or when the upstream repo updates—Transformers fetches those files and warns so you can review changes.
  - To avoid surprise updates and make runs reproducible, pin the model revision: add `--model-revision <commit-or-tag>` to your command (e.g., a commit SHA from the model repo).

- Processor/AutoProcessor mismatch (MiniCPMV)
  - Symptoms:
    - `Unrecognized processing class ... Can't instantiate a processor, a tokenizer, an image processor or a feature extractor for this model.`
    - `MiniCPMVTokenizerFast has no attribute image_processor`
  - Cause: some MiniCPM‑V snapshots need a custom `Processor` (image + text). Older Transformers or naive `AutoProcessor` calls can return a tokenizer only.
  - Fixed in this repo: the model wrapper now constructs `MiniCPMVProcessor` via dynamic modules and passes it to `model.chat`.
  - If you see this on an older checkout: update the repo and rerun.

- Stuck at first model load / no GPU activity
  - Avoid suspended runs holding cache locks (Ctrl‑Z). List and kill: `jobs -l` then `kill -TERM %<id>` (or `pkill -f lightning-detector`).
  - Force a clean local cache and skip preload to surface progress per video:
    - `HF_HOME=.hf-cache TRANSFORMERS_CACHE=.hf-cache/hub HF_HUB_ENABLE_HF_TRANSFER=1 lightning-detector scan --input videos --output reports --fps 1 --max-frames 16 --max-slice-nums 1 --attn sdpa --dtype float16 --no-preload-model`
  - Remove stale locks if any: `find .hf-cache -name '*.lock' -delete` (ensure no runs are active).

- Tensor size mismatch during inference
  - Symptom: `ERROR: inference_failed: Sizes of tensors must match ... Expected size 245 but got size 244 ...`
  - Cause: frames result in different token patch grids; the model expects identical shapes across the batch.
  - Fix: keep default resizing (448×448) or set `--image-size 448`. Avoid `--no-resize` unless you guarantee uniform shapes.
  - Fallback behavior: the CLI automatically retries by microbatching and, if needed, single‑frame calls to bypass bad mixes of shapes. Tune this with `--batch-size` (try smaller values like `8` or `4` if necessary).

- Segmentation fault during GPU move (e.g., after "Moving model to CUDA…")
  - Install Accelerate and use automatic placement: `uv pip install --prefix .venv accelerate`
  - Use safer loader flags (already in this repo): `device_map='auto'`, `low_cpu_mem_usage=True`, `offload_folder='.offload'`
  - Initialize the model before importing/using `decord` (the CLI does this now)
  - Prefer `--dtype float16` if BF16 is unstable on your GPU/driver
  - Set allocator: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

- CUDA OOM or very slow runs
  - Lower `--fps`, lower `--max-frames`, increase `--max-slice-nums`, or enable `--packing` for long videos.

- Interrupting a long run (Ctrl‑C seems unresponsive)
  - During large GPU kernels, Ctrl‑C (SIGINT) is queued and only handled when Python regains control; it can look “stuck” for a while.
  - To stop immediately, send a signal to the process: `pkill -INT -f lightning-detector` (or find PID via `ps aux | grep lightning-detector` and `kill -TERM <pid>`; escalate to `-KILL` if needed).
  - For safer first runs, bound work: `--fps 1-2`, `--max-frames 16-48`, optionally `--packing 2-3`.

Observability tips

- Watch GPU: `watch -n1 nvidia-smi` (expect non‑zero util during chat)
- CLI prints per‑video progress: `Processing … | frames=… | method=…` and `Finished … in Xs`
  - Note: current inference is batched; per‑batch progress can be limited during model.chat. If you need more granular progress, consider smaller `--max-frames` or ask us to add an always‑microbatch mode.
