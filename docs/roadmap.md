# Roadmap

This project integrates MiniCPM‑V 4.5 via `trust_remote_code=True`. Upstream changes can alter runtime behavior; this file tracks obstacles and planned work.

## Current Obstacles (as of 2025‑08)

- FlashAttention hard requirement in newer snapshots
  - Symptom: import‑time errors like: "This modeling file requires ... flash_attn" even when using `--attn sdpa`.
  - Impact: blocks runs without a matching `flash-attn` wheel for your Torch/CUDA.
  - Status: Workaround in place — use `--model-revision <sha>` plus offline env vars for reproducible runs.

- Processor/AutoProcessor mismatch
  - Symptom: `Unrecognized processing class ...` or `MiniCPMVTokenizerFast has no attribute image_processor`.
  - Cause: some snapshots need a custom Processor; `AutoProcessor` may return a tokenizer on certain Transformers versions.
  - Status: Fixed — model wrapper now constructs `MiniCPMVProcessor` dynamically and passes it to `model.chat`.

- Environment drift (Conda vs venv; Transformers too old)
  - Symptom: `cannot import name 'Qwen3Config'`.
  - Status: Documented — ensure `transformers>=4.47,<5` inside the venv and run via `.venv/bin/python -m lightning_detector.cli`.

## Near‑Term Plan

- Add an optional config for default model revision
  - So users don’t need to pass `--model-revision` every run.
  - Respect env var `MINICPM_REVISION` if set.

- Improve flash‑attn detection and guidance
  - At startup, check for `flash-attn` and print clear install vs. fallback advice based on Torch/CUDA.
  - Add a CLI flag `--attn flash_attention_2` shortcut that fails fast if wheel is missing.

- Snapshot compatibility matrix
  - Track known‑good SHAs and notes (processor behavior, attention backend, performance changes).

## Medium‑Term Plan

- Optional `flash-attn` support path
  - Provide a documented script to install matching wheels when available; otherwise, advise staying on SDPA with pinned snapshots.

- CI smoke checks for both branches
  - `main`: run offline against pinned revision.
  - `edge`: run online to surface upstream changes quickly.

## Branching Strategy

- `main` (stable): targets a pinned snapshot; changes must keep the stable workflow working.
- `edge` (latest): tracks upstream; breakages are acceptable short‑term while fixes land.

We’ll merge from `edge` to `main` once the latest snapshot is validated or when the wrapper gains the needed compatibility shims.

