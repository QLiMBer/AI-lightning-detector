# Roadmap & Ideas

This page collects near‑term ideas, experiments, and reminders.

## Reproducibility: pin model revision

- Problem: Hugging Face model code can update between runs when `trust_remote_code=True` is used.
- Current: Use `--model-revision <commit-or-tag>` to pin `openbmb/MiniCPM-V-4_5`.
- Decision/TODO: Once we settle on a stable setup, choose and document a specific revision to pin for repeatable results.
- Notes: Pinning prevents surprise downloads of updated Python files from the model repo.

## Profiles (presets)

- Goal: Quick presets for different workflows (e.g., dev vs full scan) without lengthy flags.
- Dev profile (proposal): keeps runs fast and interruptible while developing.
  - Example defaults: `--fps 2`, `--max-frames 32–64`, `--image-size 448`, `--attn sdpa`, `--dtype float16`, `--no-preload-model`.
  - Optional: smaller `--batch-size` for responsiveness during fallbacks.
- Full/Thorough profile (proposal): higher coverage for final passes.
  - Example tweaks: `--fps 4–6`, optional `--packing 2–4`, raise `--max-frames`, consider `--max-slice-nums 2` on high‑res.
- Implementation options:
  - Simple: env var `LD_PROFILE=dev` that expands to a predefined arg string.
  - Config file: `profiles.toml` with named presets.
  - CLI: `lightning-detector scan --profile dev`.

## Always‑microbatch mode (not implemented)

- Idea: Process frames in fixed microbatches to improve responsiveness and allow clean interrupts between batches.
- Trade‑offs:
  - Pros: better progress feedback, faster Ctrl‑C handling, isolates bad frames early.
  - Cons: more `model.chat` calls; may reduce peak GPU utilization. Expected throughput impact depends on microbatch size and I/O — ballpark 5–20% slower at small batch sizes.
- Next steps: add a flag (e.g., `--always-microbatch` with `--batch-size N`) and benchmark across typical clips.

## Misc ideas

- Add `--profile` help section in `docs/cli.md` once profiles land.
- Option to write a short per‑video CSV alongside JSON for quick grepping.
- Optional per‑batch progress prints (behind a verbose flag) when always‑microbatching is enabled.
