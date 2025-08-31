# Repository Guidelines

## Project Structure & Module Organization
- `lightning_detector/`: core package
  - `cli.py`: argparse CLI (`lightning-detector scan`), sampling, reporting
  - `video.py`: video I/O and frame sampling (decord, PIL)
  - `model.py`: MiniCPM‑V 4.5 wrapper (Transformers, Torch)
  - `json_utils.py`: robust JSON extraction/formatting
- `scripts/`: utilities (e.g., `inspect_videos.py` for metadata)
- `docs/`: setup, usage, plan, and references
- `videos/`: local input clips (gitignored)
- `reports/`: JSON and raw text outputs (gitignored)
- `pyproject.toml`: package metadata and console script

## Build, Run, and Development
- Environment (uv-managed): `uv lock && uv sync` then `source .venv/bin/activate`
- Install Torch (CUDA if applicable): see `docs/minicpm-v/setup.md`
- Editable install: `uv pip install --prefix .venv -e .`
- Run detector:
  - `lightning-detector scan --input videos --output reports --fps 5 --packing 0`
  - Results: `reports/<video>.json`, `reports/<video>.txt`, and `reports/index.txt`
- Inspect inputs: `python scripts/inspect_videos.py videos`

## Coding Style & Naming Conventions
- Python 3.10+, PEP 8, 4‑space indent; prefer type hints.
- Names: `snake_case` for functions/vars, `PascalCase` for classes, `UPPER_SNAKE` for constants.
- Keep modules cohesive: sampling in `video.py`, prompting/IO in `cli.py`, model calls in `model.py`.
- Docstrings: brief purpose + key args/returns; keep functions small and testable.

## Testing Guidelines
- Framework: pytest (add as dev dep when tests are introduced).
- Location: `tests/` mirroring package paths; files `test_*.py`.
- Use tiny fixtures or mocks (e.g., mock `decord.VideoReader`) to avoid large assets.
- Commands: `uv pip install --prefix .venv pytest && pytest -q`

## Commit & Pull Request Guidelines
- Conventional Commits (seen in history): `feat(cli): …`, `fix: …`, `chore: …`.
- PRs: describe scope, link issues, list commands run, and attach sample output paths (e.g., `reports/index.txt`).
- Keep diffs focused; update docs/examples if flags or outputs change.

## Security & Configuration Tips
- Large artifacts are ignored (`videos/`, `reports/`, `*.whl`); do not commit media or model weights.
- Default attention is `sdpa`; `flash-attn` is optional for performance (`--attn flash_attention_2`).
- Prefer BF16 (`--dtype bfloat16`) on modern GPUs; use FP32 only for debugging.
