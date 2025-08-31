# Contributing

This repo integrates a fast‑moving upstream model (MiniCPM‑V 4.5 via `trust_remote_code=True`). To keep development smooth and users stable, we use a two‑track workflow and small, focused PRs.

## Branches

- `main` (stable)
  - Pinned to a known‑good model revision (documented in README).
  - Preferred for demos, user runs, and day‑to‑day feature development.
- `edge` (latest)
  - Tracks the latest upstream model code (no pin). Experimental fixes live here until proven.

## Environments

Use separate virtualenvs to avoid dependency clashes:

- Stable: `.venv-stable`
  - Create: `uv pip install --prefix .venv-stable -e .`
  - Activate: `source .venv-stable/bin/activate`
  - Run with a pinned model revision (see README for the current SHA).
- Edge: `.venv-edge`
  - Create: `uv pip install --prefix .venv-edge -e .`
  - Activate: `source .venv-edge/bin/activate`
  - Run without `--model-revision` (may require extra deps like `flash-attn`).

Convenience helpers are provided under `scripts/` — see below.

## Workflow

- Most work targets `main`.
  - Create a feature branch: `git checkout -b feat/<name> main`
  - Push and open a PR to `main`.
  - Squash merge, using Conventional Commits in the title (e.g., `feat(cli): add --foo`).
- Keep `edge` current by merging `main` regularly:
  - `git checkout edge && git merge main` (or `git rebase main`). Resolve small conflicts early.
- Only merge `edge` → `main` when the latest model is stable. Otherwise, cherry‑pick isolated fixes if needed.

## Docs policy (per‑branch)

- Docs live with code on each branch (treat docs as code). Keep `README.md` accurate for that branch:
  - On `main`: show pinned, stable commands (with `--model-revision`).
  - On `edge`: note experimental/latest status and any extra setup (e.g., `flash-attn`).
- Avoid a shared README across branches; instead, merge docs changes from `main` into `edge` frequently so they don’t drift.
- Keep changes small and frequent to prevent large conflicts later.

## Helper scripts

- `scripts/use-stable.sh`: source into your shell to activate a stable workflow with the pinned revision and offline mode.
- `scripts/use-edge.sh`: source into your shell to activate an edge workflow targeting the latest upstream model.

Each script prints what it configures and provides an `ld-scan` (stable) or `ld-scan-edge` (edge) convenience function that wraps the CLI with appropriate defaults.

## Releasing

- When `edge` proves stable against the latest upstream snapshot, merge `edge` → `main`, update the pinned revision in README, and tag a release.
- If a breaking upstream change appears later, repeat: patch on `edge`, validate, then merge to `main`.

