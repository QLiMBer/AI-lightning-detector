#!/usr/bin/env bash
# Usage: source scripts/use-stable.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

PINNED_REV="a8dd5db4715809f904dbf39c2a98a6112033f0f1"

# Prefer a dedicated stable venv; fall back to .venv if present
if [[ -f .venv-stable/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv-stable/bin/activate
elif [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "No virtualenv found. Create one with:"
  echo "  uv pip install --prefix .venv-stable -e . && source .venv-stable/bin/activate"
fi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export LD_MODEL_REV="$PINNED_REV"

ld-scan() {
  python -m lightning_detector.cli scan --model-revision "${LD_MODEL_REV}" "$@"
}

echo "[stable] Activated. Offline mode ON. Default model revision: $LD_MODEL_REV"
echo "[stable] Use: ld-scan --fps 2 --packing 0  (flags optional)"

