#!/usr/bin/env bash
# Usage: source scripts/use-edge.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

# Prefer a dedicated edge venv; fall back to .venv if present
if [[ -f .venv-edge/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv-edge/bin/activate
elif [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "No virtualenv found. Create one with:"
  echo "  uv pip install --prefix .venv-edge -e . && source .venv-edge/bin/activate"
fi

unset TRANSFORMERS_OFFLINE
unset HF_HUB_OFFLINE

ld-scan-edge() {
  python -m lightning_detector.cli scan "$@"
}

echo "[edge] Activated. Online mode. No model revision pin." 
echo "[edge] Use: ld-scan-edge --fps 2 --packing 0  (flags optional)"

