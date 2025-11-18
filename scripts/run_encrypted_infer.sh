#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
VENV_DIR="$PROJECT_ROOT/.venv"
if [ ! -d "$VENV_DIR" ]; then
  echo "Virtual environment not found at $VENV_DIR. Please run scripts/setup_env.sh first." >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
python -m he.fhe_inference
