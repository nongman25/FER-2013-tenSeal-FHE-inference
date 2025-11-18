#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
VENV_DIR="${PROJECT_ROOT}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi
if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# Kaggle credentials setup (optional)
KAGGLE_JSON="$PROJECT_ROOT/kaggle.json"
if [ -f "$KAGGLE_JSON" ]; then
  mkdir -p "$HOME/.kaggle"
  cp -f "$KAGGLE_JSON" "$HOME/.kaggle/kaggle.json"
  chmod 600 "$HOME/.kaggle/kaggle.json"
  eval "$(
    KAGGLE_JSON_PATH="$KAGGLE_JSON" "$PYTHON_BIN" - <<'PY'
import json, os, shlex
path = os.environ["KAGGLE_JSON_PATH"]
with open(path, "r", encoding="utf-8") as fh:
    cfg = json.load(fh)
def emit(env_name, key):
    value = cfg.get(key)
    if value:
        print(f'export {env_name}={shlex.quote(str(value))}')
emit("KAGGLE_USERNAME", "username")
emit("KAGGLE_KEY", "key")
PY
  )"
  echo "Kaggle credentials exported and copied to \$HOME/.kaggle/kaggle.json"
else
  echo "kaggle.json not found in project root; skipping Kaggle credential setup."
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
python -m pip install --index-url "$PYTORCH_INDEX_URL" \
  torch torchvision
python -m pip install \
  pandas "numpy<2" \
  scikit-learn \
  matplotlib \
  tqdm \
  pillow \
  tenseal \
  jupyter ipykernel \
  kaggle
python -m ipykernel install --user --name fhe-emotion --display-name "FHE Emotion" --force
echo "Virtual environment ready at $VENV_DIR and Jupyter kernel 'fhe-emotion' registered."
