#!/usr/bin/env bash
# Create the verified VeRL environment without vendoring VeRL.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3}"
VERL_ENV_DIR="${VERL_ENV_DIR:-$REPO_ROOT/.venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

if ! "$PYTHON_BOOTSTRAP" -c \
  'import sys; raise SystemExit(sys.version_info < (3, 11))' 2>/dev/null; then
  echo "The pinned VeRL environment requires Python 3.11 or newer" >&2
  exit 2
fi

venv_args=()
if [[ "${REUSE_SYSTEM_TORCH:-0}" == "1" ]]; then
  venv_args+=(--system-site-packages)
fi
if [[ ! -x "$VERL_ENV_DIR/bin/python" ]]; then
  "$PYTHON_BOOTSTRAP" -m venv "${venv_args[@]}" "$VERL_ENV_DIR"
fi
PYTHON_BIN="$VERL_ENV_DIR/bin/python"

"$PYTHON_BIN" -m pip install --upgrade pip
if ! "$PYTHON_BIN" - <<'PY'
try:
    import torch
except ImportError:
    raise SystemExit(1)

raise SystemExit(
    0
    if torch.__version__.startswith("2.8.0") and torch.version.cuda == "12.8"
    else 1
)
PY
then
  "$PYTHON_BIN" -m pip install torch==2.8.0 --index-url "$TORCH_INDEX_URL"
fi

"$PYTHON_BIN" -m pip install -r "$REPO_ROOT/requirements/verl.txt"
"$PYTHON_BIN" -m pip install -e "$REPO_ROOT" --no-deps
"$PYTHON_BIN" -m pip check

"$PYTHON_BIN" - <<'PY'
import pyarrow
import torch
import transformers
import verl
import verl.trainer.sft_trainer

print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("verl", verl.__version__)
print("transformers", transformers.__version__)
print("pyarrow", pyarrow.__version__)
print("sft_trainer_import", "ok")
PY
