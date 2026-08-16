#!/usr/bin/env bash
# Build the optional FlashAttention2 dependency against the active PyTorch/CUDA.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3}"
MAX_JOBS="${MAX_JOBS:-2}"
NVCC_THREADS="${NVCC_THREADS:-1}"
FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-120}"

export CUDA_HOME MAX_JOBS NVCC_THREADS FLASH_ATTN_CUDA_ARCHS
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export FLASH_ATTENTION_FORCE_BUILD=TRUE

if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  printf 'error: nvcc not found under CUDA_HOME=%s\n' "$CUDA_HOME" >&2
  exit 2
fi

"$PYTHON_BIN" -m pip install ninja
"$PYTHON_BIN" -m pip install \
  --no-build-isolation \
  "flash-attn==$FLASH_ATTN_VERSION"

"$PYTHON_BIN" - <<'PY'
import flash_attn
import torch

print("flash_attn", flash_attn.__version__)
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "not visible")
PY
