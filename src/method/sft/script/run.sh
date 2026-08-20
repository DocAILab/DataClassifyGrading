#!/usr/bin/env bash
# Thin single-node launcher; all VeRL/Hydra configuration is forwarded unchanged.
set -euo pipefail

attention_impl=""
remove_padding=""
for override in "$@"; do
  case "$override" in
    *model.override_config.attn_implementation=*)
      attention_impl="${override##*=}"
      ;;
    model.use_remove_padding=*)
      remove_padding="${override##*=}"
      ;;
  esac
done
if [[ "$attention_impl" == "sdpa" && "$remove_padding" != "false" ]]; then
  echo "SDPA requires the explicit override model.use_remove_padding=false" >&2
  exit 2
fi

exec "${PYTHON_BIN:-python}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="${NUM_GPUS:-1}" \
  -m verl.trainer.sft_trainer \
  "$@"
