#!/usr/bin/env bash
# Thin single-node VeRL 0.8 SFT launcher. Hydra overrides remain an argv array.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_GPUS="${NUM_GPUS:-1}"
if [[ ! "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_GPUS must be a positive integer (got: $NUM_GPUS)" >&2
  exit 2
fi

attention_impl=""
remove_padding=""
for override in "$@"; do
  key="${override%%=*}"
  value="${override#*=}"
  case "$key" in
    model.override_config.attn_implementation|+model.override_config.attn_implementation)
      attention_impl="$value"
      ;;
    model.use_remove_padding)
      remove_padding="$value"
      ;;
  esac
done
if [[ "$attention_impl" == "sdpa" && "$remove_padding" != "false" && "$remove_padding" != "False" ]]; then
  echo "SDPA requires the explicit override model.use_remove_padding=false" >&2
  exit 2
fi

# Keep every override as one shell argument.  VeRL's Hydra entrypoint is the
# module after torch.distributed.run's own -m switch.
exec "$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  "--nproc-per-node=$NUM_GPUS" \
  -m verl.trainer.sft_trainer \
  "$@"
