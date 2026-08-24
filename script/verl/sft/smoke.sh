#!/usr/bin/env bash
# Reproducible single-GPU VeRL 0.8 SFT smoke test for tracked schema-v2 data.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
FIXTURE_DIR="$REPO_ROOT/tests/sft/fixtures"
BASE="${SFT_BASE:-$REPO_ROOT/.artifacts}"
MODEL_PATH="${MODEL_PATH:-$BASE/models/Qwen2.5-0.5B-Instruct}"
SMOKE_MODEL_ID="${SMOKE_MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
SMOKE_DATA_DIR="${SMOKE_DATA_DIR:-$BASE/sft-fixture-release}"
TRAIN_FILE="${TRAIN_FILE:-$SMOKE_DATA_DIR/train.parquet}"
VAL_FILE="${VAL_FILE:-$SMOKE_DATA_DIR/val.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-$BASE/outputs/sft-smoke-qwen25-05b-lora}"
STEPS="${STEPS:-2}"
EPOCHS="${EPOCHS:-$STEPS}"
ATTENTION_IMPL="${ATTENTION_IMPL:-sdpa}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

cd "$REPO_ROOT"
# VeRL is intentionally an optional local dependency. The compatibility CI
# installs it explicitly; a source checkout without it should report a skip,
# not fail before the CPU-only unit suite can run.
if ! "$PYTHON_BIN" -c 'import verl' >/dev/null 2>&1; then
  echo "skip: verl==0.8.0 is not installed (install requirements/verl.txt for the GPU smoke)"
  exit 0
fi
if ! "$PYTHON_BIN" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
  echo "skip: CUDA is unavailable (the SFT smoke requires one GPU)"
  exit 0
fi

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  if [[ "${DOWNLOAD_SMOKE_MODEL:-1}" != "1" ]]; then
    printf 'error: smoke model not found: %s\n' "$MODEL_PATH" >&2
    exit 2
  fi
  MODEL_PATH="$MODEL_PATH" SMOKE_MODEL_ID="$SMOKE_MODEL_ID" \
    "$PYTHON_BIN" - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["SMOKE_MODEL_ID"],
    local_dir=os.environ["MODEL_PATH"],
)
PY
fi

# The fixture is the same schema-v2 canonical contract consumed in production:
# metadata_fields is exactly [field_name], and Stage 2 uses the joint
# {answer, level} grading contract.
"$PYTHON_BIN" -m script.verl.sft.export \
  --canonical "$FIXTURE_DIR/canonical/all.json" \
  --output-dir "$SMOKE_DATA_DIR" \
  --registry "$FIXTURE_DIR/registry.json" \
  --corpus "$FIXTURE_DIR/corpus.json" \
  --task-config "$FIXTURE_DIR/task.json" \
  --grading-config "$FIXTURE_DIR/grading.json" \
  --metadata-fields field_name
"$PYTHON_BIN" -m script.verl.sft.validate \
  --dataset-dir "$SMOKE_DATA_DIR" \
  --registry "$FIXTURE_DIR/registry.json" \
  --corpus "$FIXTURE_DIR/corpus.json" \
  --task-config "$FIXTURE_DIR/task.json" \
  --grading-config "$FIXTURE_DIR/grading.json" \
  --metadata-fields field_name
"$PYTHON_BIN" -m script.verl.sft.check_token_budget \
  --dataset-dir "$SMOKE_DATA_DIR" \
  --model "$MODEL_PATH" \
  --max-length 512

NUM_GPUS=1 PYTHON_BIN="$PYTHON_BIN" bash "$SCRIPT_DIR/run.sh" \
  "data.train_files=$TRAIN_FILE" \
  "data.val_files=$VAL_FILE" \
  data.messages_key=messages \
  data.train_batch_size=2 \
  data.micro_batch_size_per_gpu=1 \
  data.max_token_len_per_gpu=512 \
  data.max_length=512 \
  data.truncation=error \
  data.use_dynamic_bsz=false \
  data.num_workers=0 \
  "model.path=$MODEL_PATH" \
  model.use_remove_padding=false \
  model.enable_gradient_checkpointing=false \
  "+model.override_config.attn_implementation=$ATTENTION_IMPL" \
  model.lora_rank=8 \
  model.lora_alpha=16 \
  model.target_modules=all-linear \
  engine.strategy=fsdp \
  engine.model_dtype=bfloat16 \
  engine.dtype=bfloat16 \
  engine.use_torch_compile=false \
  optim.lr=1e-4 \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=1 \
  trainer.project_name=dataclassify-sft \
  trainer.experiment_name=qwen25-05b-lora-smoke \
  "trainer.default_local_dir=$OUTPUT_DIR/checkpoints" \
  'trainer.logger=["console"]' \
  "trainer.total_epochs=$EPOCHS" \
  "trainer.total_training_steps=$STEPS" \
  trainer.save_freq=1 \
  trainer.max_ckpt_to_keep=1 \
  trainer.test_freq=-1 \
  trainer.resume_mode=disable \
  trainer.resume_from_path=null
