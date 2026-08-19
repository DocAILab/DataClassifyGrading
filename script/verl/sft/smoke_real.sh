#!/usr/bin/env bash
# Reproducible single-GPU VeRL SFT vertical smoke on the REAL Phase-8
# choice-protocol parquet (data/sft/<dataset>/, NOT the tests fixture).
#
# Verifies end-to-end for the current canonical + PromptChoice pipeline:
#   canonical/exported parquet -> VeRL SFT dataloader -> tokenizer/chat template
#   -> model forward -> finite loss -> backward -> non-zero gradient
#   -> optimizer step -> checkpoint save -> (resume_mode=auto) checkpoint reload
#
# This is a smoke harness only: tiny subset, no accuracy claims, no prompt or
# reward changes, no hyper-parameter search. All values are env-driven.
#
# Pre-gates:
#   - script.verl.sft.check_token_budget (hard): every train/val/test row must
#     fit data.max_length under the real model chat template (truncation=error).
#   - script.verl.sft.validate (soft): writes a validation report; a failure is
#     saved to <OUTPUT_DIR>/validate.failed but does not block training, so a
#     gate problem never silently masks a pipeline problem (or vice versa).
#
# Batch layout: the default train subset is 8 rows (4 stage1 + 4 stage2), so
# train_batch_size=8 / micro_batch_size_per_gpu=1 gives EXACTLY one optimizer
# step per epoch -> total_epochs=total_training_steps is exact and small.
#
# Run from the repo root in the SFT venv:
#   DATASET=pers_info DATA_DIR=... MODEL_PATH=... OUTPUT_DIR=... \
#     bash script/verl/sft/smoke_real.sh
# Reload phase (resume=auto, same OUTPUT_DIR):
#   RESUME_MODE=auto TOTAL_STEPS=5 TOTAL_EPOCHS=5 ... bash script/verl/sft/smoke_real.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BASE="${P11_BASE:-$REPO_ROOT/.artifacts/p11}"

DATASET="${DATASET:-pers_info}"
DATA_DIR="${DATA_DIR:-$BASE/sft/$DATASET}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train.parquet}"
VAL_FILE="${VAL_FILE:-$DATA_DIR/val.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-$BASE/outputs/sft-vertical-smoke-$DATASET}"
MODEL_PATH="${MODEL_PATH:-$BASE/models/Qwen2.5-0.5B-Instruct}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# ---- smoke knobs (tiny on purpose; correctness not capacity) ----
MAX_LENGTH="${MAX_LENGTH:-512}"               # verified by token-budget gate
MAX_TOKEN_PER_GPU="${MAX_TOKEN_PER_GPU:-$MAX_LENGTH}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"     # 8 rows -> 1 optimizer step/epoch
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
TOTAL_STEPS="${TOTAL_STEPS:-3}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-$TOTAL_STEPS}"  # 1 step/epoch at batch=8
SAVE_FREQ="${SAVE_FREQ:-1}"
TEST_FREQ="${TEST_FREQ:-1}"                   # 1 val-loss computation per epoch
RESUME_MODE="${RESUME_MODE:-disable}"         # run 1: disable; reload: auto
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
ATTENTION_IMPL="${ATTENTION_IMPL:-sdpa}"
GRAD_CKPT="${GRAD_CKPT:-true}"
NUM_WORKERS="${NUM_WORKERS:-0}"
METADATA_FIELDS="${METADATA_FIELDS:-field_name field_description}"

export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

cd "$REPO_ROOT"
if [[ ! -f "$TRAIN_FILE" || ! -f "$VAL_FILE" ]]; then
  echo "error: SFT parquet not found in $DATA_DIR" >&2
  exit 2
fi
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "error: smoke model not found: $MODEL_PATH" >&2
  exit 2
fi
for split in train val test; do
  if [[ ! -f "$DATA_DIR/$split.parquet" ]]; then
    echo "error: token-budget gate needs $DATA_DIR/$split.parquet" >&2
    exit 2
  fi
done
mkdir -p "$OUTPUT_DIR"

echo "[smoke-real] dataset=$DATASET data=$DATA_DIR model=$MODEL_PATH out=$OUTPUT_DIR"
echo "[smoke-real] steps=$TOTAL_STEPS epochs=$TOTAL_EPOCHS batch=$TRAIN_BATCH_SIZE micro=$MICRO_BATCH_SIZE_PER_GPU max_len=$MAX_LENGTH resume=$RESUME_MODE"

# --- soft gate: contract validation report (never masks pipeline issues) ----
if "$PYTHON_BIN" -m script.verl.sft.validate \
    --dataset-dir "$DATA_DIR" \
    --registry "cfg/task/registry/$DATASET.registry.json" \
    --corpus "cfg/task/corpus/$DATASET.corpus.json" \
    --metadata-fields $METADATA_FIELDS \
    --report "$OUTPUT_DIR/validate.report.json" > "$OUTPUT_DIR/validate.out" 2>&1; then
  echo "[smoke-real] validate: valid (see $OUTPUT_DIR/validate.report.json)"
else
  echo "[smoke-real] validate: FAILED (see $OUTPUT_DIR/validate.out)" >&2
  touch "$OUTPUT_DIR/validate.failed"
fi

# --- hard gate: every row under max_length with the real chat template ----
if ! "$PYTHON_BIN" -m script.verl.sft.check_token_budget \
    --dataset-dir "$DATA_DIR" \
    --model "$MODEL_PATH" \
    --max-length "$MAX_LENGTH" \
    --report "$OUTPUT_DIR/token_budget.report.json" > "$OUTPUT_DIR/token_budget.out" 2>&1; then
  echo "error: token-budget gate failed (see $OUTPUT_DIR/token_budget.out)" >&2
  echo "  -> raise MAX_LENGTH; truncation=error would abort training otherwise" >&2
  exit 2
fi
echo "[smoke-real] token budget: ok (max_length=$MAX_LENGTH)"

ARGS=(
  "data.train_files=$TRAIN_FILE" "data.val_files=$VAL_FILE"
  data.messages_key=messages \
  "data.train_batch_size=$TRAIN_BATCH_SIZE" \
  "data.micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU" \
  "data.max_token_len_per_gpu=$MAX_TOKEN_PER_GPU" \
  "data.max_length=$MAX_LENGTH" \
  data.use_dynamic_bsz=false \
  "data.num_workers=$NUM_WORKERS" \
  "model.path=$MODEL_PATH" \
  model.use_remove_padding=false \
  "model.enable_gradient_checkpointing=$GRAD_CKPT" \
  "+model.override_config.attn_implementation=$ATTENTION_IMPL" \
  "model.lora_rank=$LORA_RANK" \
  "model.lora_alpha=$LORA_ALPHA" \
  "model.target_modules=all-linear" \
  engine=fsdp \
  engine.strategy=fsdp \
  engine.use_torch_compile=false \
  engine.dtype=bfloat16 \
  engine.model_dtype=bfloat16 \
  "optim.lr=$LEARNING_RATE" \
  "trainer.project_name=dataclassify-sft" \
  "trainer.experiment_name=qwen25-05b-sft-vertical-smoke-$DATASET" \
  "trainer.default_local_dir=$OUTPUT_DIR/checkpoints" \
  'trainer.logger=["console"]' \
  "trainer.total_epochs=$TOTAL_EPOCHS" \
  "trainer.total_training_steps=$TOTAL_STEPS" \
  "trainer.save_freq=$SAVE_FREQ" \
  "trainer.max_ckpt_to_keep=null" \
  "trainer.test_freq=$TEST_FREQ" \
  "trainer.resume_mode=$RESUME_MODE"
)
NUM_GPUS=1 PYTHON_BIN="$PYTHON_BIN" bash "$SCRIPT_DIR/run.sh" "${ARGS[@]}"
