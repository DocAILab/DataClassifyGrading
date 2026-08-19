#!/usr/bin/env bash
# Reproducible Qwen2.5-7B-Instruct SFT/LoRA baseline launcher (Phase 13).
#
# Trains on a Phase-8 choice-protocol SFT parquet split and records every
# hyper-parameter + the resolved verl config so the baseline is fully
# reproducible (this checkpoint seeds the later GRPO/RLOO/ReMax runs).
#
# Pre-gates: contract validation (soft) + token-budget with the real model
# chat template (hard, only blocks if a row exceeds MAX_LENGTH at
# truncation=error). Every run dumps:
#   <OUTPUT_DIR>/hyperparams.json   (this launcher's explicit knobs)
#   <OUTPUT_DIR>/train_config.json  (verl resolved Hydra config dump)
#   <OUTPUT_DIR>/checkpoints/...    (LoRA FSDP checkpoints, save_freq)
#   stderr/stdout -> log  (val/loss per trainer.test_freq)
#
# Defaults are the Phase-13 "reasonable default" run. All env-driven.
#
# Run from the repo root in the SFT venv:
#   DATASET=pers_info DATA_DIR=... MODEL_PATH=... OUTPUT_DIR=... \
#     bash script/verl/sft/run_baseline.sh > baseline.log 2>&1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BASE="${P13_BASE:-$REPO_ROOT/.artifacts/p13}"

DATASET="${DATASET:-pers_info}"
DATA_DIR="${DATA_DIR:-$BASE/sft/$DATASET}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train.parquet}"
VAL_FILE="${VAL_FILE:-$DATA_DIR/val.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-$BASE/outputs/sft-baseline-$DATASET}"
MODEL_PATH="${MODEL_PATH:-$BASE/models/Qwen2.5-7B-Instruct}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# ---- hyper-parameters (all recorded in hyperparams.json) ----
SEED="${SEED:-42}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
EPOCHS="${EPOCHS:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-2}"
MAX_LENGTH="${MAX_LENGTH:-512}"
MAX_TOKEN_PER_GPU="${MAX_TOKEN_PER_GPU:-$MAX_LENGTH}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
TARGET_MODULES="${TARGET_MODULES:-all-linear}"
ATTENTION_IMPL="${ATTENTION_IMPL:-sdpa}"
GRAD_CKPT="${GRAD_CKPT:-true}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:-5}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MAX_CKPT_KEEP="${MAX_CKPT_KEEP:-null}"

GPUS="${NUM_GPUS:-1}"
GRAD_ACCUM=$(( TRAIN_BATCH_SIZE / (MICRO_BATCH_SIZE_PER_GPU * GPUS) ))

export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export DATACLASSIFY_REGISTRY_DIR="${DATACLASSIFY_REGISTRY_DIR:-cfg/task/registry}"

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_DIR"
if [[ ! -f "$TRAIN_FILE" || ! -f "$VAL_FILE" ]]; then
  echo "error: SFT parquet not found in $DATA_DIR" >&2
  exit 2
fi
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "error: model not found: $MODEL_PATH" >&2
  exit 2
fi
for split in train val test; do
  if [[ ! -f "$DATA_DIR/$split.parquet" ]]; then
    echo "warn: token-budget gate needs $DATA_DIR/$split.parquet (missing $split)" >&2
  fi
done

# ---- soft gate: contract validation report ----
if "$PYTHON_BIN" -m script.verl.sft.validate \
    --dataset-dir "$DATA_DIR" \
    --registry "cfg/task/registry/$DATASET.registry.json" \
    --corpus "cfg/task/corpus/$DATASET.corpus.json" \
    --metadata-fields field_name field_description \
    --report "$OUTPUT_DIR/validate.report.json" > "$OUTPUT_DIR/validate.out" 2>&1; then
  echo "[baseline] validate: valid"
else
  echo "[baseline] validate: FAILED (see $OUTPUT_DIR/validate.out)" >&2
fi

# ---- hard gate: token budget ----
if ! "$PYTHON_BIN" -m script.verl.sft.check_token_budget \
    --dataset-dir "$DATA_DIR" \
    --model "$MODEL_PATH" \
    --max-length "$MAX_LENGTH" \
    --report "$OUTPUT_DIR/token_budget.report.json" > "$OUTPUT_DIR/token_budget.out" 2>&1; then
  echo "error: token-budget gate failed (see $OUTPUT_DIR/token_budget.out)" >&2
  exit 2
fi
echo "[baseline] token budget ok (max_length=$MAX_LENGTH)"

# ---- record hyper-parameters ----
"$PYTHON_BIN" - <<PY > "$OUTPUT_DIR/hyperparams.json"
import json, subprocess, sys
try:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
except Exception:
    commit = "n/a"
params = {
    "dataset": "$DATASET",
    "model_path": "$MODEL_PATH",
    "train_file": "$TRAIN_FILE",
    "val_file": "$VAL_FILE",
    "seed": int("$SEED"),
    "learning_rate": float("$LR"),
    "weight_decay": float("$WEIGHT_DECAY"),
    "epochs": int("$EPOCHS"),
    "train_batch_size": int("$TRAIN_BATCH_SIZE"),
    "micro_batch_size_per_gpu": int("$MICRO_BATCH_SIZE_PER_GPU"),
    "gpus": int("$GPUS"),
    "grad_accumulation_steps": int("$GRAD_ACCUM"),
    "max_length": int("$MAX_LENGTH"),
    "max_token_len_per_gpu": int("$MAX_TOKEN_PER_GPU"),
    "lora_rank": int("$LORA_RANK"),
    "lora_alpha": int("$LORA_ALPHA"),
    "target_modules": "$TARGET_MODULES",
    "precision": "bfloat16",
    "engine": "fsdp (strategy=fsdp)",
    "optimizer": "AdamW",
    "scheduler": "constant (lr_warmup_steps_ratio=0)",
    "attention_impl": "$ATTENTION_IMPL",
    "gradient_checkpointing": "$GRAD_CKPT",
    "save_freq": int("$SAVE_FREQ"),
    "test_freq": int("$TEST_FREQ"),
    "git_commit": commit,
}
json.dump(params, sys.stdout, ensure_ascii=False, indent=2)
print()
PY
echo "[baseline] hyperparams -> $OUTPUT_DIR/hyperparams.json"

ARGS=(
  "data.train_files=$TRAIN_FILE" "data.val_files=$VAL_FILE"
  data.messages_key=messages
  "data.train_batch_size=$TRAIN_BATCH_SIZE"
  "data.micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU"
  "data.max_token_len_per_gpu=$MAX_TOKEN_PER_GPU"
  "data.max_length=$MAX_LENGTH"
  data.use_dynamic_bsz=false
  "data.num_workers=$NUM_WORKERS"
  "model.path=$MODEL_PATH"
  model.use_remove_padding=false
  "model.enable_gradient_checkpointing=$GRAD_CKPT"
  "+model.override_config.attn_implementation=$ATTENTION_IMPL"
  "model.lora_rank=$LORA_RANK"
  "model.lora_alpha=$LORA_ALPHA"
  "model.target_modules=$TARGET_MODULES"
  engine=fsdp
  engine.strategy=fsdp
  engine.use_torch_compile=false
  engine.dtype=bfloat16
  engine.model_dtype=bfloat16
  "engine.seed=$SEED"
  "optim.lr=$LR"
  "optim.weight_decay=$WEIGHT_DECAY"
  "trainer.project_name=dataclassify-sft-baseline"
  "trainer.experiment_name=qwen25-7b-sft-baseline-$DATASET"
  "trainer.default_local_dir=$OUTPUT_DIR/checkpoints"
  "trainer.seed=$SEED"
  'trainer.logger=["console"]'
  "trainer.total_epochs=$EPOCHS"
  "trainer.total_training_steps=null"
  "trainer.save_freq=$SAVE_FREQ"
  "trainer.max_ckpt_to_keep=$MAX_CKPT_KEEP"
  "trainer.test_freq=$TEST_FREQ"
  trainer.resume_mode=disable
)
NUM_GPUS="$GPUS" PYTHON_BIN="$PYTHON_BIN" bash "$SCRIPT_DIR/run.sh" "${ARGS[@]}"
