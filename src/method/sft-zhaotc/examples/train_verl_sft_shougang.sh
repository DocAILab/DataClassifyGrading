#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ztc/myhome/code/dataClassifying}"
VERL_DIR="${VERL_DIR:-/home/ztc/verl-0.6.1}"

# Use four GPUs by default, with a target under one quarter of each A100-80G GPU.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="${VERL_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MODEL_PATH="${MODEL_PATH:-${model_name_or_path:-Qwen/Qwen3.5-9B}}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data/shougang_verl}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train.parquet}"
VAL_FILE="${VAL_FILE:-${DATA_DIR}/val.parquet}"
RUN_DATE="${RUN_DATE:-$(date +%m%d)}"
RUN_BASENAME="${RUN_BASENAME:-verl_qwen3_5_9b_shougang_sft_${RUN_DATE}}"
SAVE_DIR="${SAVE_DIR:-${PROJECT_DIR}/saves/${RUN_BASENAME}}"
PROJECT_NAME="${PROJECT_NAME:-data-classifying-sft}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${RUN_BASENAME}}"

# Conservative FSDP + LoRA defaults for 4 x A100-80G, target < 20GB per GPU.
MAX_LENGTH="${MAX_LENGTH:-6144}"
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-8192}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
LR="${LR:-1e-4}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
SAVE_FREQ="${SAVE_FREQ:-after_each_epoch}"
TEST_FREQ="${TEST_FREQ:-after_each_epoch}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:--1}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:--1}"

if [[ ! -f "${TRAIN_FILE}" || ! -f "${VAL_FILE}" ]]; then
  echo "Missing VeRL parquet files. Run: bash examples/build_verl_parquet.sh" >&2
  echo "Expected: ${TRAIN_FILE} and ${VAL_FILE}" >&2
  exit 1
fi

python3 - <<'PY'
import sys
try:
    import transformers
except Exception as exc:
    print(f"Failed to import transformers: {exc}", file=sys.stderr)
    raise SystemExit(2)

if not hasattr(transformers, "AutoModelForVision2Seq"):
    print(
        "Your active transformers build does not export AutoModelForVision2Seq, "
        "which VeRL imports unconditionally. Upgrade transformers in the same env, "
        "for example: pip install -U 'transformers>=4.54.0'",
        file=sys.stderr,
    )
    raise SystemExit(2)
PY

cd "${VERL_DIR}"

torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" \
  -m verl.trainer.sft_trainer \
  model.path="${MODEL_PATH}" \
  model.tokenizer_path="${MODEL_PATH}" \
  model.trust_remote_code=true \
  model.enable_gradient_checkpointing=true \
  model.enable_activation_offload=true \
  model.use_remove_padding=true \
  model.lora_rank="${LORA_RANK}" \
  model.lora_alpha="${LORA_ALPHA}" \
  model.target_modules=all-linear \
  engine.strategy=fsdp \
  engine.fsdp_size="${NPROC_PER_NODE}" \
  engine.dtype=bfloat16 \
  engine.param_offload=false \
  engine.optimizer_offload=true \
  engine.reshard_after_forward=true \
  engine.use_torch_compile=false \
  optim.lr="${LR}" \
  optim.lr_scheduler_type=cosine \
  optim.lr_warmup_steps_ratio=0.03 \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
  data.max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
  data.use_dynamic_bsz=false \
  data.pad_mode=no_padding \
  data.max_length="${MAX_LENGTH}" \
  data.truncation=right \
  data.train_max_samples="${TRAIN_MAX_SAMPLES}" \
  data.val_max_samples="${VAL_MAX_SAMPLES}" \
  +data.multiturn.messages_key=messages \
  +data.multiturn.enable_thinking_key=enable_thinking \
  checkpoint.save_contents='["model"]' \
  checkpoint.load_contents='["model"]' \
  trainer.default_local_dir="${SAVE_DIR}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.logger='["console"]' \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.resume_mode=auto \
  trainer.device=cuda \
  "$@"
