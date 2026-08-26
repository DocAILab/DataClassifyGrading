#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ztc/myhome/code/dataClassifying}"

# Be conservative on a shared server: use four GPUs by default, with a target
# under one quarter of each A100-80G GPU's memory.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${model_name_or_path:-Qwen/Qwen3.5-9B}}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_DIR}/data/shougang_sft}"
DATASET="${DATASET:-shougang_sft_train}"
TEMPLATE="${TEMPLATE:-qwen3}"
RUN_DATE="${RUN_DATE:-$(date +%m%d)}"
RUN_BASENAME="${RUN_BASENAME:-qwen3_5_9b_shougang_lora_low_mem_${RUN_DATE}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/saves/${RUN_BASENAME}}"
RUN_CONFIG_DIR="${RUN_CONFIG_DIR:-${PROJECT_DIR}/runs}"
RUN_NAME="${RUN_NAME:-${RUN_BASENAME}}"

# Low-memory defaults for a 9B model on A100-80G, target < 20GB on each visible GPU.
CUTOFF_LEN="${CUTOFF_LEN:-6144}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LEARNING_RATE="${LEARNING_RATE:-1.0e-4}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2.0}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
SAVE_STEPS="${SAVE_STEPS:-500}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
PREPROCESSING_NUM_WORKERS="${PREPROCESSING_NUM_WORKERS:-4}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-1}"

mkdir -p "${RUN_CONFIG_DIR}" "${OUTPUT_DIR}"
CONFIG_PATH="${CONFIG_PATH:-${RUN_CONFIG_DIR}/${RUN_NAME}.yaml}"

MAX_SAMPLES_BLOCK=""
if [[ -n "${MAX_SAMPLES}" ]]; then
  MAX_SAMPLES_BLOCK="max_samples: ${MAX_SAMPLES}"
fi

cat > "${CONFIG_PATH}" <<YAML
### model
model_name_or_path: ${MODEL_NAME_OR_PATH}
trust_remote_code: true
#quantization_bit: 4
#quantization_method: bnb
flash_attn: auto

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: ${LORA_RANK}
lora_alpha: ${LORA_ALPHA}
lora_dropout: ${LORA_DROPOUT}

### dataset
dataset_dir: ${DATASET_DIR}
dataset: ${DATASET}
template: ${TEMPLATE}
enable_thinking: false
cutoff_len: ${CUTOFF_LEN}
overwrite_cache: true
preprocessing_num_workers: ${PREPROCESSING_NUM_WORKERS}
${MAX_SAMPLES_BLOCK}

### output
output_dir: ${OUTPUT_DIR}
logging_steps: ${LOGGING_STEPS}
save_steps: ${SAVE_STEPS}
plot_loss: true
overwrite_output_dir: false
save_only_model: true
report_to: none

### train
per_device_train_batch_size: ${PER_DEVICE_TRAIN_BATCH_SIZE}
gradient_accumulation_steps: ${GRADIENT_ACCUMULATION_STEPS}
learning_rate: ${LEARNING_RATE}
num_train_epochs: ${NUM_TRAIN_EPOCHS}
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true
fp16: false
gradient_checkpointing: true
dataloader_num_workers: ${DATALOADER_NUM_WORKERS}
YAML

echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Using FORCE_TORCHRUN=${FORCE_TORCHRUN}"
echo "Wrote config: ${CONFIG_PATH}"
echo "Target memory defaults: batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE}, cutoff_len=${CUTOFF_LEN}, lora_rank=${LORA_RANK}"

if ! command -v llamafactory-cli >/dev/null 2>&1; then
  echo "llamafactory-cli not found. Install LLaMA-Factory first, then rerun this script." >&2
  exit 127
fi

llamafactory-cli train "${CONFIG_PATH}"
