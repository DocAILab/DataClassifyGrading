#!/usr/bin/env bash
# Reproducible single-GPU VeRL GRPO vertical smoke on the Phase-8 choice-protocol
# RL parquet (data/rl/<dataset>/, NOT the tests fixture).
#
# Uses the pinned VeRL's own GRPO (algorithm.adv_estimator=grpo) + vLLM
# rollout. The only task wiring is the thin choice-aware reward adapter
#   reward.custom_reward_function.path=pkg://agent.training.rl.verl_adapter
# The adapter routes rollouts through the shared choice parser
#   choice-id output -> check_stageX_choices -> canonical decode -> reward
# and is the ONLY task-specific component; all GRPO math stays in verl config.
#
# This is a smoke harness only: tiny batch, no accuracy claims, no reward /
# prompt / candidate changes, no hyper-parameter search. Observed rollouts and
# per-call parse/reward results are optionally dumped by setting REWARD_LOG_DIR
# (each call appends one JSON line; values unchanged).
#
# Memory knobs (all env-driven, safe defaults; see the 4B report for the 7B
# 32-GiB single-card boundary — use 0.5B to exercise the full RL loop):
#   ENFORCE_EAGER   false -> vLLM cudagraph ; true -> eager PyTorch
#   VLLM_COMPILE_OFF 0    -> default compile ; 1 -> level 0 + cudagraph NONE
#   PARAM_OFFLOAD   false -> actor weights on GPU ; true -> FSDP param offload CPU
#   GPU_MEM_UTIL    0.5   -> vLLM gpu_memory_utilization
#
# Run from the repo root in the RL venv (verl + vllm + ray):
#   DATASET=pers_info DATACLASSIFY_REGISTRY_DIR=cfg/task/registry \
#     REWARD_LOG_DIR=/root/autodl-tmp/phase12/obs \
#     TRAIN_FILE=... VAL_FILE=... MODEL_PATH=... bash script/verl/rl/grpo_smoke.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BASE="${RL_BASE:-$REPO_ROOT/.artifacts}"

DATASET="${DATASET:-pers_info}"
# Model is a pure config knob. Default is the stage task's target 7B model
# (32-GiB single-card boundary documented in the 4B report); use a smaller
# model (e.g. 0.5B) to exercise the full RL loop on this card.
MODEL_PATH="${MODEL_PATH:-$BASE/models/Qwen2.5-7B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-${RL_DATA_DIR:-$BASE/rl/$DATASET}/train.parquet}"
VAL_FILE="${VAL_FILE:-${RL_DATA_DIR:-$BASE/rl/$DATASET}/val.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-$BASE/outputs/grpo-smoke-$DATASET}"

# ---- minimal smoke knobs (tiny on purpose) ----
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-8}"     # default: 4 stage1 + 4 stage2
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"       # prompts per iteration
ROLLOUT_N="${ROLLOUT_N:-2}"                     # GRPO group size (rollouts/prompt)
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}"  # whole GRPO groups
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-256}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
TOTAL_STEPS="${TOTAL_STEPS:-3}"
TEST_FREQ="${TEST_FREQ:-1000}"                  # skip validation during smoke
SAVE_FREQ="${SAVE_FREQ:-1000}"                   # no checkpoint during 3 steps
ACTOR_LR="${ACTOR_LR:-1e-4}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.5}"
ENFORCE_EAGER="${ENFORCE_EAGER:-false}"
VLLM_COMPILE_OFF="${VLLM_COMPILE_OFF:-0}"
PARAM_OFFLOAD="${PARAM_OFFLOAD:-false}"
REWARD_LOG_DIR="${REWARD_LOG_DIR:-}"
REGISTRY_DIR="${DATACLASSIFY_REGISTRY_DIR:-cfg/task/registry}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export DATACLASSIFY_REGISTRY_DIR="$REGISTRY_DIR"
if [[ -n "$REWARD_LOG_DIR" ]]; then
  export DATACLASSIFY_REWARD_LOG_DIR="$REWARD_LOG_DIR"
  mkdir -p "$REWARD_LOG_DIR"
  echo "reward observation log -> $REWARD_LOG_DIR/reward_runtime.jsonl"
fi
# Blackwell co-location: decoded generation is orders of magnitude faster with
# CUDA graphs than eager; FLASH_ATTN avoids a missing flashinfer backend.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_DIR"
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "error: model not found: $MODEL_PATH" >&2
  exit 2
fi
for f in "$TRAIN_FILE" "$VAL_FILE"; do
  if [[ ! -f "$f" ]]; then
    echo "error: RL parquet not found: $f" >&2
    exit 2
  fi
done

# soft pre-gate: validate the 5-field RL contract (never masks pipeline issues)
if "$PYTHON_BIN" -m script.verl.rl.validate \
    --dataset-dir "$(dirname "$TRAIN_FILE")" \
    --dataset "$DATASET" \
    --registry "$REGISTRY_DIR/$DATASET.registry.json" \
    --corpus "cfg/task/corpus/$DATASET.corpus.json" \
    --metadata-fields field_name field_description > "$OUTPUT_DIR/validate.out" 2>&1; then
  echo "[grpo-smoke] validate: valid"
else
  echo "[grpo-smoke] validate: FAILED (see $OUTPUT_DIR/validate.out)" >&2
fi

ARGS=(
  "algorithm.adv_estimator=grpo"
  "algorithm.use_kl_in_reward=False"
  "data.train_files=$TRAIN_FILE"
  "data.val_files=$VAL_FILE"
  "data.train_batch_size=$TRAIN_BATCH_SIZE"
  "data.train_max_samples=$TRAIN_MAX_SAMPLES"
  "data.val_max_samples=$VAL_MAX_SAMPLES"
  "data.max_prompt_length=$MAX_PROMPT_LENGTH"
  "data.max_response_length=$MAX_RESPONSE_LENGTH"
  "data.shuffle=False"
  "data.truncation=error"
  "data.dataloader_num_workers=0"
  "data.prompt_key=prompt"
  "data.reward_fn_key=data_source"
  "actor_rollout_ref.model.path=$MODEL_PATH"
  "actor_rollout_ref.model.use_remove_padding=False"
  "actor_rollout_ref.model.enable_gradient_checkpointing=True"
  "+actor_rollout_ref.model.override_config.attn_implementation=sdpa"
  "actor_rollout_ref.model.lora_rank=8"
  "actor_rollout_ref.model.lora_alpha=16"
  "actor_rollout_ref.model.target_modules=all-linear"
  "actor_rollout_ref.actor.strategy=fsdp"
  "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False"
  "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE"
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384"
  "actor_rollout_ref.actor.use_kl_loss=False"
  "actor_rollout_ref.actor.entropy_coeff=0"
  "actor_rollout_ref.actor.optim.lr=$ACTOR_LR"
  "actor_rollout_ref.actor.optim.optimizer=AdamW"
  "actor_rollout_ref.actor.optim.weight_decay=0.0"
  "actor_rollout_ref.rollout.name=vllm"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
  "actor_rollout_ref.rollout.n=$ROLLOUT_N"
  "actor_rollout_ref.rollout.prompt_length=$MAX_PROMPT_LENGTH"
  "actor_rollout_ref.rollout.response_length=$MAX_RESPONSE_LENGTH"
  "actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN"
  "actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL"
  "actor_rollout_ref.rollout.enforce_eager=${ENFORCE_EAGER}"
  "actor_rollout_ref.rollout.enable_chunked_prefill=False"
  "actor_rollout_ref.rollout.enable_prefix_caching=False"
  "actor_rollout_ref.rollout.agent.num_workers=1"
  "actor_rollout_ref.rollout.free_cache_engine=True"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.ref.fsdp_config.param_offload=True"
  "reward.custom_reward_function.path=pkg://agent.training.rl.verl_adapter"
  "reward.custom_reward_function.name=compute_score"
  "reward.reward_manager.name=naive"
  "reward.reward_model.enable=False"
  "reward.num_workers=1"
  "trainer.critic_warmup=0"
  "trainer.n_gpus_per_node=1"
  "trainer.nnodes=1"
  "trainer.logger=['console']"
  "trainer.project_name=dataclassify-grpo"
  "trainer.experiment_name=qwen25-grpo-smoke-$DATASET"
  "trainer.default_local_dir=$OUTPUT_DIR/checkpoints"
  "trainer.total_epochs=4"
  "trainer.total_training_steps=$TOTAL_STEPS"
  "trainer.save_freq=$SAVE_FREQ"
  "trainer.test_freq=$TEST_FREQ"
  "trainer.resume_mode=disable"
  "actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD}"
)
if [[ "${VLLM_COMPILE_OFF}" == "1" ]]; then
  ARGS+=("+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config={level: 0, cudagraph_mode: NONE}")
fi

exec "$PYTHON_BIN" -m verl.trainer.main_ppo "${ARGS[@]}"
