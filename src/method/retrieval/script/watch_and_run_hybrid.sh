#!/usr/bin/env bash
set -euo pipefail

MODEL="/root/autodl-tmp/models/bge-m3"
WORKTREE="/root/autodl-tmp/worktrees/DataClassifyGrading-stage1-hybrid"
RUN_ROOT="/root/autodl-tmp/artifacts/shougang/stage1-hybrid-v1"
DATA_DIR="/root/autodl-tmp/datasets/shougang"
REGISTRY="/root/autodl-tmp/datasets/shougang/registry.json"
PYTHON="/root/autodl-tmp/envs/verl-official-sft/bin/python"

mkdir -p "$RUN_ROOT/logs"
while [ ! -s "$MODEL/pytorch_model.bin" ] \
   || [ -e "$MODEL/pytorch_model.bin.incomplete" ] \
   || [ -e "$MODEL/tokenizer.json.incomplete" ]; do
  echo "waiting-for-bge-model $(date -Is)"
  sleep 30
done

echo "model-ready $(date -Is)"
WORKTREE="$WORKTREE" RUN_ROOT="$RUN_ROOT" DATA_DIR="$DATA_DIR" \
REGISTRY="$REGISTRY" MODEL="$MODEL" PYTHON="$PYTHON" \
bash "$WORKTREE/src/method/retrieval/script/run_stage1_bge_m3.sh"
