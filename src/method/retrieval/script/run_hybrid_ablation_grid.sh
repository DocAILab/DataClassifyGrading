#!/usr/bin/env bash
set -euo pipefail

WORKTREE="/root/autodl-tmp/worktrees/DataClassifyGrading-stage1-hybrid"
RUN_ROOT="/root/autodl-tmp/artifacts/shougang/stage1-hybrid-v1"
DATA_DIR="/root/autodl-tmp/datasets/shougang"
REGISTRY="/root/autodl-tmp/datasets/shougang/registry.json"
MODEL="/root/autodl-tmp/models/bge-m3"
PYTHON="/root/autodl-tmp/envs/verl-official-sft/bin/python"
VARIANT_ROOT="$RUN_ROOT/variants"
mkdir -p "$VARIANT_ROOT"

run_variant() {
  local name="$1" lexical="$2" dense="$3" index="$4"
  local output="$VARIANT_ROOT/$name"
  if [ -f "$output/COMPLETE" ]; then
    echo "already-complete $name"
    return
  fi
  rm -rf "$output"
  echo "starting $name lexical=$lexical dense=$dense train_index=$index"
  "$PYTHON" -m method.retrieval.script.evaluate_stage1 \
    --input-dir "$DATA_DIR" --registry "$REGISTRY" --model "$MODEL" \
    --output-dir "$output" --batch-size 32 \
    --lexical-weight "$lexical" --dense-weight "$dense" --index-weight "$index"
  test -f "$output/COMPLETE"
}

cd "$WORKTREE"
export PYTHONPATH="$WORKTREE/src"
run_variant index_dominant 0.05 0.15 0.80
run_variant index_bge 0.00 0.35 0.65
run_variant index_char 0.15 0.00 0.85
run_variant balanced 0.20 0.50 0.30
echo "ablation grid complete"
