#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/autodl-tmp
WORKSPACE="$ROOT/workspace"
RUN="$ROOT/artifacts/shougang/two-stage-grpo-v1/sft-eval-val"
PYTHON="$ROOT/envs/verl-official-sft/bin/python"
STATUS="$RUN/status.json"
mkdir -p "$RUN"

write_status() {
  printf '{\n  "stage": "sft_val_evaluation",\n  "state": "%s",\n  "updated_at": "%s",\n  "pid": %s\n}\n' \
    "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" > "$STATUS"
}

failed() {
  write_status failed
}
trap failed ERR

write_status running
cd "$WORKSPACE"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m method.sft.script.evaluate_two_stage \
  --model "$ROOT/models/Qwen2.5-7B-Instruct" \
  --adapter "$ROOT/artifacts/shougang/two-stage-grpo-v1/sft-peft-export/lora_adapter" \
  --val "$ROOT/datasets/shougang/val.json" \
  --registry "$ROOT/artifacts/shougang/orpo-v1/registry.json" \
  --output-dir "$RUN" \
  --batch-size 8 \
  > "$RUN/evaluation.log" 2>&1
test -s "$RUN/evaluation_report.json"
test -s "$RUN/predictions.jsonl"
test -s "$RUN/COMPLETE"
write_status complete
