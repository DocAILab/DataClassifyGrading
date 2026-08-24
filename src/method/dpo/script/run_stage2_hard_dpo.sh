#!/usr/bin/env bash
set -Eeuo pipefail

worktree="${WORKTREE:-/root/autodl-tmp/worktrees/DataClassifyGrading-stage2-hard-dpo}"
run_root="${RUN_ROOT:-/root/autodl-tmp/artifacts/shougang/stage2-hard-dpo-v1}"
python_bin="${PYTHON_BIN:-/root/autodl-tmp/envs/verl-official-sft/bin/python}"
model="${MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-7B-Instruct}"
sft_adapter="${SFT_ADAPTER:-/root/autodl-tmp/artifacts/shougang/sft-random-shuffled-v1/sft-peft-export/lora_adapter}"
data_dir="${DATA_DIR:-/root/autodl-tmp/datasets/shougang}"
registry="${REGISTRY_PATH:?REGISTRY_PATH is required}"
task_config="${TASK_CONFIG_PATH:?TASK_CONFIG_PATH is required}"
status_path="$run_root/status.json"

mkdir -p "$run_root/logs"
export PYTHONPATH="$worktree/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$worktree"

write_status() {
  local phase="$1"
  local state="$2"
  "$python_bin" - "$status_path" "$phase" "$state" <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
path = Path(sys.argv[1])
value = {
    "phase": sys.argv[2],
    "state": sys.argv[3],
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
temporary.replace(path)
PY
}

on_error() {
  write_status "${current_phase:-unknown}" "failed"
  touch "$run_root/FAILED"
}
trap on_error ERR

current_phase="tests"
write_status "$current_phase" "running"
"$python_bin" -m pytest -q -m 'not verl' | tee "$run_root/logs/tests.log"
touch "$run_root/TESTS_COMPLETE"

current_phase="storage_audit"
write_status "$current_phase" "running"
"$python_bin" - "$run_root/storage_report.json" <<'PY'
import json
from pathlib import Path
import sys
from method.dpo.storage import assert_training_capacity
report = assert_training_capacity("/root/autodl-tmp", "/dev/shm")
Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
PY
touch "$run_root/STORAGE_VERIFIED"

current_phase="mining"
write_status "$current_phase" "running"
if [[ ! -f "$run_root/MINING_COMPLETE" ]]; then
  "$python_bin" -m method.dpo.script.mine_preferences \
    --input-dir "$data_dir" \
    --registry "$registry" \
    --task-config "$task_config" \
    --model "$model" \
    --sft-adapter "$sft_adapter" \
    --output-dir "$run_root/mining" \
    --batch-size 5 \
    --seed 42 2>&1 | tee -a "$run_root/logs/mining.log"
  touch "$run_root/MINING_COMPLETE"
fi

current_phase="preference_export"
write_status "$current_phase" "running"
if [[ ! -f "$run_root/PREFERENCES_COMPLETE" ]]; then
  "$python_bin" -m method.dpo.script.export_preferences \
    --input-dir "$data_dir" \
    --registry "$registry" \
    --task-config "$task_config" \
    --score-path "$run_root/mining/label_scores.jsonl" \
    --output-dir "$run_root/preferences" \
    --seed 42 2>&1 | tee "$run_root/logs/preference-export.log"
  touch "$run_root/PREFERENCES_COMPLETE"
fi

current_phase="smoke"
write_status "$current_phase" "running"
if [[ ! -f "$run_root/smoke-verification/SMOKE_VERIFIED" ]]; then
  "$python_bin" -m method.dpo.script.train \
    --model "$model" \
    --sft-adapter "$sft_adapter" \
    --preferences "$run_root/preferences/preferences.parquet" \
    --output-dir "$run_root/dpo-smoke" \
    --max-steps 1 2>&1 | tee "$run_root/logs/dpo-smoke.log"
  "$python_bin" -m method.dpo.script.verify_smoke \
    --training-dir "$run_root/dpo-smoke" \
    --verification-dir "$run_root/smoke-verification" 2>&1 | tee "$run_root/logs/smoke-verification.log"
fi

current_phase="full_training"
write_status "$current_phase" "running"
if [[ ! -f "$run_root/dpo-full/COMPLETE" ]]; then
  resume_args=()
  latest_checkpoint="$(find "$run_root/dpo-full" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' 2>/dev/null | sort -V | tail -1 || true)"
  if [[ -n "$latest_checkpoint" ]]; then
    resume_args=(--resume-from-checkpoint "$run_root/dpo-full/$latest_checkpoint")
  fi
  "$python_bin" -m method.dpo.script.train \
    --model "$model" \
    --sft-adapter "$sft_adapter" \
    --preferences "$run_root/preferences/preferences.parquet" \
    --output-dir "$run_root/dpo-full" \
    "${resume_args[@]}" 2>&1 | tee -a "$run_root/logs/dpo-full.log"
fi

current_phase="paired_val"
write_status "$current_phase" "running"
if [[ ! -f "$run_root/val-evaluation/COMPLETE" ]]; then
  "$python_bin" -m method.dpo.script.evaluate \
    --input-dir "$data_dir" \
    --registry "$registry" \
    --task-config "$task_config" \
    --model "$model" \
    --sft-adapter "$sft_adapter" \
    --dpo-adapter "$run_root/dpo-full/final_adapter" \
    --output-dir "$run_root/val-evaluation" \
    --batch-size 5 \
    --seed 137 2>&1 | tee -a "$run_root/logs/val-evaluation.log"
fi

test -s "$run_root/val-evaluation/comparison_to_sft.json"

current_phase="complete"
write_status "$current_phase" "complete"
touch "$run_root/COMPLETE"
rm -f "$run_root/FAILED"
