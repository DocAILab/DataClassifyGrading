#!/usr/bin/env bash
set -euo pipefail

worktree="${WORKTREE:-/root/autodl-tmp/worktrees/DataClassifyGrading-stage2-hard-dpo}"
run_root="${RUN_ROOT:-/root/autodl-tmp/artifacts/shougang/stage2-hard-dpo-v1}"
runner="$worktree/src/method/dpo/script/run_stage2_hard_dpo.sh"
mkdir -p "$run_root"

if [[ -f "$run_root/pipeline.pid" ]]; then
  existing_pid="$(tr -cd '0-9' < "$run_root/pipeline.pid")"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "pipeline is already running with PID $existing_pid"
    exit 0
  fi
fi

setsid nohup bash "$runner" > "$run_root/nohup.log" 2>&1 < /dev/null &
pipeline_pid="$!"
printf '%s\n' "$pipeline_pid" > "$run_root/pipeline.pid"
echo "started pipeline PID $pipeline_pid"
