#!/usr/bin/env bash
set -euo pipefail
WORKTREE="${WORKTREE:-/root/autodl-tmp/worktrees/DataClassifyGrading-stage1-bge-m3}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/artifacts/shougang/stage1-bge-m3-v1}"
mkdir -p "$RUN_ROOT"
setsid nohup bash "$WORKTREE/src/method/retrieval/script/run_stage1_bge_m3.sh" \
  > "$RUN_ROOT/nohup.log" 2>&1 < /dev/null &
echo $! > "$RUN_ROOT/pipeline.pid"
echo "started PID $(cat "$RUN_ROOT/pipeline.pid")"
