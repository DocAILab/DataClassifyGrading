#!/usr/bin/env bash
set -euo pipefail

WORKTREE="${WORKTREE:-/root/autodl-tmp/worktrees/DataClassifyGrading-stage1-bge-m3}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/artifacts/shougang/stage1-bge-m3-v1}"
DATA_DIR="${DATA_DIR:-/root/autodl-tmp/datasets/shougang}"
REGISTRY="${REGISTRY:-$DATA_DIR/registry.json}"
MODEL="${MODEL:-/root/autodl-tmp/models/bge-m3}"
PYTHON="${PYTHON:-/root/autodl-tmp/envs/verl-official-sft/bin/python}"

mkdir -p "$RUN_ROOT/logs"
status() {
  "$PYTHON" - "$RUN_ROOT/status.json" "$1" "$2" <<'PY'
import datetime, json, pathlib, sys
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({"phase": sys.argv[2], "state": sys.argv[3], "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}, indent=2) + "\n")
PY
}
trap 'status pipeline failed' ERR

cd "$WORKTREE"
export PYTHONPATH="$WORKTREE/src"
status tests running
"$PYTHON" -m pytest -q tests/method/retrieval > "$RUN_ROOT/logs/tests.log" 2>&1

available_kib=$(df -Pk /root/autodl-tmp | awk 'NR==2 {print $4}')
if [ "$available_kib" -lt 8388608 ]; then
  echo "at least 8 GiB of free data-disk space is required" >&2
  exit 1
fi
if [ ! -f "$MODEL/config.json" ]; then
  echo "BGE-M3 model is incomplete: $MODEL" >&2
  exit 1
fi

status smoke running
rm -rf "$RUN_ROOT/smoke"
"$PYTHON" -m method.retrieval.script.evaluate_stage1 \
  --input-dir "$DATA_DIR" --registry "$REGISTRY" --model "$MODEL" \
  --output-dir "$RUN_ROOT/smoke" --batch-size 16 --limit 16 \
  > "$RUN_ROOT/logs/smoke.log" 2>&1
"$PYTHON" - "$RUN_ROOT/smoke/evaluation_report.json" <<'PY'
import json, pathlib, sys
report = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert report["rows"] == 16
assert report["embedding_dimension"] == 1024
assert report["real_test_split_read"] is False
assert report["metadata_fields"] == ["field_name"]
PY
touch "$RUN_ROOT/SMOKE_VERIFIED"

status full running
rm -rf "$RUN_ROOT/full"
"$PYTHON" -m method.retrieval.script.evaluate_stage1 \
  --input-dir "$DATA_DIR" --registry "$REGISTRY" --model "$MODEL" \
  --output-dir "$RUN_ROOT/full" --batch-size 32 \
  > "$RUN_ROOT/logs/full.log" 2>&1
touch "$RUN_ROOT/FULL_COMPLETE"
status complete complete
touch "$RUN_ROOT/COMPLETE"
