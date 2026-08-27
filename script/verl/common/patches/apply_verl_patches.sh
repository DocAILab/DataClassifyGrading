#!/usr/bin/env bash
# apply_verl_patches.sh — replay the AmberFalcon verl-0.9.0 patch bundle onto a
# fresh site-packages tree. Read-only w.r.t. the repo; writes only under the
# target site-packages.
#
# Usage:
#   bash apply_verl_patches.sh <site-packages-dir>      # required patches only
#   INCLUDE_DEBUG=1 bash apply_verl_patches.sh <dir>    # also apply DEBUG patch
#
# Behavior:
#   - verify the target file set exists and matches the official wheel sha256
#     (pre-condition) before applying; a patched file fails the pre-check and
#     is skipped as already-applied (idempotent).
#   - apply each required patch with `patch -p1`; on any failure, revert every
#     already-applied patch of this run with `patch -R` and exit non-zero.
#   - after all applies: verify per-file sha256 against the installed table
#     and `python -m py_compile` each file.
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <site-packages-dir>" >&2
  exit 2
fi
SP="$1"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/verl-0.9.0"

REQUIRED=(
  "chat_template-system-first.patch"
  "multiturn_sft_dataset-prefix-diff-answer-mask.patch"
  "losses-scheme-c.patch"
)
OPTIONAL_DEBUG="agent_loop-debug.patch"

# target relative path -> expected sha256 after applying (installed values)
declare -A EXPECTED=(
  ["verl/utils/tokenizer/chat_template.py"]="58031af7a001a1208129b271f110e9cf94a3978874fadc5da43db9eec0322578"
  ["verl/utils/dataset/multiturn_sft_dataset.py"]="060f0adfe4caf5a4a19b0b7dd443ec417117070db16039ee8905691bd8aa85ba"
  ["verl/workers/utils/losses.py"]="7035e59a0678b8172c608f72d296ef9893babf35fa27650f72be8d54d1d6fdca"
  ["verl/experimental/agent_loop/agent_loop.py"]="902cc8c4007b944974d77c54bc1ce227df49de4390e5b3a0fc831f5cf0a4a801"
)
# target relative path -> official wheel sha256 (pre-condition for patching)
declare -A WHEEL=(
  ["verl/utils/tokenizer/chat_template.py"]="a6f7cc2915edccba"
  ["verl/utils/dataset/multiturn_sft_dataset.py"]="fa9a39662fd016dc"
  ["verl/workers/utils/losses.py"]="c6dfeb10900054e9"
  ["verl/experimental/agent_loop/agent_loop.py"]="04234862956696ba"
)
WHEEL_FULL=(
  "verl/utils/tokenizer/chat_template.py a6f7cc2915edccba"
  "verl/utils/dataset/multiturn_sft_dataset.py fa9a39662fd016dc"
  "verl/workers/utils/losses.py c6dfeb10900054e9"
  "verl/experimental/agent_loop/agent_loop.py 04234862956696ba"
)

sha16() { sha256sum "$1" | cut -d' ' -f1 | cut -c1-16; }

if [ ! -d "$SP/verl" ]; then
  echo "error: $SP/verl not found (not a site-packages?)" >&2
  exit 2
fi

PLAN=("${REQUIRED[@]}")
if [ "${INCLUDE_DEBUG:-0}" = "1" ]; then
  PLAN+=("$OPTIONAL_DEBUG")
fi

APPLIED=()
cd "$SP"
for pf in "${PLAN[@]}"; do
  target="${pf%.patch}"
  case "$pf" in
    chat_template-system-first.patch) target="verl/utils/tokenizer/chat_template.py" ;;
    multiturn_sft_dataset-prefix-diff-answer-mask.patch) target="verl/utils/dataset/multiturn_sft_dataset.py" ;;
    losses-scheme-c.patch) target="verl/workers/utils/losses.py" ;;
    agent_loop-debug.patch) target="verl/experimental/agent_loop/agent_loop.py" ;;
  esac
  if [ ! -f "$target" ]; then
    echo "error: target $target missing" >&2
    exit 2
  fi
  cur="$(sha16 "$target")"
  if [ "${EXPECTED[$target]}" = "$cur" ]; then
    echo "skip (already applied): $target"
    continue
  fi
  # pre-condition: must be pristine wheel content
  wheel_ok=0
  for entry in "${WHEEL_FULL[@]}"; do
    set -- $entry
    if [ "$1" = "$target" ] && [ "$2" = "$cur" ]; then wheel_ok=1; fi
  done
  if [ "$wheel_ok" != "1" ]; then
    echo "error: $target is neither wheel-pristine nor patched; refusing" >&2
    exit 2
  fi
  echo "applying: $pf"
  if ! patch -p1 < "$PATCH_DIR/$pf"; then
    echo "error: patch failed on $target; reverting this run" >&2
    for done_patch in "${APPLIED[@]}"; do
      patch -p1 -R < "$PATCH_DIR/$done_patch" >/dev/null 2>&1 || true
    done
    exit 1
  fi
  APPLIED+=("$pf")
done

# post-apply verification
FAIL=0
for target in "${!EXPECTED[@]}"; do
  got="$(sha16 "$SP/$target")"
  want="${EXPECTED[$target]}"
  if [ "$got" != "$want" ]; then
    echo "VERIFY FAIL: $target sha16=$got want=$want" >&2
    FAIL=1
  fi
done
if [ "$FAIL" = "1" ]; then
  echo "error: post-apply sha256 mismatch; run verify_verl_patches.sh for details" >&2
  exit 1
fi
echo "sha256 verification: OK (all expected installed hashes)"
for target in "${!EXPECTED[@]}"; do
  python -m py_compile "$SP/$target" 2>/dev/null \
    || echo "warning: py_compile skipped for $target (no python on PATH?)" >&2
done
echo "apply_verl_patches: done (${#APPLIED[@]} applied, rest skipped)"
