#!/usr/bin/env bash
# verify_verl_patches.sh — read-only verification of an existing verl install
# against the AmberFalcon patch table (sha256 + py_compile + dry-run status).
#
# Usage: bash verify_verl_patches.sh <site-packages-dir>
set -uo pipefail

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

declare -A EXPECTED=(
  ["verl/utils/tokenizer/chat_template.py"]="58031af7a001a1208129b271f110e9cf94a3978874fadc5da43db9eec0322578"
  ["verl/utils/dataset/multiturn_sft_dataset.py"]="060f0adfe4caf5a4a19b0b7dd443ec417117070db16039ee8905691bd8aa85ba"
  ["verl/workers/utils/losses.py"]="7035e59a0678b8172c608f72d296ef9893babf35fa27650f72be8d54d1d6fdca"
  ["verl/experimental/agent_loop/agent_loop.py"]="902cc8c4007b944974d77c54bc1ce227df49de4390e5b3a0fc831f5cf0a4a801"
)

sha256_of() { sha256sum "$1" | cut -d' ' -f1; }

[ -d "$SP/verl" ] || { echo "error: $SP/verl not found" >&2; exit 2; }

FAIL=0
for target in "${!EXPECTED[@]}"; do
  f="$SP/$target"
  if [ ! -f "$f" ]; then
    echo "MISSING: $target" >&2; FAIL=1; continue
  fi
  got="$(sha256_of "$f")"
  want="${EXPECTED[$target]}"
  if [ "$got" = "$want" ]; then
    echo "OK   : $target"
  else
    echo "DRIFT: $target sha256=$got (want $want)" >&2; FAIL=1
  fi
  python -m py_compile "$f" 2>/dev/null && echo "  compile OK" \
    || echo "  compile FAIL" >&2
done

cd "$SP"
for pf in "${REQUIRED[@]}"; do
  if patch -p1 --dry-run -R < "$PATCH_DIR/$pf" >/dev/null 2>&1; then
    echo "APPLIED: $pf (reverse-dry-run matches → patch present)"
  else
    echo "NOT-APPLIED or drift: $pf" >&2
  fi
done
if [ -n "${OPTIONAL_DEBUG:-}" ]; then
  if patch -p1 --dry-run -R < "$PATCH_DIR/$OPTIONAL_DEBUG" >/dev/null 2>&1; then
    echo "APPLIED (debug, P2): $OPTIONAL_DEBUG"
  else
    echo "NOT-APPLIED (debug patch absent — expected for formal envs): $OPTIONAL_DEBUG"
  fi
fi

if [ "$FAIL" = "1" ]; then
  echo "verify_verl_patches: FAIL (see DRIFT/MISSING above)" >&2
  exit 1
fi
echo "verify_verl_patches: PASS"
