#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ztc/myhome/code/dataClassifying}"
REF_FILE="${REF_FILE:-${PROJECT_DIR}/data/shougang_sft/shougang_sft_test.json}"
PRED_FILE="${PRED_FILE:-}"
PRED_FORMAT="${PRED_FORMAT:-auto}"
REF_FORMAT="${REF_FORMAT:-auto}"
PRED_FIELD="${PRED_FIELD:-prediction}"
OUTPUT_REPORT="${OUTPUT_REPORT:-${PROJECT_DIR}/data/shougang_sft/eval_report.json}"

if [[ -z "${PRED_FILE}" ]]; then
  echo "PRED_FILE is required" >&2
  exit 1
fi

python3 "${PROJECT_DIR}/scripts/eval_exact_match.py" \
  --reference "${REF_FILE}" \
  --prediction "${PRED_FILE}" \
  --reference-format "${REF_FORMAT}" \
  --prediction-format "${PRED_FORMAT}" \
  --prediction-field "${PRED_FIELD}" \
  --output-report "${OUTPUT_REPORT}"
