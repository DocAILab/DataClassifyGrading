#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ztc/myhome/code/dataClassifying}"
DATASET_ROOT="${DATASET_ROOT:-/home/ztc/myhome/data/clsData}"
DOMAIN="${DOMAIN:-shougang}"
SPLITS="${SPLITS:-train val test all}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/data/preprocessed/${DOMAIN}}"

EXTRA_ARGS=()

if [[ "${FAIL_ON_EMPTY_LABEL:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--fail-on-empty-label)
fi

python3 "${PROJECT_DIR}/scripts/preprocess_clsdata.py" \
  --dataset-root "${DATASET_ROOT}" \
  --domain "${DOMAIN}" \
  --splits ${SPLITS} \
  --output-dir "${OUT_DIR}" \
  "${EXTRA_ARGS[@]}"
