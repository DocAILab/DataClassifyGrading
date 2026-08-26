#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ztc/myhome/code/dataClassifying}"
DATASET_ROOT="${DATASET_ROOT:-/home/ztc/myhome/data/clsData}"
DOMAIN="${DOMAIN:-shougang}"
SPLIT="${SPLIT:-all}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/data}"
MIN_COUNT="${MIN_COUNT:-1}"
SORT_BY="${SORT_BY:-path}"

python3 "${PROJECT_DIR}/scripts/inspect_catalog.py" \
  --input "${DATASET_ROOT}/${DOMAIN}/${SPLIT}.json" \
  --output-json "${OUT_DIR}/${DOMAIN}_catalog_stats.json" \
  --output-txt "${OUT_DIR}/${DOMAIN}_catalog.txt" \
  --min-count "${MIN_COUNT}" \
  --sort-by "${SORT_BY}"
