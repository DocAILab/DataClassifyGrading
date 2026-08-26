#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ztc/myhome/code/dataClassifying}"
SFT_DIR="${SFT_DIR:-${PROJECT_DIR}/data/shougang_sft}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/data/shougang_verl}"
DOMAIN="${DOMAIN:-shougang}"
SPLITS="${SPLITS:-train val test}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"

mkdir -p "${OUT_DIR}"

for split in ${SPLITS}; do
  python3 "${PROJECT_DIR}/scripts/convert_sft_to_verl_parquet.py" \
    --input "${SFT_DIR}/${DOMAIN}_sft_${split}.json" \
    --output "${OUT_DIR}/${split}.parquet" \
    --enable-thinking "${ENABLE_THINKING}"
done
