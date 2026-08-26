#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ztc/myhome/code/dataClassifying}"
DATASET_ROOT="${DATASET_ROOT:-/home/ztc/myhome/data/clsData}"
DOMAIN="${DOMAIN:-shougang}"
SPLITS="${SPLITS:-train val test}"
CATALOG_SOURCE="${CATALOG_SOURCE:-all}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/data/shougang_sft}"
FORMAT="${FORMAT:-sharegpt}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-json}"
FIELD_SOURCE="${FIELD_SOURCE:-field_name}"
FIELD_CASE="${FIELD_CASE:-original}"
SEED="${SEED:-42}"

EXTRA_ARGS=()

if [[ "${CATALOG_SHUFFLE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--catalog-shuffle)
fi

if [[ "${INCLUDE_EMPTY_LABELS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--include-empty-labels)
fi

if [[ -n "${INCLUDE_METADATA:-}" ]]; then
  read -r -a METADATA_KEYS <<< "${INCLUDE_METADATA}"
  EXTRA_ARGS+=(--include-metadata "${METADATA_KEYS[@]}")
fi

if [[ "${WRITE_DATASET_INFO:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--write-dataset-info)
fi

python3 "${PROJECT_DIR}/scripts/build_sft.py" \
  --dataset-root "${DATASET_ROOT}" \
  --domain "${DOMAIN}" \
  --splits ${SPLITS} \
  --catalog-source "${CATALOG_SOURCE}" \
  --output-dir "${OUT_DIR}" \
  --format "${FORMAT}" \
  --output-format "${OUTPUT_FORMAT}" \
  --field-source "${FIELD_SOURCE}" \
  --field-case "${FIELD_CASE}" \
  --seed "${SEED}" \
  "${EXTRA_ARGS[@]}"
