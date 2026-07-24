#!/usr/bin/env bash
set -euo pipefail

ARM="${1:?Usage: run_exp54_rar_sft_smoke.sh ARM OUTPUT_DIR}"
OUTPUT_DIR="${2:?Usage: run_exp54_rar_sft_smoke.sh ARM OUTPUT_DIR}"
GPU_INDEX="${GPU_INDEX:?Set GPU_INDEX to one reviewed RTX A6000 index}"
SMOKE_AUTHORIZATION="${SMOKE_AUTHORIZATION:?A reviewed smoke authorization is required}"

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python -m thesis_exp.exp54_rar_sft.train_rar_sft_smoke \
  --arm "${ARM}" \
  --smoke-authorization "${SMOKE_AUTHORIZATION}" \
  --output-dir "${OUTPUT_DIR}"
