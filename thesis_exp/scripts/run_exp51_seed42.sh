#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
if [[ "${RUN_FORMAL:-0}" != "1" ]]; then
  echo "Refusing formal Exp51 seed42 without RUN_FORMAL=1" >&2
  exit 2
fi
GPU_ID="${GPU_ID:-0}"
MODEL_PATH="${MODEL_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
mkdir -p thesis_exp/outputs/exp51_hmsa/logs_private
CUDA_VISIBLE_DEVICES="${GPU_ID}" EXP51_REQUIRE_SOURCE_LOCK=1 PYTHONPATH=. python -m thesis_exp.exp51_hmsa.train \
  --model_name_or_path "${MODEL_PATH}" \
  --seed 42 \
  --gradient_checkpointing \
  --local_files_only \
  2>&1 | tee thesis_exp/outputs/exp51_hmsa/logs_private/seed42_hmsa.log
PYTHONPATH=. python -m thesis_exp.exp51_hmsa.gate
