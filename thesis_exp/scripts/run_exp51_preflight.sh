#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
GPU_ID="${GPU_ID:-0}"
MODEL_PATH="${MODEL_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH=. python -m thesis_exp.exp51_hmsa.preflight --model_name_or_path "${MODEL_PATH}"
