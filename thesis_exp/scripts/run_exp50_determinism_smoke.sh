#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
if [[ -f "${HOME}/miniconda3/bin/activate" ]]; then source "${HOME}/miniconda3/bin/activate" "${CONDA_ENV:-llama_factory}"; fi
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
GPU_ID="${GPU_ID:-0}"
ROOT="thesis_exp/outputs/exp50_cahs/audit/determinism"
mkdir -p "${ROOT}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -m thesis_exp.exp50_cahs.determinism_smoke --model-name-or-path "${MODEL_NAME_OR_PATH}" --output "${ROOT}/run_a.json" --microbatches 64
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -m thesis_exp.exp50_cahs.determinism_smoke --model-name-or-path "${MODEL_NAME_OR_PATH}" --output "${ROOT}/run_b.json" --microbatches 64
"${PYTHON_BIN}" -m thesis_exp.exp50_cahs.determinism_smoke --compare "${ROOT}/run_a.json" "${ROOT}/run_b.json" --output "${ROOT}/comparison.json"
