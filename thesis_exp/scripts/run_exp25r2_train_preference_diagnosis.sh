#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ENV="${CONDA_ENV:-llama_factory}"
GPU_ID="${GPU_ID:-0}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/jpang/models/modelscope/Qwen/Qwen3-4B}"
OUT_DIR="${OUT_DIR:-thesis_exp/exp17_low_score_evidence/outputs/exp25r2_train_preference_diagnosis_seed42}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
LOG_STEPS="${LOG_STEPS:-25}"
WRITE_PAIR_DETAILS="${WRITE_PAIR_DETAILS:-0}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

if [[ -f "${HOME}/miniconda3/bin/activate" ]]; then
  source "${HOME}/miniconda3/bin/activate" "${CONDA_ENV}"
fi

python -m py_compile thesis_exp/exp17_low_score_evidence/diagnose_exp25_train_preference.py

args=(
  --model-name-or-path "${MODEL_NAME_OR_PATH}"
  --out-dir "${OUT_DIR}"
  --per-device-eval-batch-size "${PER_DEVICE_EVAL_BATCH_SIZE}"
  --max-examples "${MAX_EXAMPLES}"
  --log-steps "${LOG_STEPS}"
)
if [[ "${WRITE_PAIR_DETAILS}" == "1" ]]; then
  args+=(--write-pair-details)
fi

cat <<CONFIG
Exp25R2 train preference diagnosis
CONDA_ENV=${CONDA_ENV}
GPU_ID=${GPU_ID}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}
OUT_DIR=${OUT_DIR}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE}
MAX_EXAMPLES=${MAX_EXAMPLES}
WRITE_PAIR_DETAILS=${WRITE_PAIR_DETAILS}
CONFIG

python thesis_exp/exp17_low_score_evidence/diagnose_exp25_train_preference.py "${args[@]}"

echo "Exp25R2 train preference diagnosis completed."
