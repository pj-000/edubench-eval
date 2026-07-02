#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-llama_factory}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/jpang/models/modelscope/Qwen/Qwen3-4B}"
BACKEND="${BACKEND:-vllm}"
GPU_ID="${GPU_ID:-1}"
DATA_PATH="${DATA_PATH:-thesis_exp/data/processed/edubench_scoring_all.jsonl}"
SPLIT_DIR="${SPLIT_DIR:-thesis_exp/data/splits/question_seed42}"
OUT_DIR="${OUT_DIR:-thesis_exp/exp17_low_score_evidence/outputs/exp19_r0a_qwen4b_direct_baseline_seed42}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DTYPE="${DTYPE:-auto}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-0}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"
OVERWRITE="${OVERWRITE:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ -f "${HOME}/miniconda3/bin/activate" ]]; then
  if [[ "${CONDA_ENV}" != "base" && -d "${HOME}/miniconda3/envs/${CONDA_ENV}" ]]; then
    source "${HOME}/miniconda3/bin/activate" "${CONDA_ENV}"
  fi
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

args=(
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --backend "${BACKEND}"
  --data_path "${DATA_PATH}"
  --split_dir "${SPLIT_DIR}"
  --out_dir "${OUT_DIR}"
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --seed "${SEED}"
  --batch_size "${BATCH_SIZE}"
  --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}"
  --dtype "${DTYPE}"
  --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}"
  --max_model_len "${MAX_MODEL_LEN}"
)

if [[ "${MAX_EXAMPLES}" != "0" ]]; then
  args+=(--max_examples "${MAX_EXAMPLES}")
fi
if [[ "${ENABLE_THINKING}" == "1" ]]; then
  args+=(--enable_thinking)
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  args+=(--overwrite)
fi

cat <<CONFIG
Exp19-R0A Qwen3-4B direct scoring baseline
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}
BACKEND=${BACKEND}
GPU_ID=${GPU_ID}
DATA_PATH=${DATA_PATH}
SPLIT_DIR=${SPLIT_DIR}
OUT_DIR=${OUT_DIR}
TEMPERATURE=${TEMPERATURE}
TOP_P=${TOP_P}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS}
SEED=${SEED}
BATCH_SIZE=${BATCH_SIZE}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}
ENABLE_THINKING=${ENABLE_THINKING}
CONFIG

python thesis_exp/exp17_low_score_evidence/run_exp19_r0a_qwen4b_direct_baseline.py "${args[@]}"

echo "Exp19-R0A completed. Lightweight summaries are under ${OUT_DIR}/tables, ${OUT_DIR}/reports, and ${OUT_DIR}/decision."
