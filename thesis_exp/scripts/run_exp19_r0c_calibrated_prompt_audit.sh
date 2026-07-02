#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-vllm_qwen_env}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/jpang/models/modelscope/Qwen/Qwen3-4B}"
BACKEND="${BACKEND:-vllm}"
GPU_ID="${GPU_ID:-1}"
DATA_PATH="${DATA_PATH:-thesis_exp/data/splits/question_seed42/dev.jsonl}"
R0A_DIR="${R0A_DIR:-thesis_exp/exp17_low_score_evidence/outputs/exp19_r0a_qwen4b_direct_baseline_seed42}"
R0B_DIR="${R0B_DIR:-thesis_exp/exp17_low_score_evidence/outputs/exp19_r0b_failure_first_prompt_audit_seed42}"
D1_DIR="${D1_DIR:-thesis_exp/exp17_low_score_evidence/outputs/d1_hidden_failure_audit_seed42_dev/summary_human_rationale_recovered}"
OUT_DIR="${OUT_DIR:-thesis_exp/exp17_low_score_evidence/outputs/exp19_r0c_calibrated_prompt_audit_seed42}"
PROMPT_CONFIGS="${PROMPT_CONFIGS:-R0C_0_exact_r0a_reproduction R0C_1_balanced_failure_first R0C_2_evidence_required_cap R0C_3_two_pass_balanced R0C_4_high_score_protection}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-224}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DTYPE="${DTYPE:-auto}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-0}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"
ALLOW_REGEX_SCORE_FALLBACK="${ALLOW_REGEX_SCORE_FALLBACK:-0}"
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
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

args=(
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --backend "${BACKEND}"
  --data_path "${DATA_PATH}"
  --r0a_dir "${R0A_DIR}"
  --r0b_dir "${R0B_DIR}"
  --d1_dir "${D1_DIR}"
  --out_dir "${OUT_DIR}"
  --prompt_configs "${PROMPT_CONFIGS}"
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
if [[ "${ALLOW_REGEX_SCORE_FALLBACK}" == "1" ]]; then
  args+=(--allow_regex_score_fallback)
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  args+=(--overwrite)
fi

cat <<CONFIG
Exp19-R0C Qwen3-4B calibrated failure-first prompt audit
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}
BACKEND=${BACKEND}
GPU_ID=${GPU_ID}
DATA_PATH=${DATA_PATH}
R0A_DIR=${R0A_DIR}
R0B_DIR=${R0B_DIR}
D1_DIR=${D1_DIR}
OUT_DIR=${OUT_DIR}
PROMPT_CONFIGS=${PROMPT_CONFIGS}
TEMPERATURE=${TEMPERATURE}
TOP_P=${TOP_P}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS}
SEED=${SEED}
BATCH_SIZE=${BATCH_SIZE}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}
ENABLE_THINKING=${ENABLE_THINKING}
CONFIG

python thesis_exp/exp17_low_score_evidence/run_exp19_r0c_calibrated_prompt_audit.py "${args[@]}"

echo "Exp19-R0C completed. Lightweight summaries are under ${OUT_DIR}/tables, ${OUT_DIR}/reports, and ${OUT_DIR}/decision."
