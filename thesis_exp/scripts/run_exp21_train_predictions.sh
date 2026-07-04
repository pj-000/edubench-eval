#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-llama_factory}"
GPU_LIST="${GPU_LIST:-0 1}"
OUT_DIR="${OUT_DIR:-thesis_exp/exp17_low_score_evidence/outputs/exp21_d1_like_risk_annotation_seed42}"
CONFIG_DIR="${CONFIG_DIR:-${OUT_DIR}/train_predict_configs}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/train_predict_logs}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

RUN_NAMES=(
  "r5g_a3_real_only_s50_b0p05_lr5em6"
  "r5h_h1_from_r5f2_real_highprotect_s10_b0p02_lr1e6"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

if ! command -v llamafactory-cli >/dev/null 2>&1; then
  echo "ERROR: llamafactory-cli not found in CONDA_ENV=${CONDA_ENV}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

echo "Preparing Exp21 train prediction datasets/configs..."
python thesis_exp/exp17_low_score_evidence/construct_exp21_d1_like_risk_annotation_candidates.py

read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -lt 1 ]]; then
  echo "ERROR: GPU_LIST is empty" >&2
  exit 1
fi

run_one() {
  local run_name="$1"
  local gpu_id="$2"
  local config_path="${CONFIG_DIR}/${run_name}.yaml"
  local run_out="${OUT_DIR}/train_predictions/${run_name}"
  local log_path="${LOG_DIR}/predict_train_${run_name}_gpu${gpu_id}.log"

  if [[ ! -f "${config_path}" ]]; then
    echo "ERROR: missing train predict config: ${config_path}" >&2
    exit 1
  fi
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${run_out}/generated_predictions.jsonl" ]]; then
    echo "Skipping ${run_name}: generated_predictions.jsonl exists (${run_out})"
    return 0
  fi
  rm -rf "${run_out}"
  mkdir -p "${run_out}"
  echo "Starting Exp21 train prediction ${run_name} on GPU ${gpu_id}; log=${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" llamafactory-cli train "${config_path}" 2>&1 | tee "${log_path}"
  echo "Completed Exp21 train prediction ${run_name}"
}

pids=()
for idx in "${!RUN_NAMES[@]}"; do
  gpu_id="${GPUS[$((idx % ${#GPUS[@]}))]}"
  run_one "${RUN_NAMES[$idx]}" "${gpu_id}" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

echo "Collecting Exp21 train annotation candidates..."
python thesis_exp/exp17_low_score_evidence/construct_exp21_d1_like_risk_annotation_candidates.py
echo "Exp21 train prediction and candidate construction completed."
