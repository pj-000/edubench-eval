#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-llama_factory}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
DEV_PATH="${DEV_PATH:-thesis_exp/data/splits/question_seed42/dev.jsonl}"
GPU_LIST="${GPU_LIST:-6}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PRECISION="${PRECISION:-bf16}"
QMR_RUN_DIR="${QMR_RUN_DIR:-thesis_exp/outputs/exp16_boundary_linking/runs/qmr/seed_42}"
METRIC_RUBRIC_RUN_DIR="${METRIC_RUBRIC_RUN_DIR:-thesis_exp/outputs/exp16_boundary_linking/runs/metric_rubric/seed_42}"
QMR_CKPT="${QMR_CKPT:-${QMR_RUN_DIR}/checkpoint_best/state_dict.pt}"
METRIC_RUBRIC_CKPT="${METRIC_RUBRIC_CKPT:-${METRIC_RUBRIC_RUN_DIR}/checkpoint_best/state_dict.pt}"
CACHE_ROOT="${CACHE_ROOT:-thesis_exp/outputs/exp16_boundary_linking/rq1_cache_eval}"
ORIGINAL_DIAG_DIR="${ORIGINAL_DIAG_DIR:-thesis_exp/outputs/exp16_boundary_linking/rq1_diagnosis}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'USAGE'
Run Exp16A RQ1 eval-time boundary-cache diagnosis for seed42 qmr and metric_rubric.

Defaults:
  QMR_RUN_DIR=thesis_exp/outputs/exp16_boundary_linking/runs/qmr/seed_42
  METRIC_RUBRIC_RUN_DIR=thesis_exp/outputs/exp16_boundary_linking/runs/metric_rubric/seed_42
  DEV_PATH=thesis_exp/data/splits/question_seed42/dev.jsonl
  CACHE_ROOT=thesis_exp/outputs/exp16_boundary_linking/rq1_cache_eval

The script exports raw cache predictions inside CACHE_ROOT/scout_seed42/{variant}/, then writes
lightweight CSV/MD diagnostics under CACHE_ROOT/rq1_diagnosis/ and CACHE_ROOT/.
Do not commit the raw predictions jsonl files.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -f "${HOME}/miniconda3/bin/activate" ]]; then
  if [[ "${CONDA_ENV}" != "base" && ! -d "${HOME}/miniconda3/envs/${CONDA_ENV}" ]]; then
    echo "WARNING: conda env ${CONDA_ENV} was not found; using current shell." >&2
  elif ! source "${HOME}/miniconda3/bin/activate" "${CONDA_ENV}"; then
    echo "WARNING: could not activate conda env ${CONDA_ENV}; using current shell." >&2
  fi
fi

missing=0
for path in "${DEV_PATH}" "${QMR_RUN_DIR}/config.json" "${METRIC_RUBRIC_RUN_DIR}/config.json" "${QMR_CKPT}" "${METRIC_RUBRIC_CKPT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: required path is missing: ${path}" >&2
    missing=1
  fi
done
if [[ "${missing}" == "1" ]]; then
  cat >&2 <<CONFIG_HELP

Set paths explicitly if this server uses a different layout, for example:
  QMR_RUN_DIR=/path/to/qmr/seed_42 \\
  METRIC_RUBRIC_RUN_DIR=/path/to/metric_rubric/seed_42 \\
  QMR_CKPT=/path/to/qmr/checkpoint_best/state_dict.pt \\
  METRIC_RUBRIC_CKPT=/path/to/metric_rubric/checkpoint_best/state_dict.pt \\
  MODEL_NAME_OR_PATH=/path/to/model \\
  DEV_PATH=/path/to/dev.jsonl \\
  ./thesis_exp/scripts/run_exp16a_rq1_boundary_cache_diagnosis.sh
CONFIG_HELP
  exit 1
fi

read -r -a GPU_ARRAY <<< "${GPU_LIST//,/ }"
if [[ "${#GPU_ARRAY[@]}" -lt 1 ]]; then
  echo "ERROR: GPU_LIST must be non-empty." >&2
  exit 1
fi

cat <<CONFIG
Exp16A RQ1 boundary cache diagnosis
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}
DEV_PATH=${DEV_PATH}
GPU_LIST=${GPU_LIST}
BATCH_SIZE=${BATCH_SIZE}
PRECISION=${PRECISION}
QMR_RUN_DIR=${QMR_RUN_DIR}
METRIC_RUBRIC_RUN_DIR=${METRIC_RUBRIC_RUN_DIR}
CACHE_ROOT=${CACHE_ROOT}
CONFIG

run_export() {
  local variant="$1"
  local run_dir="$2"
  local ckpt="$3"
  local gpu="$4"
  local output_dir="${CACHE_ROOT}/scout_seed42/${variant}"
  mkdir -p "${output_dir}"
  echo "Exporting ${variant} dev predictions with boundary cache on GPU ${gpu}; output=${output_dir}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    python -m thesis_exp.src.edujudge.exp16_boundary_linking.export_predictions_boundary_linking \
      --run_dir "${run_dir}" \
      --checkpoint_path "${ckpt}" \
      --output_dir "${output_dir}" \
      --model_name_or_path "${MODEL_NAME_OR_PATH}" \
      --variant "${variant}" \
      --dev_path "${DEV_PATH}" \
      --splits dev \
      --batch_size "${BATCH_SIZE}" \
      --precision "${PRECISION}" \
      --use_boundary_cache true \
      --trust_remote_code
  )
}

run_export qmr "${QMR_RUN_DIR}" "${QMR_CKPT}" "${GPU_ARRAY[0]}"
metric_gpu="${GPU_ARRAY[0]}"
if [[ "${#GPU_ARRAY[@]}" -gt 1 ]]; then
  metric_gpu="${GPU_ARRAY[1]}"
fi
run_export metric_rubric "${METRIC_RUBRIC_RUN_DIR}" "${METRIC_RUBRIC_CKPT}" "${metric_gpu}"

python -m thesis_exp.src.edujudge.exp16_boundary_linking.analyze_boundary_failure \
  --input_root "${CACHE_ROOT}" \
  --output_dir "${CACHE_ROOT}/rq1_diagnosis" \
  --variants qmr metric_rubric \
  --splits dev \
  --seed 42

python -m thesis_exp.src.edujudge.exp16_boundary_linking.compare_cache_eval \
  --original_dir "${ORIGINAL_DIAG_DIR}" \
  --cache_dir "${CACHE_ROOT}/rq1_diagnosis" \
  --output_dir "${CACHE_ROOT}"
