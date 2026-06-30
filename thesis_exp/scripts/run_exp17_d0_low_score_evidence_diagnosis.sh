#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-llama_factory}"
PREDICTIONS_PATH="${PREDICTIONS_PATH:-thesis_exp/outputs/exp16_boundary_linking/rq1_cache_eval/scout_seed42/qmr/predictions_dev.jsonl}"
DEV_PATH="${DEV_PATH:-thesis_exp/data/splits/question_seed42/dev.jsonl}"
TABLES_DIR="${TABLES_DIR:-thesis_exp/outputs/exp17_low_score_evidence_diagnosis/tables}"
REPORTS_DIR="${REPORTS_DIR:-thesis_exp/outputs/exp17_low_score_evidence_diagnosis/reports}"
AUDIT_SIZE="${AUDIT_SIZE:-50}"
CONTROLS_PER_CASE="${CONTROLS_PER_CASE:-3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'USAGE'
Run Exp17-D0 low-score evidence diagnosis.

This script does not train a model. It reads Exp16A qmr boundary-cache dev predictions and the
original dev split, then writes lightweight CSV/MD outputs.

Defaults:
  PREDICTIONS_PATH=thesis_exp/outputs/exp16_boundary_linking/rq1_cache_eval/scout_seed42/qmr/predictions_dev.jsonl
  DEV_PATH=thesis_exp/data/splits/question_seed42/dev.jsonl
  TABLES_DIR=thesis_exp/outputs/exp17_low_score_evidence_diagnosis/tables
  REPORTS_DIR=thesis_exp/outputs/exp17_low_score_evidence_diagnosis/reports
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
for path in "${PREDICTIONS_PATH}" "${DEV_PATH}"; do
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: required input is missing: ${path}" >&2
    missing=1
  fi
done
if [[ "${missing}" == "1" ]]; then
  cat >&2 <<HELP

Run Exp16A boundary-cache diagnosis first, or set PREDICTIONS_PATH explicitly. Example:
  PREDICTIONS_PATH=/path/to/predictions_dev.jsonl \\
  DEV_PATH=/path/to/dev.jsonl \\
  ./thesis_exp/scripts/run_exp17_d0_low_score_evidence_diagnosis.sh
HELP
  exit 1
fi

cat <<CONFIG
Exp17-D0 low-score evidence diagnosis
PREDICTIONS_PATH=${PREDICTIONS_PATH}
DEV_PATH=${DEV_PATH}
TABLES_DIR=${TABLES_DIR}
REPORTS_DIR=${REPORTS_DIR}
AUDIT_SIZE=${AUDIT_SIZE}
CONTROLS_PER_CASE=${CONTROLS_PER_CASE}
CONFIG

python -m thesis_exp.src.edujudge.exp17_low_score_evidence_diagnosis.d0_low_score_evidence_diagnosis \
  --predictions_path "${PREDICTIONS_PATH}" \
  --dev_path "${DEV_PATH}" \
  --tables_dir "${TABLES_DIR}" \
  --reports_dir "${REPORTS_DIR}" \
  --audit_size "${AUDIT_SIZE}" \
  --controls_per_case "${CONTROLS_PER_CASE}"
