#!/usr/bin/env bash
set -euo pipefail

SEEDS="${SEEDS:-42 43 44}"
GPU_LIST="${GPU_LIST:-6 7}"
EPOCHS="${EPOCHS:-3}"
MODE="${MODE:-formal}"
EVAL_TEST="${EVAL_TEST:-1}"
EXP13_RUNS="${EXP13_RUNS:-score_proj_l2h_lam0p20}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
RESET_RUN_DIR="${RESET_RUN_DIR:-0}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-32}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
SELECTION_DELTA="${SELECTION_DELTA:-0.005}"
EXP13_PARALLEL="${EXP13_PARALLEL:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export SEEDS
export GPU_LIST
export EPOCHS
export MODE
export EVAL_TEST
export EXP13_RUNS
export SKIP_COMPLETED
export RESET_RUN_DIR
export PER_DEVICE_TRAIN_BATCH_SIZE
export PER_DEVICE_EVAL_BATCH_SIZE
export GRADIENT_ACCUMULATION_STEPS
export LEARNING_RATE
export SELECTION_DELTA
export EXP13_PARALLEL

if [[ "${MODE}" != "formal" ]]; then
  echo "ERROR: formal locked workflow must run with MODE=formal, got ${MODE}" >&2
  exit 1
fi
if [[ "${EVAL_TEST}" != "1" ]]; then
  echo "ERROR: formal locked workflow must run with EVAL_TEST=1." >&2
  exit 1
fi

python -m thesis_exp.src.edujudge.exp13_risk_boundary_map_oc.validate_formal_lock
"${SCRIPT_DIR}/run_exp13_risk_boundary_map_oc.sh"
python -m thesis_exp.src.edujudge.exp13_risk_boundary_map_oc.collect_exp13_results \
  --mode "${MODE}" \
  --runs "${EXP13_RUNS}" \
  --delta "${SELECTION_DELTA}"
python -m thesis_exp.src.edujudge.exp13_risk_boundary_map_oc.readability_check_exp13
