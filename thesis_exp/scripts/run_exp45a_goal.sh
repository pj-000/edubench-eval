#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${EDUBENCH_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"
PYTHON="${PYTHON:-/home/jpang/miniconda3/envs/llama_factory/bin/python}"
[[ -x "${PYTHON}" ]] || PYTHON="$(command -v python3)"
OUT="thesis_exp/exp45_dopr_head/outputs/exp45a_dopr_seed42"
mkdir -p "${OUT}/state" "${OUT}/logs_private"
trap 'code=$?; printf "%s\n" "${code}" >"${OUT}/state/goal_exit_code.txt"' EXIT

"${PYTHON}" -m py_compile \
  thesis_exp/exp45_dopr_head/common.py \
  thesis_exp/exp45_dopr_head/resolve_exp45a_inputs.py \
  thesis_exp/exp45_dopr_head/train_or_restore_exp45a_e4_encoders.py \
  thesis_exp/exp45_dopr_head/extract_exp45a_frozen_embeddings.py \
  thesis_exp/exp45_dopr_head/diagnose_exp45a_train_prototypes.py \
  thesis_exp/exp45_dopr_head/modeling_exp45a_heads.py \
  thesis_exp/exp45_dopr_head/train_exp45a_decoupled_heads.py \
  thesis_exp/exp45_dopr_head/collect_exp45a_seed42.py \
  thesis_exp/exp45_dopr_head/bootstrap_exp45a_question_key.py \
  thesis_exp/exp45_dopr_head/analyze_exp45a_decision.py
for script in thesis_exp/scripts/run_exp45a_prepare.sh thesis_exp/scripts/run_exp45a_encoder_smoke.sh thesis_exp/scripts/run_exp45a_encoders_and_embeddings.sh thesis_exp/scripts/run_exp45a_prototype_diagnostic.sh thesis_exp/scripts/run_exp45a_heads.sh thesis_exp/scripts/run_exp45a_collect.sh thesis_exp/scripts/run_exp45a_goal.sh; do bash -n "${script}"; done

bash thesis_exp/scripts/run_exp45a_prepare.sh
bash thesis_exp/scripts/run_exp45a_encoder_smoke.sh
bash thesis_exp/scripts/run_exp45a_encoders_and_embeddings.sh
bash thesis_exp/scripts/run_exp45a_prototype_diagnostic.sh
prototype_status=$("${PYTHON}" -c 'import json; print(json.load(open("thesis_exp/exp45_dopr_head/outputs/exp45a_dopr_seed42/decision/exp45a_prototype_signal_decision.json"))["status"])')
if [[ "${prototype_status}" != "PROTOTYPE_SIGNAL_GO" ]]; then
  printf "%s\n" "${prototype_status}" >"${OUT}/state/final_status.txt"
  echo "Exp45A stopped at preregistered prototype gate: ${prototype_status}"
  exit 0
fi
bash thesis_exp/scripts/run_exp45a_heads.sh
bash thesis_exp/scripts/run_exp45a_collect.sh
"${PYTHON}" -c 'import json; print(json.load(open("thesis_exp/exp45_dopr_head/outputs/exp45a_dopr_seed42/decision/exp45a_seed42_decision.json"))["status"])' >"${OUT}/state/final_status.txt"
cat "${OUT}/state/final_status.txt"
