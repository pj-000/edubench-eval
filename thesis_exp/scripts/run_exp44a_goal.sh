#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${EDUBENCH_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"
PYTHON="${PYTHON:-/home/jpang/miniconda3/envs/llama_factory/bin/python}"
OUT="thesis_exp/exp44_taco_score/outputs/exp44a_taco_seed42"
mkdir -p "${OUT}/state"
trap 'code=$?; printf "%s\n" "${code}" >"${OUT}/state/goal_exit_code.txt"' EXIT

"${PYTHON}" -m py_compile \
  thesis_exp/exp44_taco_score/common.py \
  thesis_exp/exp44_taco_score/resolve_exp44a_exp43_inputs.py \
  thesis_exp/exp44_taco_score/prepare_exp44a_triplets.py \
  thesis_exp/exp44_taco_score/modeling_exp44a_taco.py \
  thesis_exp/exp44_taco_score/losses_exp44a_taco.py \
  thesis_exp/exp44_taco_score/audit_exp44a_loss_scales.py \
  thesis_exp/exp44_taco_score/train_exp44a_groupcv.py \
  thesis_exp/exp44_taco_score/collect_exp44a_seed42.py \
  thesis_exp/exp44_taco_score/bootstrap_exp44a_question_key.py \
  thesis_exp/exp44_taco_score/analyze_exp44a_decision.py
for script in thesis_exp/scripts/run_exp44a_prepare.sh thesis_exp/scripts/run_exp44a_smoke.sh thesis_exp/scripts/run_exp44a_seed42_groupcv.sh thesis_exp/scripts/run_exp44a_collect.sh thesis_exp/scripts/run_exp44a_goal.sh; do bash -n "${script}"; done

bash thesis_exp/scripts/run_exp44a_prepare.sh
[[ "$("${PYTHON}" -c 'import json; print(json.load(open("thesis_exp/exp44_taco_score/outputs/exp44a_taco_seed42/decision/exp44a_data_decision.json"))["status"])')" == "TRIPLET_DATA_GO" ]]
bash thesis_exp/scripts/run_exp44a_smoke.sh
[[ "$("${PYTHON}" -c 'import json; print(json.load(open("thesis_exp/exp44_taco_score/outputs/exp44a_taco_seed42/decision/exp44a_loss_scale_decision.json"))["status"])')" == "LOSS_SCALE_GO" ]]
[[ "$("${PYTHON}" -c 'import json; print(json.load(open("thesis_exp/exp44_taco_score/outputs/exp44a_taco_seed42/decision/exp44a_smoke_decision.json"))["status"])')" == "SMOKE_GO" ]]
bash thesis_exp/scripts/run_exp44a_seed42_groupcv.sh
bash thesis_exp/scripts/run_exp44a_collect.sh

