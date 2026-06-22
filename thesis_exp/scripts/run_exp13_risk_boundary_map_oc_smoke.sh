#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

python -m thesis_exp.src.edujudge.exp13_risk_boundary_map_oc.smoke_check_exp13
python -m thesis_exp.src.edujudge.exp13_risk_boundary_map_oc.preflight_exp13 \
  --seeds "42" \
  --gpu_list "6 7" \
  --runs "point_pair_proj_l2h_lam0p20" \
  --epochs "1" \
  --mode "scout" \
  --eval_test "0" \
  --selection_delta "0.005"
python -m thesis_exp.src.edujudge.exp13_risk_boundary_map_oc.collect_exp13_results \
  --mode "scout" \
  --runs "point_pair_proj_l2h_lam0p20" \
  --delta "0.005"
python -m thesis_exp.src.edujudge.exp13_risk_boundary_map_oc.readability_check_exp13
