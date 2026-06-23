#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

python -m thesis_exp.src.edujudge.exp14_logit_margin_tail_risk_oc.smoke_check_exp14
python -m thesis_exp.src.edujudge.exp14_logit_margin_tail_risk_oc.preflight_exp14 \
  --mode scout \
  --eval_test 0 \
  --runs "score_logit_margin_lam0p01_alllow"
python -m thesis_exp.src.edujudge.exp14_logit_margin_tail_risk_oc.collect_exp14_results \
  --mode scout \
  --runs "score_logit_margin_lam0p01_alllow"
python -m thesis_exp.src.edujudge.exp14_logit_margin_tail_risk_oc.readability_check_exp14
