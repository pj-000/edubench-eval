#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${RUN_API:-0}" != "1" ]]; then
  echo "Exp39B-R1 API execution requires RUN_API=1" >&2
  exit 2
fi
OUT_DIR="${EXP39B_R1_OUT_DIR:-thesis_exp/exp39b_educfa_rlcr/outputs/exp39b_r1_response_disjoint_pilot_seed44}"
WORKERS="${API_WORKERS:-4}"
python thesis_exp/exp39b_educfa_rlcr/run_exp39b_qwen_clause_planner.py --out-dir "${OUT_DIR}" --workers "${WORKERS}"
python thesis_exp/exp39b_educfa_rlcr/validate_exp39b_clause_plans.py --out-dir "${OUT_DIR}"
python thesis_exp/exp39b_educfa_rlcr/run_exp39b_qwen_span_editor.py --out-dir "${OUT_DIR}" --workers "${WORKERS}"
python thesis_exp/exp39b_educfa_rlcr/apply_exp39b_locked_span_patch.py --out-dir "${OUT_DIR}"
