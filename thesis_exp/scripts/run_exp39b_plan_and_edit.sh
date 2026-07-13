#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${RUN_API:-0}" != "1" ]]; then
  echo "Exp39B API execution requires RUN_API=1" >&2
  exit 2
fi

python thesis_exp/exp39b_educfa_rlcr/run_exp39b_qwen_clause_planner.py
python thesis_exp/exp39b_educfa_rlcr/validate_exp39b_clause_plans.py
python thesis_exp/exp39b_educfa_rlcr/run_exp39b_qwen_span_editor.py
python thesis_exp/exp39b_educfa_rlcr/apply_exp39b_locked_span_patch.py
