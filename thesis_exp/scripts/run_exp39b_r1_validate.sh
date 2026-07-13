#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

OUT_DIR="${EXP39B_R1_OUT_DIR:-thesis_exp/exp39b_educfa_rlcr/outputs/exp39b_r1_response_disjoint_pilot_seed44}"
PLAN_STATUS="$(python - "${OUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1]) / "decision/exp39b_plan_validation_decision.json"
print(json.loads(path.read_text(encoding="utf-8"))["status"])
PY
)"
if [[ "${PLAN_STATUS}" == "PLAN_VALIDATION_GO" ]]; then
  python thesis_exp/exp39b_educfa_rlcr/validate_exp39b_pilot.py --out-dir "${OUT_DIR}"
fi
python thesis_exp/exp39b_educfa_rlcr/analyze_exp39b_r1_pilot.py --out-dir "${OUT_DIR}"
