#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

STATUS="$(python - <<'PY'
import json
from pathlib import Path
path = Path("thesis_exp/exp39b_educfa_rlcr/outputs/exp39b_rlcr_pilot_seed43/decision/exp39b_protocol_prepare_decision.json")
print(json.loads(path.read_text(encoding="utf-8"))["status"])
PY
)"

if [[ "${STATUS}" == "PROTOCOL_PREPARE_GO" ]]; then
  python thesis_exp/exp39b_educfa_rlcr/validate_exp39b_pilot.py
fi
python thesis_exp/exp39b_educfa_rlcr/analyze_exp39b_pilot.py
