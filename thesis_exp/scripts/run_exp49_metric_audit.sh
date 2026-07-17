#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m thesis_exp.exp49_cphce.audit_history_registry
"${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path
p = Path('thesis_exp/outputs/exp49_cphce/audit/bin_agreement_audit.json')
value = json.loads(p.read_text())
if value['status'] != 'REPRODUCED_SPLIT_TOLERANT':
    raise SystemExit(f"Bin Agreement audit did not reproduce: {value['status']}")
print('Exp49 metric/history audit passed')
PY
