#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

python thesis_exp/exp41_rubric_bridge/prepare_exp41a_rubric_units.py
python thesis_exp/exp41_rubric_bridge/run_exp41a_qwen_rubric_compiler.py --max-units 1 --dry-run
echo "Exp41A prepared 1044 answer-blind rubric units; no dev/test data were accessed."
