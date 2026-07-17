#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
SERVER_HOST="${SERVER_HOST:-jpang@39.103.98.135}"
SERVER_PORT="${SERVER_PORT:-23722}"
SERVER_REPO="${SERVER_REPO:-~/edubench-eval-exp2}"
RSYNC_SSH="ssh -p ${SERVER_PORT} -o BatchMode=yes -o ConnectTimeout=10"
FILES=(
  thesis_exp/exp50_cahs
  thesis_exp/configs/exp50_cahs
  thesis_exp/tests/test_exp50_targets.py
  thesis_exp/tests/test_exp50_loss_equivalence.py
  thesis_exp/tests/test_exp50_contract.py
  thesis_exp/scripts/run_exp50_cpu_smoke.sh
  thesis_exp/scripts/run_exp50_determinism_smoke.sh
  thesis_exp/scripts/run_exp50_seed42.sh
  thesis_exp/scripts/sync_exp50_to_server.sh
  thesis_exp/outputs/exp50_cahs/audit
)
rsync -azR --exclude='__pycache__/' --exclude='*.pyc' -e "${RSYNC_SSH}" "${FILES[@]}" "${SERVER_HOST}:${SERVER_REPO%/}/"
ssh -p "${SERVER_PORT}" -o BatchMode=yes "${SERVER_HOST}" "bash -lc 'source ~/miniconda3/bin/activate llama_factory && cd ${SERVER_REPO} && export PYTHONPATH=\"\$(pwd):\${PYTHONPATH:-}\" && chmod +x thesis_exp/scripts/run_exp50_*.sh thesis_exp/scripts/sync_exp50_to_server.sh && python -m py_compile thesis_exp/exp50_cahs/*.py && bash -n thesis_exp/scripts/run_exp50_*.sh thesis_exp/scripts/sync_exp50_to_server.sh'"
echo "Exp50 code synced without test data."
