#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
SERVER_HOST="${SERVER_HOST:-jpang@39.103.98.135}"
SERVER_PORT="${SERVER_PORT:-23722}"
SERVER_REPO="${SERVER_REPO:-~/edubench-eval-exp2}"
RSYNC_SSH="ssh -p ${SERVER_PORT} -o BatchMode=yes -o ConnectTimeout=10"
FILES=(
  thesis_exp/exp36_safer_score
  thesis_exp/scripts/run_exp36a_oof_baseline.sh
  thesis_exp/scripts/run_exp36a_safer_smoke.sh
  thesis_exp/scripts/run_exp36a_seed42_scout.sh
  thesis_exp/scripts/run_exp36a_multiseed_after_gate.sh
  thesis_exp/scripts/sync_exp36a_to_server.sh
)
rsync -azR --exclude='outputs/' --exclude='__pycache__/' --exclude='*.pyc' --exclude='private/' --exclude='*.jsonl' --exclude='*.pt' --exclude='*.safetensors' --exclude='*.log' -e "${RSYNC_SSH}" "${FILES[@]}" "${SERVER_HOST}:${SERVER_REPO%/}/"
# Teacher private files are pre-existing server inputs and are never copied by this script.
ssh -p "${SERVER_PORT}" -o BatchMode=yes -o ConnectTimeout=10 "${SERVER_HOST}" \
  "cd ${SERVER_REPO} && chmod +x thesis_exp/scripts/run_exp36a_*.sh thesis_exp/scripts/sync_exp36a_to_server.sh && python -m py_compile thesis_exp/exp36_safer_score/*.py && bash -n thesis_exp/scripts/run_exp36a_oof_baseline.sh thesis_exp/scripts/run_exp36a_safer_smoke.sh thesis_exp/scripts/run_exp36a_seed42_scout.sh thesis_exp/scripts/run_exp36a_multiseed_after_gate.sh"
echo "Exp36A code synced to ${SERVER_HOST}:${SERVER_REPO}"
