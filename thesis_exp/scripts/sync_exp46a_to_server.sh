#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${EDUBENCH_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"
SERVER_HOST="${SERVER_HOST:-jpang@39.103.98.135}"
SERVER_PORT="${SERVER_PORT:-23722}"
SERVER_REPO="${SERVER_REPO:-~/edubench-eval-exp2}"
RSYNC_SSH="ssh -p ${SERVER_PORT} -o BatchMode=yes -o ConnectTimeout=10"
rsync -azR --exclude='__pycache__/' --exclude='*.pyc' --exclude='*.pt' --exclude='*.jsonl' --exclude='logs_private/' --exclude='runs/' -e "${RSYNC_SSH}" \
  thesis_exp/exp46_hato_kd thesis_exp/scripts/run_exp46a_matrix.sh thesis_exp/scripts/run_exp46a_teacher_smoke.sh \
  thesis_exp/scripts/run_exp46a_teacher_groupcv.sh thesis_exp/scripts/run_exp46a_teacher_collect.sh \
  thesis_exp/scripts/run_exp46a_student_smoke.sh thesis_exp/scripts/run_exp46a_student_groupcv.sh thesis_exp/scripts/run_exp46a_student_collect.sh \
  thesis_exp/scripts/run_exp46a_goal.sh thesis_exp/scripts/sync_exp46a_to_server.sh \
  "${SERVER_HOST}:${SERVER_REPO%/}/"
ssh -p "${SERVER_PORT}" -o BatchMode=yes -o ConnectTimeout=10 "${SERVER_HOST}" \
  "cd ${SERVER_REPO} && chmod +x thesis_exp/scripts/run_exp46a_*.sh thesis_exp/scripts/sync_exp46a_to_server.sh && /home/jpang/miniconda3/envs/llama_factory/bin/python -m compileall -q thesis_exp/exp46_hato_kd && for f in thesis_exp/scripts/run_exp46a_*.sh thesis_exp/scripts/sync_exp46a_to_server.sh; do bash -n \"\$f\"; done"
