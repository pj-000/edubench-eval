#!/usr/bin/env bash
set -euo pipefail

SERVER_HOST="${SERVER_HOST:-jpang@39.103.98.135}"
SERVER_PORT="${SERVER_PORT:-23722}"
SERVER_REPO="${SERVER_REPO:-~/edubench-eval-exp2}"
BRANCH="${BRANCH:-main}"
SYNC_MODE="${SYNC_MODE:-copy}"
SSH_OPTS=(
  -p "${SERVER_PORT}"
  -o BatchMode=yes
  -o ConnectTimeout=10
  -o ServerAliveInterval=5
  -o ServerAliveCountMax=2
)
FILES=(
  thesis_exp/scripts/run_exp06_qds1_smoke.sh
  thesis_exp/scripts/run_exp06_qds1_train.sh
  thesis_exp/scripts/sync_exp06_qds1_to_server.sh
  thesis_exp/src/edujudge/exp06_training/__init__.py
  thesis_exp/src/edujudge/exp06_training/collect_qds1_results.py
  thesis_exp/src/edujudge/exp06_training/train_qds1_ordinal.py
)

cat <<CONFIG
Sync Exp6 QD-S1 scripts to server
SERVER_HOST=${SERVER_HOST}
SERVER_PORT=${SERVER_PORT}
SERVER_REPO=${SERVER_REPO}
BRANCH=${BRANCH}
SYNC_MODE=${SYNC_MODE}
CONFIG

if [[ "${SYNC_MODE}" == "git" ]]; then
  ssh "${SSH_OPTS[@]}" "${SERVER_HOST}" "export GIT_TERMINAL_PROMPT=0 && cd ${SERVER_REPO} && git fetch origin ${BRANCH} && git checkout ${BRANCH} && git pull --ff-only origin ${BRANCH}"
else
  COPYFILE_DISABLE=1 tar --no-xattrs -cf - "${FILES[@]}" | ssh "${SSH_OPTS[@]}" "${SERVER_HOST}" "export LC_ALL=C LANG=C && cd ${SERVER_REPO} && tar -xf - && chmod +x thesis_exp/scripts/run_exp06_qds1_smoke.sh thesis_exp/scripts/run_exp06_qds1_train.sh thesis_exp/scripts/sync_exp06_qds1_to_server.sh && python -m py_compile thesis_exp/src/edujudge/exp06_training/*.py && find thesis_exp/src/edujudge/exp06_training -name '__pycache__' -type d -prune -exec rm -rf {} + && git status --short thesis_exp/scripts/run_exp06_qds1_smoke.sh thesis_exp/scripts/run_exp06_qds1_train.sh thesis_exp/scripts/sync_exp06_qds1_to_server.sh thesis_exp/src/edujudge/exp06_training"
fi
