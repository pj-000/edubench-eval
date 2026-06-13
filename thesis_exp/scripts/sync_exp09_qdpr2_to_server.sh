#!/usr/bin/env bash
set -euo pipefail

SERVER_HOST="${SERVER_HOST:-jpang@39.103.98.135}"
SERVER_PORT="${SERVER_PORT:-23722}"
SERVER_REPO="${SERVER_REPO:-~/edubench-eval-exp2}"
RSYNC_SSH="ssh -p ${SERVER_PORT} -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=2"
SSH_OPTS=(
  -p "${SERVER_PORT}"
  -o BatchMode=yes
  -o ConnectTimeout=10
  -o ServerAliveInterval=5
  -o ServerAliveCountMax=2
)
FILES=(
  thesis_exp/configs/exp09_pairwise_ordinal
  thesis_exp/scripts/run_exp09_qdpr2_smoke.sh
  thesis_exp/scripts/run_exp09_qdpr2_train.sh
  thesis_exp/scripts/sync_exp09_qdpr2_to_server.sh
  thesis_exp/src/edujudge/exp09_pairwise_ordinal
  thesis_exp/outputs/exp09_pairwise_ordinal_qdpr2/configs
  thesis_exp/outputs/exp09_pairwise_ordinal_qdpr2/tables
  thesis_exp/outputs/exp09_pairwise_ordinal_qdpr2/reports
  thesis_exp/outputs/exp09_pairwise_ordinal_qdpr2/sanity_check_qdpr2_setup.md
  thesis_exp/outputs/exp09_pairwise_ordinal_qdpr2/readability_check_qdpr2.md
)

cat <<CONFIG
Sync Exp9 QD-PR2 anchored pairwise scaffold to server
SERVER_HOST=${SERVER_HOST}
SERVER_PORT=${SERVER_PORT}
SERVER_REPO=${SERVER_REPO}
CONFIG

rsync -azR \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='*.bin' \
  --exclude='*.ckpt' \
  --exclude='*.safetensors' \
  --exclude='*.npy' \
  --exclude='*.npz' \
  --exclude='pairs/' \
  --exclude='runs/' \
  --exclude='smoke_test/' \
  --exclude='logs/' \
  --stats \
  -e "${RSYNC_SSH}" \
  "${FILES[@]}" \
  "${SERVER_HOST}:${SERVER_REPO%/}/"

ssh "${SSH_OPTS[@]}" "${SERVER_HOST}" \
  "export LC_ALL=C LANG=C && cd ${SERVER_REPO} && chmod +x thesis_exp/scripts/run_exp09_qdpr2_smoke.sh thesis_exp/scripts/run_exp09_qdpr2_train.sh thesis_exp/scripts/sync_exp09_qdpr2_to_server.sh && python -m py_compile thesis_exp/src/edujudge/exp09_pairwise_ordinal/*.py && bash -n thesis_exp/scripts/run_exp09_qdpr2_smoke.sh && bash -n thesis_exp/scripts/run_exp09_qdpr2_train.sh && bash -n thesis_exp/scripts/sync_exp09_qdpr2_to_server.sh && python -m thesis_exp.src.edujudge.exp09_pairwise_ordinal.sanity_check_qdpr2_setup && python -m thesis_exp.src.edujudge.exp09_pairwise_ordinal.readability_check_qdpr2 && git status --short thesis_exp/src/edujudge/exp09_pairwise_ordinal thesis_exp/configs/exp09_pairwise_ordinal thesis_exp/scripts/run_exp09_qdpr2_smoke.sh thesis_exp/scripts/run_exp09_qdpr2_train.sh thesis_exp/scripts/sync_exp09_qdpr2_to_server.sh thesis_exp/outputs/exp09_pairwise_ordinal_qdpr2 | sed -n '1,160p'"
