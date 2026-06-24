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
  thesis_exp/configs/exp15_posthoc_ordinal_calibration
  thesis_exp/scripts/run_exp15_posthoc_ordinal_calibration.sh
  thesis_exp/scripts/sync_exp15_posthoc_ordinal_calibration_to_server.sh
  thesis_exp/src/edujudge/exp15_posthoc_ordinal_calibration
  thesis_exp/src/edujudge/exp12_monotonic_projection_map_oc/monotone_projection.py
  thesis_exp/outputs/exp15_posthoc_ordinal_calibration
)

cat <<CONFIG
Sync Exp15 post-hoc ordinal calibration to server
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
  --exclude='*.jsonl' \
  --exclude='predictions/' \
  --exclude='arrays/' \
  --exclude='runs/' \
  --exclude='logs/' \
  --stats \
  -e "${RSYNC_SSH}" \
  "${FILES[@]}" \
  "${SERVER_HOST}:${SERVER_REPO%/}/"

ssh "${SSH_OPTS[@]}" "${SERVER_HOST}" \
  "export LC_ALL=C LANG=C && cd ${SERVER_REPO} && chmod +x thesis_exp/scripts/run_exp15_posthoc_ordinal_calibration.sh thesis_exp/scripts/sync_exp15_posthoc_ordinal_calibration_to_server.sh && python -m compileall thesis_exp/src/edujudge/exp15_posthoc_ordinal_calibration >/tmp/exp15_compile.log && ./thesis_exp/scripts/run_exp15_posthoc_ordinal_calibration.sh && git status --short thesis_exp/src/edujudge/exp15_posthoc_ordinal_calibration thesis_exp/scripts/run_exp15_posthoc_ordinal_calibration.sh thesis_exp/scripts/sync_exp15_posthoc_ordinal_calibration_to_server.sh thesis_exp/outputs/exp15_posthoc_ordinal_calibration | sed -n '1,220p'"
