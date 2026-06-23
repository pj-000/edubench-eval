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
  thesis_exp/configs/exp13_risk_boundary_map_oc
  thesis_exp/scripts/run_exp13_risk_boundary_map_oc.sh
  thesis_exp/scripts/run_exp13_risk_boundary_map_oc_formal_locked.sh
  thesis_exp/scripts/run_exp13_risk_boundary_map_oc_smoke.sh
  thesis_exp/scripts/sync_exp13_risk_boundary_map_oc_to_server.sh
  thesis_exp/src/edujudge/exp09_pairwise_ordinal
  thesis_exp/src/edujudge/exp12_monotonic_projection_map_oc
  thesis_exp/src/edujudge/exp13_risk_boundary_map_oc
  thesis_exp/outputs/exp13_risk_boundary_map_oc
)

cat <<CONFIG
Sync Exp13 risk-boundary MAP-OC scaffold to server
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
  "export LC_ALL=C LANG=C && cd ${SERVER_REPO} && chmod +x thesis_exp/scripts/run_exp13_risk_boundary_map_oc.sh thesis_exp/scripts/run_exp13_risk_boundary_map_oc_formal_locked.sh thesis_exp/scripts/run_exp13_risk_boundary_map_oc_smoke.sh thesis_exp/scripts/sync_exp13_risk_boundary_map_oc_to_server.sh && python -m compileall thesis_exp/src/edujudge/exp13_risk_boundary_map_oc thesis_exp/src/edujudge/exp09_pairwise_ordinal >/tmp/exp13_compile.log && python -m thesis_exp.src.edujudge.exp13_risk_boundary_map_oc.validate_formal_lock && ./thesis_exp/scripts/run_exp13_risk_boundary_map_oc_smoke.sh && MODE=scout EVAL_TEST=0 python -m thesis_exp.src.edujudge.exp13_risk_boundary_map_oc.collect_exp13_results && python -m thesis_exp.src.edujudge.exp13_risk_boundary_map_oc.readability_check_exp13 && git status --short thesis_exp/src/edujudge/exp09_pairwise_ordinal thesis_exp/src/edujudge/exp13_risk_boundary_map_oc thesis_exp/configs/exp13_risk_boundary_map_oc thesis_exp/scripts/run_exp13_risk_boundary_map_oc.sh thesis_exp/scripts/run_exp13_risk_boundary_map_oc_formal_locked.sh thesis_exp/scripts/run_exp13_risk_boundary_map_oc_smoke.sh thesis_exp/scripts/sync_exp13_risk_boundary_map_oc_to_server.sh thesis_exp/outputs/exp13_risk_boundary_map_oc | sed -n '1,220p'"
