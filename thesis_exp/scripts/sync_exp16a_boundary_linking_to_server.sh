#!/usr/bin/env bash
set -euo pipefail

SERVER_HOST="${SERVER_HOST:-jpang@39.103.98.135}"
SERVER_PORT="${SERVER_PORT:-23722}"
SERVER_REPO="${SERVER_REPO:-~/edubench-eval-exp2}"
CONDA_ENV="${CONDA_ENV:-llama_factory}"
RSYNC_SSH="ssh -p ${SERVER_PORT} -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=2"
SSH_OPTS=(
  -p "${SERVER_PORT}"
  -o BatchMode=yes
  -o ConnectTimeout=10
  -o ServerAliveInterval=5
  -o ServerAliveCountMax=2
)
FILES=(
  thesis_exp/src/edujudge/exp16_boundary_linking
  thesis_exp/scripts/run_exp16a_boundary_linking_sanity.sh
  thesis_exp/scripts/run_exp16a_boundary_linking_scout.sh
  thesis_exp/scripts/run_exp16a_rq1_boundary_cache_diagnosis.sh
  thesis_exp/scripts/run_exp16a_rq1_stability_seeds.sh
  thesis_exp/scripts/sync_exp16a_boundary_linking_to_server.sh
)

cat <<CONFIG
Sync Exp16A boundary-linking scaffold to server
SERVER_HOST=${SERVER_HOST}
SERVER_PORT=${SERVER_PORT}
SERVER_REPO=${SERVER_REPO}
CONDA_ENV=${CONDA_ENV}
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
  --exclude='runs/' \
  --exclude='logs/' \
  --exclude='sanity/dry_run/checkpoint_best/' \
  --stats \
  -e "${RSYNC_SSH}" \
  "${FILES[@]}" \
  "${SERVER_HOST}:${SERVER_REPO%/}/"

ssh "${SSH_OPTS[@]}" "${SERVER_HOST}" \
  "bash -lc 'export LC_ALL=C LANG=C && source ~/miniconda3/bin/activate ${CONDA_ENV} && cd ${SERVER_REPO} && chmod +x thesis_exp/scripts/run_exp16a_boundary_linking_sanity.sh thesis_exp/scripts/run_exp16a_boundary_linking_scout.sh thesis_exp/scripts/run_exp16a_rq1_boundary_cache_diagnosis.sh thesis_exp/scripts/run_exp16a_rq1_stability_seeds.sh thesis_exp/scripts/sync_exp16a_boundary_linking_to_server.sh && python -m compileall thesis_exp/src/edujudge/exp16_boundary_linking && bash -n thesis_exp/scripts/run_exp16a_boundary_linking_sanity.sh && bash -n thesis_exp/scripts/run_exp16a_boundary_linking_scout.sh && bash -n thesis_exp/scripts/run_exp16a_rq1_boundary_cache_diagnosis.sh && bash -n thesis_exp/scripts/run_exp16a_rq1_stability_seeds.sh && python -m thesis_exp.src.edujudge.exp16_boundary_linking.sanity_check_boundary_linking && git status --short thesis_exp/src/edujudge/exp16_boundary_linking thesis_exp/scripts/run_exp16a_boundary_linking_sanity.sh thesis_exp/scripts/run_exp16a_boundary_linking_scout.sh thesis_exp/scripts/run_exp16a_rq1_boundary_cache_diagnosis.sh thesis_exp/scripts/run_exp16a_rq1_stability_seeds.sh thesis_exp/scripts/sync_exp16a_boundary_linking_to_server.sh thesis_exp/outputs/exp16_boundary_linking | sed -n \"1,200p\"'"
