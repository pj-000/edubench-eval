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
  thesis_exp/configs/exp10_qdpr2_module_ablation
  thesis_exp/scripts/run_exp10_qdpr2_module_ablation.sh
  thesis_exp/scripts/sync_exp10_qdpr2_module_ablation_to_server.sh
  thesis_exp/src/edujudge/exp09_pairwise_ordinal
  thesis_exp/src/edujudge/exp10_qdpr2_module_ablation
  thesis_exp/outputs/exp10_qdpr2_module_ablation/configs
  thesis_exp/outputs/exp10_qdpr2_module_ablation/tables
  thesis_exp/outputs/exp10_qdpr2_module_ablation/reports
  thesis_exp/outputs/exp10_qdpr2_module_ablation/sanity_check_exp10_setup.md
  thesis_exp/outputs/exp10_qdpr2_module_ablation/readability_check_exp10.md
)

cat <<CONFIG
Sync Exp10 QD-PR2 module ablation scaffold to server
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
  "export LC_ALL=C LANG=C && cd ${SERVER_REPO} && chmod +x thesis_exp/scripts/run_exp10_qdpr2_module_ablation.sh thesis_exp/scripts/sync_exp10_qdpr2_module_ablation_to_server.sh && python -m py_compile thesis_exp/src/edujudge/exp09_pairwise_ordinal/*.py thesis_exp/src/edujudge/exp10_qdpr2_module_ablation/*.py && bash -n thesis_exp/scripts/run_exp10_qdpr2_module_ablation.sh && bash -n thesis_exp/scripts/sync_exp10_qdpr2_module_ablation_to_server.sh && python -m thesis_exp.src.edujudge.exp10_qdpr2_module_ablation.sanity_check_exp10_setup && python -m thesis_exp.src.edujudge.exp10_qdpr2_module_ablation.collect_exp10_results && python -m thesis_exp.src.edujudge.exp10_qdpr2_module_ablation.readability_check_exp10 && git status --short thesis_exp/src/edujudge/exp09_pairwise_ordinal thesis_exp/src/edujudge/exp10_qdpr2_module_ablation thesis_exp/configs/exp10_qdpr2_module_ablation thesis_exp/scripts/run_exp10_qdpr2_module_ablation.sh thesis_exp/scripts/sync_exp10_qdpr2_module_ablation_to_server.sh thesis_exp/outputs/exp10_qdpr2_module_ablation | sed -n '1,200p'"
