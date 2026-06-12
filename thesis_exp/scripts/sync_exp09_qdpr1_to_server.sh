#!/usr/bin/env bash
set -euo pipefail

SERVER_HOST="${SERVER_HOST:-jpang@39.103.98.135}"
SERVER_PORT="${SERVER_PORT:-23722}"
SERVER_REPO="${SERVER_REPO:-~/edubench-eval-exp2}"
BRANCH="${BRANCH:-main}"
SYNC_MODE="${SYNC_MODE:-rsync}"
SSH_OPTS=(
  -p "${SERVER_PORT}"
  -o BatchMode=yes
  -o ConnectTimeout=10
  -o ServerAliveInterval=5
  -o ServerAliveCountMax=2
)
RSYNC_SSH="ssh -p ${SERVER_PORT} -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=2"
FILES=(
  thesis_exp/configs/exp09_pairwise_ordinal
  thesis_exp/scripts/run_exp09_qdpr1_smoke.sh
  thesis_exp/scripts/run_exp09_qdpr1_train.sh
  thesis_exp/scripts/sync_exp09_qdpr1_to_server.sh
  thesis_exp/src/edujudge/exp09_pairwise_ordinal
  thesis_exp/outputs/exp06_synthetic_low_score/training_datasets/QD-S0_human_only
  thesis_exp/outputs/exp06_question_disjoint_baselines/runs/QD-B0_human_only_ordinary_ordinal/tables
  thesis_exp/outputs/exp06_question_disjoint_baselines/runs/QD-B0_human_only_ordinary_ordinal/run_metadata.json
  thesis_exp/outputs/exp06_question_disjoint_baselines/runs/QD-B1_human_only_L1_weighted_ordinal/tables
  thesis_exp/outputs/exp06_question_disjoint_baselines/runs/QD-B1_human_only_L1_weighted_ordinal/run_metadata.json
  thesis_exp/outputs/exp07_rank_consistent_ordinal/runs/QD-R1_CORAL_human_only/tables
  thesis_exp/outputs/exp07_rank_consistent_ordinal/runs/QD-R1_CORAL_human_only/run_metadata.json
  thesis_exp/outputs/exp08_edurisk_loss/runs/QD-ER1_EduRisk_human_only/tables
  thesis_exp/outputs/exp08_edurisk_loss/runs/QD-ER1_EduRisk_human_only/run_metadata.json
  thesis_exp/outputs/exp09_pairwise_ordinal/tables
  thesis_exp/outputs/exp09_pairwise_ordinal/reports
  thesis_exp/outputs/exp09_pairwise_ordinal/pairs
  thesis_exp/outputs/exp09_pairwise_ordinal/report.md
  thesis_exp/outputs/exp09_pairwise_ordinal/review_package.md
  thesis_exp/outputs/exp09_pairwise_ordinal/sanity_check_exp09_setup.md
  thesis_exp/outputs/exp09_pairwise_ordinal/readability_check_exp09.md
)

cat <<CONFIG
Sync Exp9 QD-PR1 pairwise ordinal code to server
SERVER_HOST=${SERVER_HOST}
SERVER_PORT=${SERVER_PORT}
SERVER_REPO=${SERVER_REPO}
BRANCH=${BRANCH}
SYNC_MODE=${SYNC_MODE}
CONFIG

POST_SYNC_CHECK="export LC_ALL=C LANG=C && cd ${SERVER_REPO} && chmod +x thesis_exp/scripts/run_exp09_qdpr1_smoke.sh thesis_exp/scripts/run_exp09_qdpr1_train.sh thesis_exp/scripts/sync_exp09_qdpr1_to_server.sh && python -m py_compile thesis_exp/src/edujudge/exp09_pairwise_ordinal/*.py && bash -n thesis_exp/scripts/run_exp09_qdpr1_smoke.sh && bash -n thesis_exp/scripts/run_exp09_qdpr1_train.sh && bash -n thesis_exp/scripts/sync_exp09_qdpr1_to_server.sh && python -m thesis_exp.src.edujudge.exp09_pairwise_ordinal.sanity_check_exp09_setup && python -m thesis_exp.src.edujudge.exp09_pairwise_ordinal.readability_check_exp09 && find thesis_exp/src/edujudge/exp09_pairwise_ordinal -name '__pycache__' -type d -prune -exec rm -rf {} + && git status --short thesis_exp/src/edujudge/exp09_pairwise_ordinal thesis_exp/configs/exp09_pairwise_ordinal thesis_exp/scripts/run_exp09_qdpr1_smoke.sh thesis_exp/scripts/run_exp09_qdpr1_train.sh thesis_exp/scripts/sync_exp09_qdpr1_to_server.sh thesis_exp/outputs/exp09_pairwise_ordinal"

if [[ "${SYNC_MODE}" == "git" ]]; then
  ssh "${SSH_OPTS[@]}" "${SERVER_HOST}" \
    "export GIT_TERMINAL_PROMPT=0 && cd ${SERVER_REPO} && git fetch origin ${BRANCH} && git checkout ${BRANCH} && git pull --ff-only origin ${BRANCH} && ${POST_SYNC_CHECK#*&& cd ${SERVER_REPO} && }"
elif [[ "${SYNC_MODE}" == "rsync" ]]; then
  rsync -azR --exclude='__pycache__/' --exclude='*.pyc' --exclude='*.pt' --exclude='*.pth' \
    --exclude='*.bin' --exclude='*.ckpt' --exclude='*.safetensors' --exclude='*.npy' \
    --exclude='*.npz' --stats -e "${RSYNC_SSH}" "${FILES[@]}" "${SERVER_HOST}:${SERVER_REPO%/}/"
  ssh "${SSH_OPTS[@]}" "${SERVER_HOST}" "${POST_SYNC_CHECK}"
else
  COPYFILE_DISABLE=1 tar --no-xattrs -cf - "${FILES[@]}" | ssh "${SSH_OPTS[@]}" "${SERVER_HOST}" \
    "export LC_ALL=C LANG=C && cd ${SERVER_REPO} && tar -xf - && ${POST_SYNC_CHECK#*&& cd ${SERVER_REPO} && }"
fi
