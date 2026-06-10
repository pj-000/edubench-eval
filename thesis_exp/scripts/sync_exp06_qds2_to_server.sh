#!/usr/bin/env bash
set -euo pipefail

SERVER_HOST="${SERVER_HOST:-jpang@39.103.98.135}"
SERVER_PORT="${SERVER_PORT:-23722}"
SERVER_REPO="${SERVER_REPO:-~/edubench-eval-exp2}"
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
  thesis_exp/scripts/run_exp06_qds2_smoke.sh
  thesis_exp/scripts/run_exp06_qds2_train.sh
  thesis_exp/scripts/sync_exp06_qds2_to_server.sh
  thesis_exp/src/edujudge/exp06_training/__init__.py
  thesis_exp/src/edujudge/exp06_training/collect_qds1_results.py
  thesis_exp/src/edujudge/exp06_training/collect_qds2_results.py
  thesis_exp/src/edujudge/exp06_training/train_qds1_ordinal.py
  thesis_exp/src/edujudge/exp06_training/train_qds2_l1.py
  thesis_exp/outputs/exp06_synthetic_low_score/training_datasets/QD-S2_human_plus_synthetic_L1
)

cat <<CONFIG
Sync Exp6 QD-S2 scripts and dataset to server
SERVER_HOST=${SERVER_HOST}
SERVER_PORT=${SERVER_PORT}
SERVER_REPO=${SERVER_REPO}
SYNC_MODE=${SYNC_MODE}
CONFIG

POST_SYNC_CHECK="export LC_ALL=C LANG=C && cd ${SERVER_REPO} && chmod +x thesis_exp/scripts/run_exp06_qds2_smoke.sh thesis_exp/scripts/run_exp06_qds2_train.sh thesis_exp/scripts/sync_exp06_qds2_to_server.sh && python -m py_compile thesis_exp/src/edujudge/exp06_training/*.py && find thesis_exp/src/edujudge/exp06_training -name '__pycache__' -type d -prune -exec rm -rf {} + && wc -l thesis_exp/outputs/exp06_synthetic_low_score/training_datasets/QD-S2_human_plus_synthetic_L1/train.jsonl thesis_exp/outputs/exp06_synthetic_low_score/training_datasets/QD-S2_human_plus_synthetic_L1/dev.jsonl thesis_exp/outputs/exp06_synthetic_low_score/training_datasets/QD-S2_human_plus_synthetic_L1/test.jsonl && FORMAL_RUN=0 REQUIRE_CUDA=0 python -m thesis_exp.src.edujudge.exp06_training.train_qds2_l1 --preflight_only --data_dir thesis_exp/outputs/exp06_synthetic_low_score/training_datasets/QD-S2_human_plus_synthetic_L1 --output_dir /tmp/exp06_qds2_preflight --checkpoint_output_dir /tmp/exp06_qds2_preflight_ckpt && cat /tmp/exp06_qds2_preflight/tables/class_weights.csv"

if [[ "${SYNC_MODE}" == "rsync" ]]; then
  rsync -azR --stats -e "${RSYNC_SSH}" "${FILES[@]}" "${SERVER_HOST}:${SERVER_REPO%/}/"
  ssh "${SSH_OPTS[@]}" "${SERVER_HOST}" "${POST_SYNC_CHECK}"
else
  COPYFILE_DISABLE=1 tar --no-xattrs -cf - "${FILES[@]}" | ssh "${SSH_OPTS[@]}" "${SERVER_HOST}" "export LC_ALL=C LANG=C && cd ${SERVER_REPO} && tar -xf - && ${POST_SYNC_CHECK#*&& }"
fi
