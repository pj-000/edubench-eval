#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SERVER_HOST="${SERVER_HOST:-jpang@39.103.98.135}"
SERVER_PORT="${SERVER_PORT:-23722}"
SERVER_REPO="${SERVER_REPO:-~/edubench-eval-exp2}"
RSYNC_SSH="ssh -p ${SERVER_PORT} -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=2"
FILES=(
  thesis_exp/exp17_low_score_evidence/exp27lr1_common.py
  thesis_exp/exp17_low_score_evidence/prepare_exp27o_361_in_place_pilot_datasets.py
  thesis_exp/exp17_low_score_evidence/validate_exp27o_361_in_place_pilot_datasets.py
  thesis_exp/exp17_low_score_evidence/outputs/exp27o_361_in_place_pilot_seed42/configs
  thesis_exp/exp17_low_score_evidence/outputs/exp27o_361_in_place_pilot_seed42/decision
  thesis_exp/exp17_low_score_evidence/outputs/exp27o_361_in_place_pilot_seed42/reports
  thesis_exp/exp17_low_score_evidence/outputs/exp27o_361_in_place_pilot_seed42/tables
  thesis_exp/exp17_low_score_evidence/outputs/exp27o_361_in_place_pilot_seed42/private/data
  thesis_exp/exp17_low_score_evidence/outputs/exp27p_soft_target_reranker_seed42/configs
  thesis_exp/src/edujudge/exp27p
  thesis_exp/scripts/run_exp27o_prepare_361_in_place_pilot_datasets.sh
  thesis_exp/scripts/run_exp27p_soft_target_smoke.sh
  thesis_exp/scripts/run_exp27p_seed42_scout.sh
  thesis_exp/scripts/run_exp27p_multiseed_after_scout.sh
  thesis_exp/scripts/sync_exp27p_to_server.sh
)

rsync -azR \
  --exclude='__pycache__/' --exclude='*.pyc' --exclude='*.pt' --exclude='*.bin' \
  --exclude='*.safetensors' --exclude='*.log' --exclude='predictions_private/' \
  -e "${RSYNC_SSH}" "${FILES[@]}" "${SERVER_HOST}:${SERVER_REPO%/}/"

ssh -p "${SERVER_PORT}" -o BatchMode=yes -o ConnectTimeout=10 "${SERVER_HOST}" \
  "cd ${SERVER_REPO} && chmod +x thesis_exp/scripts/run_exp27p_*.sh thesis_exp/scripts/sync_exp27p_to_server.sh && python -m compileall -q thesis_exp/src/edujudge/exp27p thesis_exp/exp17_low_score_evidence/prepare_exp27o_361_in_place_pilot_datasets.py thesis_exp/exp17_low_score_evidence/validate_exp27o_361_in_place_pilot_datasets.py && python -m thesis_exp.src.edujudge.exp27p.validate_exp27p_training_setup --exp27o-dir thesis_exp/exp17_low_score_evidence/outputs/exp27o_361_in_place_pilot_seed42"

echo "Exp27P code and private Exp27O train variants synced to ${SERVER_HOST}:${SERVER_REPO}"
