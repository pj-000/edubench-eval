#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SERVER_HOST="${SERVER_HOST:-jpang@39.103.98.135}"
SERVER_PORT="${SERVER_PORT:-23722}"
SERVER_REPO="${SERVER_REPO:-~/edubench-eval-exp2}"
DATA_ROOT="thesis_exp/exp17_low_score_evidence/outputs/exp28e_paper_ce_training_variants_seed42/private/datasets"
SSH_ARGS=(
  -p "${SERVER_PORT}"
  -o BatchMode=yes
  -o ConnectTimeout=10
  -o ServerAliveInterval=5
  -o ServerAliveCountMax=2
)
RSYNC_SSH="ssh -p ${SERVER_PORT} -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=2"

[[ -d "${DATA_ROOT}" ]] || { echo "Missing Exp28E datasets: ${DATA_ROOT}" >&2; exit 2; }
for variant in \
  b0_original_human \
  b1_primary_teacher_all \
  b2_selective_dual_teacher \
  b3_filter_unresolved \
  b4_random_transition_control; do
  [[ -f "${DATA_ROOT}/${variant}/train.jsonl" ]] || { echo "Missing ${variant}/train.jsonl" >&2; exit 2; }
  [[ -f "${DATA_ROOT}/${variant}/dev.jsonl" ]] || { echo "Missing ${variant}/dev.jsonl" >&2; exit 2; }
  [[ ! -f "${DATA_ROOT}/${variant}/test.jsonl" ]] || {
    echo "Refusing pre-lock sync because ${variant}/test.jsonl exists" >&2
    exit 2
  }
done

FILES=(
  thesis_exp/src/edujudge/exp02
  thesis_exp/src/edujudge/exp27p/common.py
  thesis_exp/exp17_low_score_evidence/prepare_exp28e_ce_training_variants.py
  thesis_exp/exp17_low_score_evidence/collect_exp28e_multiseed_dev_results.py
  thesis_exp/exp17_low_score_evidence/bootstrap_exp28f_dev_differences.py
  thesis_exp/exp17_low_score_evidence/lock_exp28f_final_test.py
  thesis_exp/exp17_low_score_evidence/prepare_exp28g_one_shot_test.py
  thesis_exp/exp17_low_score_evidence/collect_exp28g_final_test_results.py
  thesis_exp/exp17_low_score_evidence/bootstrap_exp28h_final_test.py
  thesis_exp/exp17_low_score_evidence/plot_exp28_paper_results.py
  thesis_exp/scripts/run_exp28e_reranker_multiseed_dev.sh
  thesis_exp/scripts/run_exp28g_one_shot_final_test.sh
  thesis_exp/scripts/sync_exp28_paper_experiment_to_server.sh
  "${DATA_ROOT}"
  thesis_exp/exp17_low_score_evidence/outputs/exp28e_paper_ce_training_variants_seed42/tables
  thesis_exp/exp17_low_score_evidence/outputs/exp28e_paper_ce_training_variants_seed42/reports
  thesis_exp/exp17_low_score_evidence/outputs/exp28e_paper_ce_training_variants_seed42/decision
)

echo "Syncing Exp28 paper experiment to ${SERVER_HOST}:${SERVER_REPO}"
rsync -azR \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='test.jsonl' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='*.bin' \
  --exclude='*.safetensors' \
  --exclude='*.npy' \
  --exclude='*.npz' \
  --exclude='*.log' \
  -e "${RSYNC_SSH}" \
  "${FILES[@]}" \
  "${SERVER_HOST}:${SERVER_REPO%/}/"

ssh "${SSH_ARGS[@]}" "${SERVER_HOST}" \
  "cd ${SERVER_REPO} && chmod +x thesis_exp/scripts/run_exp28e_reranker_multiseed_dev.sh thesis_exp/scripts/run_exp28g_one_shot_final_test.sh thesis_exp/scripts/sync_exp28_paper_experiment_to_server.sh && python -m py_compile thesis_exp/exp17_low_score_evidence/prepare_exp28e_ce_training_variants.py thesis_exp/exp17_low_score_evidence/collect_exp28e_multiseed_dev_results.py thesis_exp/exp17_low_score_evidence/bootstrap_exp28f_dev_differences.py thesis_exp/exp17_low_score_evidence/lock_exp28f_final_test.py thesis_exp/exp17_low_score_evidence/prepare_exp28g_one_shot_test.py thesis_exp/exp17_low_score_evidence/collect_exp28g_final_test_results.py thesis_exp/exp17_low_score_evidence/bootstrap_exp28h_final_test.py thesis_exp/exp17_low_score_evidence/plot_exp28_paper_results.py && bash -n thesis_exp/scripts/run_exp28e_reranker_multiseed_dev.sh thesis_exp/scripts/run_exp28g_one_shot_final_test.sh thesis_exp/scripts/sync_exp28_paper_experiment_to_server.sh"

echo "Exp28 paper experiment sync and remote preflight completed."
