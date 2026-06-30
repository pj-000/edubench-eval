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
  thesis_exp/__init__.py
  thesis_exp/src/__init__.py
  thesis_exp/src/edujudge/__init__.py
  thesis_exp/src/edujudge/exp16_boundary_linking
  thesis_exp/exp17_low_score_evidence/train_exp17_a1_evidence_head.py
  thesis_exp/exp17_low_score_evidence/collect_exp17_a1_results.py
  thesis_exp/exp17_low_score_evidence/diagnostics/build_train_hidden_failure_signals.py
  thesis_exp/exp17_low_score_evidence/outputs/exp17_a0_train_hidden_failure_signals_seed42/train_hidden_failure_candidates.csv
  thesis_exp/exp17_low_score_evidence/outputs/exp17_a0_train_hidden_failure_signals_seed42/train_clean_high_controls.csv
  thesis_exp/exp17_low_score_evidence/outputs/exp17_a0_train_hidden_failure_signals_seed42/train_hidden_failure_pairs.csv
  thesis_exp/exp17_low_score_evidence/outputs/d1_hidden_failure_audit_seed42_dev/d1_hidden_failure_annotation_template_filled_human_rationale_recovered.csv
  thesis_exp/exp17_low_score_evidence/outputs/d1_hidden_failure_audit_seed42_dev/d1_matched_case_control_review.csv
  thesis_exp/exp17_low_score_evidence/outputs/d1_hidden_failure_audit_seed42_dev/summary_human_rationale_recovered
  thesis_exp/scripts/run_exp17_a1_evidence_head_scout.sh
  thesis_exp/scripts/sync_exp17_a1_evidence_head_to_server.sh
)

cat <<CONFIG
Sync Exp17-A1 evidence-head scaffold to server
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
  --exclude='*.jsonl' \
  --exclude='logs/' \
  --exclude='runs/' \
  --exclude='checkpoint_best/' \
  --stats \
  -e "${RSYNC_SSH}" \
  "${FILES[@]}" \
  "${SERVER_HOST}:${SERVER_REPO%/}/"

ssh "${SSH_OPTS[@]}" "${SERVER_HOST}" \
  "bash -lc 'export LC_ALL=C LANG=C && source ~/miniconda3/bin/activate ${CONDA_ENV} && cd ${SERVER_REPO} && export PYTHONPATH=\"\$(pwd):\${PYTHONPATH:-}\" && chmod +x thesis_exp/scripts/run_exp17_a1_evidence_head_scout.sh thesis_exp/scripts/sync_exp17_a1_evidence_head_to_server.sh && python -m py_compile thesis_exp/exp17_low_score_evidence/train_exp17_a1_evidence_head.py thesis_exp/exp17_low_score_evidence/collect_exp17_a1_results.py thesis_exp/exp17_low_score_evidence/diagnostics/build_train_hidden_failure_signals.py && bash -n thesis_exp/scripts/run_exp17_a1_evidence_head_scout.sh && bash -n thesis_exp/scripts/sync_exp17_a1_evidence_head_to_server.sh && python thesis_exp/exp17_low_score_evidence/train_exp17_a1_evidence_head.py --help >/tmp/exp17_a1_train_help.txt && python thesis_exp/exp17_low_score_evidence/collect_exp17_a1_results.py --help >/tmp/exp17_a1_collect_help.txt && if [[ -f thesis_exp/outputs/exp16_boundary_linking/runs/qmr/seed_42/checkpoint_best/state_dict.pt ]]; then echo \"Exp16A qmr init checkpoint: FOUND\"; else echo \"WARNING: Exp16A qmr init checkpoint missing on server\"; fi && git status --short thesis_exp/exp17_low_score_evidence thesis_exp/scripts/run_exp17_a1_evidence_head_scout.sh thesis_exp/scripts/sync_exp17_a1_evidence_head_to_server.sh thesis_exp/src/edujudge/exp16_boundary_linking | sed -n \"1,200p\"'"
