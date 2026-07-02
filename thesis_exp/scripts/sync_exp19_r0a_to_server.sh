#!/usr/bin/env bash
set -euo pipefail

SERVER_HOST="${SERVER_HOST:-jpang@39.103.98.135}"
SERVER_PORT="${SERVER_PORT:-23722}"
SERVER_REPO="${SERVER_REPO:-~/edubench-eval-exp2}"
CONDA_ENV="${CONDA_ENV:-vllm_qwen_env}"
RSYNC_SSH="ssh -p ${SERVER_PORT} -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=2"
SSH_OPTS=(
  -p "${SERVER_PORT}"
  -o BatchMode=yes
  -o ConnectTimeout=10
  -o ServerAliveInterval=5
  -o ServerAliveCountMax=2
)
FILES=(
  thesis_exp/exp17_low_score_evidence/run_exp19_r0a_qwen4b_direct_baseline.py
  thesis_exp/exp17_low_score_evidence/README_exp19_r0a_qwen4b_direct_baseline.md
  thesis_exp/scripts/run_exp19_r0a_qwen4b_direct_baseline.sh
  thesis_exp/scripts/sync_exp19_r0a_to_server.sh
)

cat <<CONFIG
Sync Exp19-R0A Qwen3-4B direct baseline to server
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
  --stats \
  -e "${RSYNC_SSH}" \
  "${FILES[@]}" \
  "${SERVER_HOST}:${SERVER_REPO%/}/"

ssh "${SSH_OPTS[@]}" "${SERVER_HOST}" \
  "bash -lc 'export LC_ALL=C LANG=C && source ~/miniconda3/bin/activate ${CONDA_ENV} && cd ${SERVER_REPO} && export PYTHONPATH=\"\$(pwd):\${PYTHONPATH:-}\" && chmod +x thesis_exp/scripts/run_exp19_r0a_qwen4b_direct_baseline.sh thesis_exp/scripts/sync_exp19_r0a_to_server.sh && python -m py_compile thesis_exp/exp17_low_score_evidence/run_exp19_r0a_qwen4b_direct_baseline.py && python thesis_exp/exp17_low_score_evidence/run_exp19_r0a_qwen4b_direct_baseline.py --help >/tmp/exp19_r0a_help.txt && python thesis_exp/exp17_low_score_evidence/run_exp19_r0a_qwen4b_direct_baseline.py --dry_run --max_examples 5 --out_dir /tmp/exp19_r0a_dryrun --overwrite && bash -n thesis_exp/scripts/run_exp19_r0a_qwen4b_direct_baseline.sh && bash -n thesis_exp/scripts/sync_exp19_r0a_to_server.sh && if [[ -d /home/jpang/models/modelscope/Qwen/Qwen3-4B ]]; then echo \"Qwen3-4B model path: FOUND\"; else echo \"WARNING: Qwen3-4B model path missing\"; fi && nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | sed -n \"1,4p\" && git status --short thesis_exp/exp17_low_score_evidence/run_exp19_r0a_qwen4b_direct_baseline.py thesis_exp/exp17_low_score_evidence/README_exp19_r0a_qwen4b_direct_baseline.md thesis_exp/scripts/run_exp19_r0a_qwen4b_direct_baseline.sh thesis_exp/scripts/sync_exp19_r0a_to_server.sh | sed -n \"1,120p\"'"
