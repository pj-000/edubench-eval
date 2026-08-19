#!/usr/bin/env bash
# Copy only the new Exp57 source/protocol to the existing training repository.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SERVER_HOST="${SERVER_HOST:-jpang@39.103.98.135}"
SERVER_PORT="${SERVER_PORT:-23722}"
SERVER_REPO="${SERVER_REPO:-/home/jpang/edubench-eval-exp2}"
SSH_ARGS=(-p "${SERVER_PORT}" -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=2)
RSYNC_SSH="ssh -p ${SERVER_PORT} -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=2"

FILES=(
  thesis_exp/exp57_cbrd
  thesis_exp/configs/exp57_cbrd/stage0_protocol.json
  thesis_exp/tests/test_exp57_cbrd_stage0.py
  thesis_exp/scripts/run_exp57_cbrd_stage0.sh
  thesis_exp/scripts/sync_exp57_cbrd_to_server.sh
)

# Read-only preflight: do not overwrite a pre-existing Exp57 directory.
ssh "${SSH_ARGS[@]}" "${SERVER_HOST}" \
  "test -d '${SERVER_REPO}' && if test -e '${SERVER_REPO}/thesis_exp/exp57_cbrd'; then echo 'Remote Exp57 already exists; inspect it before syncing.' >&2; exit 3; fi"

rsync -azR --exclude='__pycache__/' --exclude='*.pyc' --exclude='legacy_source/' \
  -e "${RSYNC_SSH}" "${FILES[@]}" "${SERVER_HOST}:${SERVER_REPO%/}/"

ssh "${SSH_ARGS[@]}" "${SERVER_HOST}" \
  "cd '${SERVER_REPO}' && chmod +x thesis_exp/scripts/run_exp57_cbrd_stage0.sh thesis_exp/scripts/sync_exp57_cbrd_to_server.sh && /home/jpang/miniconda3/envs/llama_factory/bin/python -m py_compile thesis_exp/exp57_cbrd/*.py thesis_exp/tests/test_exp57_cbrd_stage0.py && git status --short thesis_exp/exp57_cbrd thesis_exp/configs/exp57_cbrd thesis_exp/tests/test_exp57_cbrd_stage0.py thesis_exp/scripts/run_exp57_cbrd_stage0.sh thesis_exp/scripts/sync_exp57_cbrd_to_server.sh"
