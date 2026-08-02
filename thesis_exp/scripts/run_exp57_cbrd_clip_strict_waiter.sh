#!/usr/bin/env bash
# Start the strict cuBLAS replay after the first diagnostic campaign completes.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

GPU_ID="${GPU_ID:?Set an available RTX 3090 index}"
ASSIGNMENT="${ASSIGNMENT:?Set gpu6 or gpu7}"
DIAGNOSTIC_DIR="thesis_exp/outputs/exp57_cbrd/audit/clip_gradient"

while [[ "$(find "${DIAGNOSTIC_DIR}" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)" -lt 9 ]]; do
  sleep 15
done

GPU_ID="${GPU_ID}" \
WAIT_SESSION="exp57_clip_diagnostic_already_complete" \
ASSIGNMENT="${ASSIGNMENT}" \
AUDIT_DIR="thesis_exp/outputs/exp57_cbrd/audit/clip_gradient_strict" \
LOG_PREFIX="clip_strict" \
CUBLAS_WORKSPACE_CONFIG=":4096:8" \
bash thesis_exp/scripts/run_exp57_cbrd_clip_audit_worker.sh
