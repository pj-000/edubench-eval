#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

declare -A SCHEDULES
SCHEDULES[4]="62:direct_residual_blocked 62:routed_hmsa 62:orthogonal_only 62:parallel_only 65:direct_residual_blocked 65:routed_hmsa 65:orthogonal_only"
SCHEDULES[6]="63:direct_residual_blocked 63:routed_hmsa 63:orthogonal_only 63:parallel_only 65:parallel_only 66:direct_residual_blocked 66:routed_hmsa"
SCHEDULES[7]="64:direct_residual_blocked 64:routed_hmsa 64:orthogonal_only 64:parallel_only 66:orthogonal_only 66:parallel_only"

for gpu in 4 6 7; do
  session="exp62_gpu${gpu}"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "Refusing to replace existing tmux session ${session}" >&2
    exit 3
  fi
  tmux new-session -d -s "${session}" \
    "cd '${REPO_ROOT}' && GPU_ID='${gpu}' SCHEDULE='${SCHEDULES[${gpu}]}' bash thesis_exp/scripts/run_exp62_queue.sh"
  echo "launched ${session}: ${SCHEDULES[${gpu}]}"
done
