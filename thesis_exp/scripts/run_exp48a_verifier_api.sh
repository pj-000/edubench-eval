#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PROVIDER="${PROVIDER:-deepseek}"
WORKERS="${WORKERS:-2}"
MAX_ROWS="${MAX_ROWS:-}"
VERIFIERS="${VERIFIERS:-a b}"
SESSION_SUFFIX="${SESSION_SUFFIX:-$(date +%Y%m%d_%H%M%S)}"

pids=()
for verifier_id in ${VERIFIERS}; do
  if [[ -n "${MAX_ROWS}" ]]; then
    python -m thesis_exp.exp48_eduq_tail.run_exp48a_verifier_api \
      --verifier-id "${verifier_id}" \
      --provider "${PROVIDER}" \
      --workers "${WORKERS}" \
      --session-id "exp48a_${PROVIDER}_verifier_${verifier_id}_${SESSION_SUFFIX}" \
      --max-rows "${MAX_ROWS}" &
  else
    python -m thesis_exp.exp48_eduq_tail.run_exp48a_verifier_api \
      --verifier-id "${verifier_id}" \
      --provider "${PROVIDER}" \
      --workers "${WORKERS}" \
      --session-id "exp48a_${PROVIDER}_verifier_${verifier_id}_${SESSION_SUFFIX}" &
  fi
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
