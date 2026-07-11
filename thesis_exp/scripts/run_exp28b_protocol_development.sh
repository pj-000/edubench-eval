#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_API="${RUN_API:-0}"
SUBSET="${SUBSET:-protocol_development}"
PROTOCOLS="${PROTOCOLS:-p0_holistic_zero_shot p1_rubric_first p2_rubric_verify_then_score}"
PROVIDERS="${PROVIDERS:-qwen deepseek}"
MAX_ROWS="${MAX_ROWS:-0}"

if [[ "${RUN_API}" == "1" ]]; then
  [[ -n "${QWEN_API_KEY:-}" ]] || { echo "Missing QWEN_API_KEY" >&2; exit 2; }
  [[ -n "${DEEPSEEK_API_KEY:-}" ]] || { echo "Missing DEEPSEEK_API_KEY" >&2; exit 2; }
fi

echo "Exp28 protocol development"
echo "RUN_API=${RUN_API}"
echo "SUBSET=${SUBSET}"
echo "PROTOCOLS=${PROTOCOLS}"
echo "PROVIDERS=${PROVIDERS}"
echo "Keys are read from environment variables and are never printed."

run_one() {
  local provider="$1"
  local protocol="$2"
  if [[ "${RUN_API}" == "1" ]]; then
    python thesis_exp/exp17_low_score_evidence/run_exp28b_teacher_protocol_api.py \
      --provider "${provider}" \
      --protocol "${protocol}" \
      --subset "${SUBSET}" \
      --max-rows "${MAX_ROWS}" \
      --run-api \
      --resume
  else
    python thesis_exp/exp17_low_score_evidence/run_exp28b_teacher_protocol_api.py \
      --provider "${provider}" \
      --protocol "${protocol}" \
      --subset "${SUBSET}" \
      --max-rows "${MAX_ROWS}"
  fi
}

run_provider_queue() {
  local provider="$1"
  for protocol in ${PROTOCOLS}; do
    run_one "${provider}" "${protocol}"
  done
}

pids=()
names=()
for provider in ${PROVIDERS}; do
  run_provider_queue "${provider}" &
  pids+=("$!")
  names+=("${provider}")
done

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "Failed: ${names[$index]}" >&2
    failed=1
  fi
done
exit "${failed}"
