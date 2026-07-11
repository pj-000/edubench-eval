#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f ".env.exp28.local" ]]; then
  set -a
  source ".env.exp28.local"
  set +a
fi

RUN_API="${RUN_API:-0}"
DECISION="thesis_exp/exp17_low_score_evidence/outputs/exp28c_teacher_protocol_evaluation_seed42/decision/exp28c_protocol_development_protocol_decision.json"
[[ -f "${DECISION}" ]] || { echo "Missing development decision: ${DECISION}" >&2; exit 2; }
PROTOCOL="$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("selected_protocol") or "")' "${DECISION}")"
[[ -n "${PROTOCOL}" ]] || { echo "Development has not selected a protocol" >&2; exit 2; }

if [[ "${RUN_API}" == "1" ]]; then
  [[ -n "${QWEN_API_KEY:-}" ]] || { echo "Missing QWEN_API_KEY" >&2; exit 2; }
  [[ -n "${DEEPSEEK_API_KEY:-}" ]] || { echo "Missing DEEPSEEK_API_KEY" >&2; exit 2; }
fi

run_provider() {
  local provider="$1"
  if [[ "${RUN_API}" == "1" ]]; then
    python thesis_exp/exp17_low_score_evidence/run_exp28b_teacher_protocol_api.py \
      --provider "${provider}" --protocol "${PROTOCOL}" --subset sealed_qualification --run-api --resume
  else
    python thesis_exp/exp17_low_score_evidence/run_exp28b_teacher_protocol_api.py \
      --provider "${provider}" --protocol "${PROTOCOL}" --subset sealed_qualification --max-rows 1
  fi
}

echo "Exp28 sealed qualification; protocol=${PROTOCOL}; RUN_API=${RUN_API}"
run_provider qwen & qwen_pid=$!
run_provider deepseek & deepseek_pid=$!
failed=0
wait "${qwen_pid}" || failed=1
wait "${deepseek_pid}" || failed=1
[[ "${failed}" == "0" ]] || exit 1

if [[ "${RUN_API}" == "1" ]]; then
  python thesis_exp/exp17_low_score_evidence/collect_exp28c_teacher_protocol_results.py \
    --subset sealed_qualification
else
  echo "Dry run completed; collector skipped."
fi
