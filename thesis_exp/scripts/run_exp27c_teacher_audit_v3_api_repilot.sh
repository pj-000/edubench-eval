#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

OUT_DIR="${OUT_DIR:-thesis_exp/exp17_low_score_evidence/outputs/exp27c_teacher_audit_v3_seed42}"
MAX_ROWS="${MAX_ROWS:-0}"
SLEEP_SECONDS="${SLEEP_SECONDS:-0}"
TIMEOUT="${TIMEOUT:-120}"
RETRIES="${RETRIES:-2}"
SCHEMA_REPAIR_RETRIES="${SCHEMA_REPAIR_RETRIES:-1}"
THINKING="${THINKING:-omit}"
PREPARE_FIRST="${PREPARE_FIRST:-1}"
PARALLEL_PROVIDERS="${PARALLEL_PROVIDERS:-1}"

cat <<CONFIG
Exp27C teacher-audit v3 real API re-pilot
OUT_DIR=${OUT_DIR}
MAX_ROWS=${MAX_ROWS}
SLEEP_SECONDS=${SLEEP_SECONDS}
TIMEOUT=${TIMEOUT}
RETRIES=${RETRIES}
SCHEMA_REPAIR_RETRIES=${SCHEMA_REPAIR_RETRIES}
THINKING=${THINKING}
PREPARE_FIRST=${PREPARE_FIRST}
PARALLEL_PROVIDERS=${PARALLEL_PROVIDERS}

This step calls teacher APIs. It does not train, does not use GPU, and does not read test labels.
API keys are read only from QWEN_API_KEY and DEEPSEEK_API_KEY environment variables.
CONFIG

if [[ -z "${QWEN_API_KEY:-}" ]]; then
  echo "Missing QWEN_API_KEY" >&2
  exit 2
fi
if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "Missing DEEPSEEK_API_KEY" >&2
  exit 2
fi

if [[ "${PREPARE_FIRST}" == "1" ]]; then
  ./thesis_exp/scripts/run_exp27c_teacher_audit_v3_prepare.sh
fi

common_args=(
  --out-dir "${OUT_DIR}"
  --max-rows "${MAX_ROWS}"
  --run-api
  --resume
  --temperature 0
  --sleep-seconds "${SLEEP_SECONDS}"
  --timeout "${TIMEOUT}"
  --retries "${RETRIES}"
  --schema-repair-retries "${SCHEMA_REPAIR_RETRIES}"
  --thinking "${THINKING}"
)

run_provider_stage() {
  local provider="$1"
  local stage="$2"
  python thesis_exp/exp17_low_score_evidence/run_exp27c_teacher_audit_api.py \
    --provider "${provider}" \
    --stage "${stage}" \
    "${common_args[@]}"
}

run_stage_for_both_providers() {
  local stage="$1"
  if [[ "${PARALLEL_PROVIDERS}" == "1" ]]; then
    echo "Starting Exp27C ${stage} stage for qwen and deepseek in parallel."
    run_provider_stage qwen "${stage}" &
    local qwen_pid=$!
    run_provider_stage deepseek "${stage}" &
    local deepseek_pid=$!
    wait "${qwen_pid}"
    wait "${deepseek_pid}"
  else
    echo "Starting Exp27C ${stage} stage for qwen and deepseek sequentially."
    run_provider_stage qwen "${stage}"
    run_provider_stage deepseek "${stage}"
  fi
}

run_stage_for_both_providers blind
run_stage_for_both_providers audit

python thesis_exp/exp17_low_score_evidence/collect_exp27c_teacher_audit_results.py \
  --out-dir "${OUT_DIR}"

python thesis_exp/exp17_low_score_evidence/validate_exp27c_teacher_audit.py \
  --out-dir "${OUT_DIR}" \
  --with-annotations

echo "Exp27C teacher-audit v3 API re-pilot completed."
