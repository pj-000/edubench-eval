#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

OUT_DIR="${OUT_DIR:-thesis_exp/exp17_low_score_evidence/outputs/exp27a_teacher_audit_pilot_seed42}"
PACKETS="${PACKETS:-${OUT_DIR}/packets/exp27a_pilot_blind_packets.jsonl}"
PROVIDERS="${PROVIDERS:-qwen deepseek}"
STAGES="${STAGES:-blind}"
MAX_ROWS="${MAX_ROWS:-0}"
THINKING="${THINKING:-omit}"
SLEEP_SECONDS="${SLEEP_SECONDS:-0}"
TIMEOUT="${TIMEOUT:-120}"
RETRIES="${RETRIES:-2}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

cat <<CONFIG
Exp27A teacher-audit pilot API runner
OUT_DIR=${OUT_DIR}
PACKETS=${PACKETS}
PROVIDERS=${PROVIDERS}
STAGES=${STAGES}
MAX_ROWS=${MAX_ROWS}
THINKING=${THINKING}
SLEEP_SECONDS=${SLEEP_SECONDS}
TIMEOUT=${TIMEOUT}
RETRIES=${RETRIES}

API keys are read only from environment variables:
- QWEN_API_KEY for provider=qwen
- DEEPSEEK_API_KEY for provider=deepseek

This step calls teacher APIs only. It does not train and does not read dev/test labels.
CONFIG

python -m py_compile \
  thesis_exp/exp17_low_score_evidence/run_exp27a_teacher_audit_api.py \
  thesis_exp/exp17_low_score_evidence/validate_exp27a_teacher_audit.py

for provider in ${PROVIDERS}; do
  for stage in ${STAGES}; do
    echo "Starting Exp27A provider=${provider} stage=${stage}"
    python thesis_exp/exp17_low_score_evidence/run_exp27a_teacher_audit_api.py \
      --provider "${provider}" \
      --stage "${stage}" \
      --packets "${PACKETS}" \
      --out-dir "${OUT_DIR}" \
      --max-rows "${MAX_ROWS}" \
      --thinking "${THINKING}" \
      --sleep "${SLEEP_SECONDS}" \
      --timeout "${TIMEOUT}" \
      --retries "${RETRIES}"
  done
done

python thesis_exp/exp17_low_score_evidence/validate_exp27a_teacher_audit.py \
  --out-dir "${OUT_DIR}"

echo "Exp27A teacher-audit pilot API run completed."
