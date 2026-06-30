#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

CASES_CSV="${CASES_CSV:-thesis_exp/exp17_low_score_evidence/outputs/d1_hidden_failure_audit_seed42_dev/d1_exact_cases_for_manual_review.csv}"
SOURCE_ROOT="${SOURCE_ROOT:-.}"
OUT_DIR="${OUT_DIR:-thesis_exp/exp17_low_score_evidence/outputs/d1_hidden_failure_audit_seed42_dev/human_rationale_recovery}"

REQUIRED_FILES=(
  "${CASES_CSV}"
  "${SOURCE_ROOT}/5-grades/5_merge_human_metric_en.jsonl"
  "${SOURCE_ROOT}/5-grades/5_merge_human_metric_zh.jsonl"
  "${SOURCE_ROOT}/5-grades/5_human_1.jsonl"
  "${SOURCE_ROOT}/5-grades/5_human_2.jsonl"
  "${SOURCE_ROOT}/5-grades/5_human_3.jsonl"
)

for path in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 2
  fi
done

python thesis_exp/exp17_low_score_evidence/diagnostics/recover_human_rationales.py \
  --cases-csv "${CASES_CSV}" \
  --source-root "${SOURCE_ROOT}" \
  --out-dir "${OUT_DIR}"

echo "Exp17-D1 human rationale recovery outputs written to: ${OUT_DIR}"
