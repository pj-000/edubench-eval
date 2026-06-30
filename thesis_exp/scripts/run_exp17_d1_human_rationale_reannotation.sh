#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

BASE_DIR="${BASE_DIR:-thesis_exp/exp17_low_score_evidence/outputs/d1_hidden_failure_audit_seed42_dev}"
RECOVERY_DIR="${RECOVERY_DIR:-${BASE_DIR}/human_rationale_recovery}"
ANNOTATED_CSV="${ANNOTATED_CSV:-${BASE_DIR}/d1_hidden_failure_annotation_template_filled_human_rationale_recovered.csv}"
SUMMARY_DIR="${SUMMARY_DIR:-${BASE_DIR}/summary_human_rationale_recovered}"

./thesis_exp/scripts/run_exp17_d1_human_rationale_recovery.sh

python thesis_exp/exp17_low_score_evidence/diagnostics/reannotate_d1_with_human_rationales.py \
  --template-csv "${BASE_DIR}/d1_hidden_failure_annotation_template.csv" \
  --recovered-csv "${RECOVERY_DIR}/d1_human_rationale_recovered.csv" \
  --out-csv "${ANNOTATED_CSV}" \
  --report "${BASE_DIR}/d1_human_rationale_reannotation_report.md"

python thesis_exp/exp17_low_score_evidence/diagnostics/summarize_hidden_failure_audit.py \
  --annotated-csv "${ANNOTATED_CSV}" \
  --case-control-csv "${BASE_DIR}/d1_matched_case_control_review.csv" \
  --out-dir "${SUMMARY_DIR}" \
  --split dev

echo "Exp17-D1 human-rationale reannotation summary written to: ${SUMMARY_DIR}"
