#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
EXP27LR1_DIR="${EXP27LR1_DIR:-thesis_exp/exp17_low_score_evidence/outputs/exp27lr1_balanced_crossfit_review_seed42}"
OUT_DIR="${OUT_DIR:-thesis_exp/exp17_low_score_evidence/outputs/exp27m_model_review_audit_policy_seed42}"
REVIEWER_A="${REVIEWER_A:-${EXP27LR1_DIR}/private/exp27lr1_reviewer_a_normalized_107.jsonl}"
REVIEWER_B="${REVIEWER_B:-${EXP27LR1_DIR}/private/exp27lr1_reviewer_b_normalized_107.jsonl}"
ADJUDICATION="${ADJUDICATION:-${EXP27LR1_DIR}/private/exp27lr1_codex_model_adjudication_all38.jsonl}"

for required in "${REVIEWER_A}" "${REVIEWER_B}" "${ADJUDICATION}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing private model-review input: ${required}" >&2
    echo "Exp27M requires both normalized 107-row reviews and the exact 38-row model adjudication." >&2
    exit 2
  fi
done

"${PYTHON_BIN}" -m thesis_exp.exp17_low_score_evidence.analyze_exp27m_model_review_audit_policy \
  --exp27lr1-dir "${EXP27LR1_DIR}" \
  --out-dir "${OUT_DIR}" \
  --reviewer-a "${REVIEWER_A}" \
  --reviewer-b "${REVIEWER_B}" \
  --adjudication "${ADJUDICATION}" \
  --bootstrap-resamples 2000

"${PYTHON_BIN}" -m thesis_exp.exp17_low_score_evidence.validate_exp27m_model_review_audit_policy \
  --out-dir "${OUT_DIR}"

echo "Exp27M model-review audit policy acceptance complete: ${OUT_DIR}"
