#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUT_DIR:-thesis_exp/exp17_low_score_evidence/outputs/exp27l_group_crossfit_calibration_seed42}"

if [[ ! -f "${OUT_DIR}/data/exp27l_question_key_fold_assignment.csv" ]]; then
  echo "Missing Exp27L fold assignment. Run ./thesis_exp/scripts/run_exp27l_crossfit_prepare.sh first." >&2
  exit 2
fi

"${PYTHON_BIN}" -m thesis_exp.exp17_low_score_evidence.fit_exp27l_group_crossfit_calibration \
  --out-dir "${OUT_DIR}"
"${PYTHON_BIN}" -m thesis_exp.exp17_low_score_evidence.analyze_exp27l_group_crossfit_calibration \
  --out-dir "${OUT_DIR}" \
  --bootstrap-resamples 2000
"${PYTHON_BIN}" -m thesis_exp.exp17_low_score_evidence.prepare_exp27l_external_review_lockbox \
  --out-dir "${OUT_DIR}" \
  --include-targeted-ambiguity
"${PYTHON_BIN}" -m thesis_exp.exp17_low_score_evidence.validate_exp27l_group_crossfit_calibration \
  --out-dir "${OUT_DIR}"

echo "Exp27L OOF analysis complete: ${OUT_DIR}"
