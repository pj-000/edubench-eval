#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUT_DIR:-thesis_exp/exp17_low_score_evidence/outputs/exp27lr1_balanced_crossfit_review_seed42}"

"${PYTHON_BIN}" -m thesis_exp.exp17_low_score_evidence.prepare_exp27lr1_balanced_group_crossfit --out-dir "${OUT_DIR}"
"${PYTHON_BIN}" -m thesis_exp.exp17_low_score_evidence.fit_exp27lr1_balanced_group_crossfit --out-dir "${OUT_DIR}"
"${PYTHON_BIN}" -m thesis_exp.exp17_low_score_evidence.analyze_exp27lr1_balanced_group_crossfit --out-dir "${OUT_DIR}" --bootstrap-resamples 2000

echo "Exp27L-R1 balanced CPU crossfit complete: ${OUT_DIR}"
