#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUT_DIR:-thesis_exp/exp17_low_score_evidence/outputs/exp27lr1_balanced_crossfit_review_seed42}"

if [[ ! -f "${OUT_DIR}/data/exp27lr1_oof_risk_predictions.csv" ]]; then
  echo "Missing Exp27L-R1 OOF risk predictions. Run run_exp27lr1_balanced_crossfit.sh first." >&2
  exit 2
fi

"${PYTHON_BIN}" -m thesis_exp.exp17_low_score_evidence.prepare_exp27lr1_blind_review_sets --out-dir "${OUT_DIR}"
"${PYTHON_BIN}" -m thesis_exp.exp17_low_score_evidence.validate_exp27lr1_balanced_group_crossfit --out-dir "${OUT_DIR}"

echo "Exp27L-R1 target-aware blind-review packets prepared: ${OUT_DIR}"
