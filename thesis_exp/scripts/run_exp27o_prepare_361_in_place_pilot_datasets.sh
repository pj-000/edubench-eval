#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUT_DIR:-thesis_exp/exp17_low_score_evidence/outputs/exp27o_361_in_place_pilot_seed42}"

"${PYTHON_BIN}" -m thesis_exp.exp17_low_score_evidence.prepare_exp27o_361_in_place_pilot_datasets \
  --out-dir "${OUT_DIR}"
"${PYTHON_BIN}" -m thesis_exp.exp17_low_score_evidence.validate_exp27o_361_in_place_pilot_datasets \
  --out-dir "${OUT_DIR}" \
  --require-private

echo "Exp27O 361-row in-place pilot datasets ready: ${OUT_DIR}"
