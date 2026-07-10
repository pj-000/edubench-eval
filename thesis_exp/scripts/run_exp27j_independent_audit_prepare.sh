#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

OUT_DIR="${OUT_DIR:-thesis_exp/exp17_low_score_evidence/outputs/exp27j_independent_audit_seed42}"
SEED="${SEED:-42}"
PYTHON="${PYTHON:-python3}"

"${PYTHON}" thesis_exp/exp17_low_score_evidence/prepare_exp27j_independent_audit_benchmark.py \
  --out-dir "${OUT_DIR}" \
  --seed "${SEED}"

"${PYTHON}" thesis_exp/exp17_low_score_evidence/audit_exp27i_calibration_implementation.py \
  --out-dir "${OUT_DIR}"

"${PYTHON}" thesis_exp/exp17_low_score_evidence/validate_exp27j_independent_audit.py \
  --out-dir "${OUT_DIR}"

"${PYTHON}" thesis_exp/exp17_low_score_evidence/analyze_exp27j_independent_audit.py \
  --out-dir "${OUT_DIR}" \
  --allow-missing-reviews

echo "Exp27J preparation complete: ${OUT_DIR}"
