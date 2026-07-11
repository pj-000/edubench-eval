#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="thesis_exp/exp17_low_score_evidence/outputs/exp27r_final_test_campaign_seed42_44"
git fetch origin main >/dev/null
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git rev-parse origin/main)}"

"${PYTHON_BIN}" thesis_exp/exp17_low_score_evidence/prepare_exp27r_final_test_lock.py \
  --out-dir "${OUT_DIR}" \
  --expected-commit "${SOURCE_COMMIT}" \
  --crossed-resamples 5000
"${PYTHON_BIN}" thesis_exp/exp17_low_score_evidence/validate_exp27r_final_test_lock.py \
  --out-dir "${OUT_DIR}"

echo "Exp27R phase-1 lock PASS; test was not read."

