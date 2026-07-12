#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

OUT_DIR="${OUT_DIR:-thesis_exp/exp37_failure_evidence_qualification/outputs/exp37a_failure_evidence_qualification_seed42}"

python thesis_exp/exp37_failure_evidence_qualification/validate_exp37a_review_outputs.py \
  --out-dir "${OUT_DIR}"

python thesis_exp/exp37_failure_evidence_qualification/analyze_exp37a_failure_evidence_qualification.py \
  --out-dir "${OUT_DIR}"

echo "Exp37A analysis complete: ${OUT_DIR}"
