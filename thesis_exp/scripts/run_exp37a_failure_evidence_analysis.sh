#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

OUT_DIR="${OUT_DIR:-thesis_exp/exp37_failure_evidence_qualification/outputs/exp37a_r1_model_reviewed_qualification_seed42}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir|--out-dir) OUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

python thesis_exp/exp37_failure_evidence_qualification/prepare_exp37a_adjudication_packets.py \
  --out-dir "${OUT_DIR}"

python thesis_exp/exp37_failure_evidence_qualification/validate_exp37a_review_outputs.py \
  --out-dir "${OUT_DIR}"

python thesis_exp/exp37_failure_evidence_qualification/analyze_exp37a_failure_evidence_qualification.py \
  --out-dir "${OUT_DIR}"

echo "Exp37A-R1 analysis complete: ${OUT_DIR}"
