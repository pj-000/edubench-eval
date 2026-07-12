#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

OUT_DIR="${OUT_DIR:-thesis_exp/exp37_failure_evidence_qualification/outputs/exp37a_failure_evidence_qualification_seed42}"
SEED="${SEED:-42}"

python thesis_exp/exp37_failure_evidence_qualification/prepare_exp37a_failure_evidence_qualification.py \
  --train-jsonl thesis_exp/data/splits/paper_like_triple_seed42/train.jsonl \
  --out-dir "${OUT_DIR}" --seed "${SEED}"

python thesis_exp/exp37_failure_evidence_qualification/recover_exp37a_human_reasons.py \
  --train-jsonl thesis_exp/data/splits/paper_like_triple_seed42/train.jsonl \
  --out-dir "${OUT_DIR}"

python thesis_exp/exp37_failure_evidence_qualification/audit_exp36a_shuffled_control_effective_changes.py \
  --train-jsonl thesis_exp/data/splits/paper_like_triple_seed42/train.jsonl \
  --out-dir "${OUT_DIR}"

python thesis_exp/exp37_failure_evidence_qualification/validate_exp37a_review_outputs.py \
  --out-dir "${OUT_DIR}" --allow-missing-reviews

python thesis_exp/exp37_failure_evidence_qualification/analyze_exp37a_failure_evidence_qualification.py \
  --out-dir "${OUT_DIR}" --allow-missing-reviews

echo "Exp37A prepare complete: train-only packets and pending-review diagnostics are in ${OUT_DIR}"
