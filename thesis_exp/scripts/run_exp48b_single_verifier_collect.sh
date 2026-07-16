#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
OUT_DIR="${OUT_DIR:-thesis_exp/exp48_eduq_tail/outputs/exp48b_metric_rubric_local_edit_pilot}"
python -m thesis_exp.exp48_eduq_tail.audit_exp48a_style_shortcuts \
  --families "${OUT_DIR}/private/generated_families/exp48b_constructed_families.jsonl" \
  --out-dir "${OUT_DIR}" --labels 2,3,4 --output-prefix exp48b
python -m thesis_exp.exp48_eduq_tail.analyze_exp48b_single_verifier_pilot --out-dir "${OUT_DIR}" "$@"
