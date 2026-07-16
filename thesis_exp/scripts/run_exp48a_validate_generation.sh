#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
FAMILIES="${FAMILIES:-thesis_exp/exp48_eduq_tail/outputs/exp48a_qualification_pilot/private/generated_families/exp48a_generated_families.jsonl}"
TRAIN="${TRAIN:-thesis_exp/data/splits/paper_like_triple_seed42/train.jsonl}"
OUT_DIR="${OUT_DIR:-thesis_exp/exp48_eduq_tail/outputs/exp48a_qualification_pilot}"
python -m thesis_exp.exp48_eduq_tail.validate_exp48a_generated_families --families "${FAMILIES}" --train "${TRAIN}" --out-dir "${OUT_DIR}"
python -m thesis_exp.exp48_eduq_tail.audit_exp48a_question_novelty --families "${FAMILIES}" --train "${TRAIN}" --out-dir "${OUT_DIR}"
python -m thesis_exp.exp48_eduq_tail.audit_exp48a_style_shortcuts --families "${FAMILIES}" --out-dir "${OUT_DIR}"
