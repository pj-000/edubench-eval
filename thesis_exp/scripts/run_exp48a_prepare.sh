#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
TRAIN="${TRAIN:-thesis_exp/data/splits/paper_like_triple_seed42/train.jsonl}"
OUT_DIR="${OUT_DIR:-thesis_exp/exp48_eduq_tail/outputs/exp48a_qualification_pilot}"
SEED="${SEED:-48}"
python -m thesis_exp.exp48_eduq_tail.prepare_exp48a_source_blueprints --train "${TRAIN}" --out-dir "${OUT_DIR}" --seed "${SEED}"
