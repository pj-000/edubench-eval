#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"

"${PYTHON}" thesis_exp/exp42_rubidist/prepare_exp42a_factorial_variants.py \
  --tokenizer-name-or-path "${MODEL_NAME_OR_PATH}"
"${PYTHON}" thesis_exp/exp42_rubidist/prepare_exp42a_groupcv_folds.py
"${PYTHON}" thesis_exp/exp42_rubidist/audit_exp41a_postfit_compiled_coverage.py \
  --tokenizer-name-or-path "${MODEL_NAME_OR_PATH}"

echo "Exp42A prepare PASS: four locked variants, exact Exp41 folds, no dev/test access."
