#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES=""
export PYTHONHASHSEED=42

python -m py_compile \
  thesis_exp/exp35_edudart_cal/prepare_exp35a_model_reviewed_qualification.py \
  thesis_exp/exp35_edudart_cal/validate_exp35a_preparation.py
python thesis_exp/exp35_edudart_cal/prepare_exp35a_model_reviewed_qualification.py "$@"
python thesis_exp/exp35_edudart_cal/validate_exp35a_preparation.py
