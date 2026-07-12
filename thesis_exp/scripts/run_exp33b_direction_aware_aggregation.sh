#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES=""
export PYTHONHASHSEED=42

python -m py_compile \
  thesis_exp/exp33_direction_aware_aggregation/prepare_exp33b_direction_aware_aggregation.py \
  thesis_exp/exp33_direction_aware_aggregation/validate_exp33b_direction_aware_aggregation.py

bash -n thesis_exp/scripts/run_exp33b_direction_aware_aggregation.sh

python thesis_exp/exp33_direction_aware_aggregation/prepare_exp33b_direction_aware_aggregation.py "$@"
python thesis_exp/exp33_direction_aware_aggregation/validate_exp33b_direction_aware_aggregation.py --heavy "$@"
