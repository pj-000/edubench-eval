#!/usr/bin/env bash
set -euo pipefail
MODE=groupcv VARIANTS="E0 E1 E2 E3 E4 E5 E6 E6N" SEEDS="42 43 44" FOLDS="0 1 2 3 4" bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_exp43_matrix.sh"
PYTHON="${PYTHON:-/home/jpang/miniconda3/envs/llama_factory/bin/python}"; "${PYTHON}" -m thesis_exp.exp43_rubimor.collect_exp43_groupcv --require-complete --require-fingerprints
"${PYTHON}" -m thesis_exp.exp43_rubimor.bootstrap_exp43_groupcv
"${PYTHON}" -m thesis_exp.exp43_rubimor.analyze_exp43_stage_gates --stage stage6
