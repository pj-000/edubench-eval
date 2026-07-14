#!/usr/bin/env bash
set -euo pipefail
MODE=headline VARIANTS="E0 E3 E5 E6 E6N" SEEDS="42 43 44" FOLDS=0 bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_exp43_matrix.sh"
PYTHON="${PYTHON:-/home/jpang/miniconda3/envs/llama_factory/bin/python}"; "${PYTHON}" -m thesis_exp.exp43_rubimor.collect_exp43_headline_dev

