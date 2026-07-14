#!/usr/bin/env bash
set -euo pipefail
MODE=groupcv VARIANTS="E5" SEEDS=42 FOLDS="0 1 2 3 4" bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_exp43_matrix.sh"
PYTHON="${PYTHON:-/home/jpang/miniconda3/envs/llama_factory/bin/python}"; "${PYTHON}" -m thesis_exp.exp43_rubimor.collect_exp43_groupcv --variants E0 E1 E2 E3 E4 E5 --seeds 42 --require-complete
"${PYTHON}" -m thesis_exp.exp43_rubimor.analyze_exp43_stage_gates --stage stage4

