#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO_ROOT="${EDUBENCH_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"; cd "${REPO_ROOT}"
MODE=smoke VARIANTS="E0 E4 E5 E6 E6N" SEEDS=42 FOLDS=0 bash thesis_exp/scripts/run_exp43_matrix.sh
"${PYTHON:-/home/jpang/miniconda3/envs/llama_factory/bin/python}" -m thesis_exp.exp43_rubimor.analyze_exp43_stage_gates --stage smoke

