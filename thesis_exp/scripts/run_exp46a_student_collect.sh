#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${EDUBENCH_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"
PYTHON="${PYTHON:-/home/jpang/miniconda3/envs/llama_factory/bin/python}"
"${PYTHON}" -m thesis_exp.exp46_hato_kd.collect_exp46_student --require-complete
"${PYTHON}" -m thesis_exp.exp46_hato_kd.bootstrap_exp46 --stage student --replicates 5000
"${PYTHON}" -m thesis_exp.exp46_hato_kd.analyze_exp46_student_gate
