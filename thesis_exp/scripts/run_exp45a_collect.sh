#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${EDUBENCH_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"
PYTHON="${PYTHON:-/home/jpang/miniconda3/envs/llama_factory/bin/python}"
[[ -x "${PYTHON}" ]] || PYTHON="$(command -v python3)"
"${PYTHON}" -m thesis_exp.exp45_dopr_head.collect_exp45a_seed42 --require-complete
"${PYTHON}" -m thesis_exp.exp45_dopr_head.bootstrap_exp45a_question_key --replicates 5000
"${PYTHON}" -m thesis_exp.exp45_dopr_head.analyze_exp45a_decision
