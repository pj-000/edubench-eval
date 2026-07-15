#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${EDUBENCH_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"
PYTHON="${PYTHON:-/home/jpang/miniconda3/envs/llama_factory/bin/python}"
MODEL="${RUBIMOR_MODEL_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"

"${PYTHON}" -m thesis_exp.exp44_taco_score.collect_exp44a_seed42 --model-name-or-path "${MODEL}" --require-complete
"${PYTHON}" -m thesis_exp.exp44_taco_score.bootstrap_exp44a_question_key --replicates 5000
"${PYTHON}" -m thesis_exp.exp44_taco_score.analyze_exp44a_decision

