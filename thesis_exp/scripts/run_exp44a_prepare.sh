#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${EDUBENCH_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-/home/jpang/miniconda3/envs/llama_factory/bin/python}"
MODEL="${RUBIMOR_MODEL_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
[[ -d "${MODEL}" ]] || { echo "Missing local model: ${MODEL}" >&2; exit 2; }

"${PYTHON}" -m thesis_exp.exp44_taco_score.resolve_exp44a_exp43_inputs \
  --model-name-or-path "${MODEL}"
"${PYTHON}" -m thesis_exp.exp44_taco_score.prepare_exp44a_triplets

