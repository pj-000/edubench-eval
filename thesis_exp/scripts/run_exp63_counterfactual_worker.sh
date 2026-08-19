#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 SEED [SEED ...]" >&2
  exit 2
fi

MODEL_PATH=/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B
PYTHON_BIN=/home/jpang/miniconda3/envs/llama_factory/bin/python

for seed in "$@"; do
  "$PYTHON_BIN" -m thesis_exp.exp63_same_state_counterfactual.counterfactual \
    --model_name_or_path "$MODEL_PATH" \
    --seed "$seed"
done
