#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
GPU_LIST="${GPU_LIST:-0}"
GPU="${GPU_LIST%% *}"
for variant in v0h_human_empirical v4_hails; do
  CUDA_VISIBLE_DEVICES="${GPU}" python thesis_exp/exp38_hails_score/train_exp38a_groupcv.py \
    --variant "${variant}" \
    --fold 0 \
    --model-name-or-path "${MODEL_NAME_OR_PATH}" \
    --epochs 1 \
    --max-train-rows 32 \
    --max-eval-rows 16
done
python thesis_exp/exp38_hails_score/collect_exp38a_groupcv.py \
  --variants v0h_human_empirical v4_hails \
  --allow-incomplete
