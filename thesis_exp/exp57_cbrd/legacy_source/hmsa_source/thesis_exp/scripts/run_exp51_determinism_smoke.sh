#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
GPU_ID="${GPU_ID:-0}"
MODEL_PATH="${MODEL_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
ROOT=thesis_exp/outputs/exp51_hmsa/audit/determinism
for RUN_NAME in run_a run_b; do
  CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH=. python -m thesis_exp.exp51_hmsa.train \
    --model_name_or_path "${MODEL_PATH}" \
    --output_dir "${ROOT}/${RUN_NAME}" \
    --checkpoint_output_dir "thesis_exp/artifacts/exp51_hmsa/determinism/${RUN_NAME}/best" \
    --max_train_samples 128 \
    --max_eval_samples 16 \
    --num_train_epochs 2 \
    --gradient_checkpointing \
    --local_files_only \
    --no_progress_bar
done
PYTHONPATH=. python -m thesis_exp.exp51_hmsa.determinism_smoke
