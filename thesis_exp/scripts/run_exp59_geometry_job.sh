#!/usr/bin/env bash
# Run one source-locked Exp59 train/dev job on an explicitly selected RTX 3090.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SEED="${SEED:?Set one frozen seed from 42 to 46}"
VARIANT="${VARIANT:?Set parallel_only or orthogonal_only}"
GPU_ID="${GPU_ID:?Set an available RTX 3090 index}"
PYTHON_BIN="${PYTHON_BIN:-/home/jpang/miniconda3/envs/llama_factory/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"

case "${SEED}" in 42|43|44|45|46) ;; *) exit 2 ;; esac
case "${VARIANT}" in parallel_only|orthogonal_only) ;; *) exit 2 ;; esac
GPU_NAME="$(nvidia-smi --id="${GPU_ID}" --query-gpu=name --format=csv,noheader,nounits)"
if [[ "${GPU_NAME}" != *"3090"* ]]; then
  echo "GPU ${GPU_ID} is not an RTX 3090 (${GPU_NAME})" >&2
  exit 2
fi

OUTPUT_DIR="thesis_exp/outputs/exp59_residual_geometry/runs/${VARIANT}/seed_${SEED}"
CHECKPOINT_DIR="thesis_exp/artifacts/exp59_residual_geometry/${VARIANT}/seed_${SEED}"
LOG_DIR="thesis_exp/outputs/exp59_residual_geometry/logs_private"
mkdir -p "${LOG_DIR}"
if [[ -e "${OUTPUT_DIR}/run_summary.json" || -e "${CHECKPOINT_DIR}/best/state_dict.pt" ]]; then
  echo "Refusing to overwrite Exp59 ${VARIANT} seed ${SEED}" >&2
  exit 3
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export EXP59_REQUIRE_SOURCE_LOCK=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -m thesis_exp.exp59_residual_geometry.train \
  --model_name_or_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --checkpoint_output_dir "${CHECKPOINT_DIR}" \
  --variant "${VARIANT}" \
  --seed "${SEED}" \
  --max_length 2048 \
  --num_train_epochs 10 \
  --learning_rate 2e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.05 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 32 \
  --max_grad_norm 1.0 \
  --bf16 true \
  --gradient_checkpointing \
  --local_files_only \
  2>&1 | tee "${LOG_DIR}/${VARIANT}_seed_${SEED}.log"
