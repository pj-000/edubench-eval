#!/usr/bin/env bash
# Run one source-locked Exp57 scientific-cohort job on an RTX A6000.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

VARIANT="${VARIANT:?Set one frozen Exp57 scientific variant}"
SEED="${SEED:?Set one frozen seed}"
GPU_ID="${GPU_ID:?Set an available RTX A6000 index}"
PYTHON_BIN="${PYTHON_BIN:-/home/jpang/miniconda3/envs/llama_factory/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"

case "${VARIANT}" in
  dual_hard|consensus_only|routed_hmsa|residual_only|sign_flipped|shuffled_residual) ;;
  *) echo "Unauthorized A6000 scientific variant: ${VARIANT}" >&2; exit 2 ;;
esac
case "${SEED}" in
  42|43|44|45|46) ;;
  *) echo "Seed is outside the frozen Exp57 campaign: ${SEED}" >&2; exit 2 ;;
esac
GPU_NAME="$(nvidia-smi --id="${GPU_ID}" --query-gpu=name --format=csv,noheader,nounits)"
if [[ "${GPU_NAME}" != *"A6000"* ]]; then
  echo "GPU ${GPU_ID} is not an RTX A6000 (${GPU_NAME})" >&2
  exit 2
fi

OUTPUT_DIR="thesis_exp/outputs/exp57_cbrd/runs_a6000/${VARIANT}/seed_${SEED}"
CHECKPOINT_DIR="thesis_exp/artifacts/exp57_cbrd_a6000/${VARIANT}/seed_${SEED}"
LOG_DIR="thesis_exp/outputs/exp57_cbrd/logs_private_a6000"
mkdir -p "${LOG_DIR}"
if [[ -e "${OUTPUT_DIR}/run_summary.json" || -e "${CHECKPOINT_DIR}/best/state_dict.pt" ]]; then
  echo "Refusing to overwrite an existing A6000 cohort run: ${VARIANT} seed ${SEED}" >&2
  exit 3
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export EXP57_REQUIRE_SOURCE_LOCK=1
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -m thesis_exp.exp57_cbrd.train \
  --variant "${VARIANT}" \
  --model_name_or_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --checkpoint_output_dir "${CHECKPOINT_DIR}" \
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
