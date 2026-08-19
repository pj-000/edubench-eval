#!/usr/bin/env bash
set -euo pipefail

STAGE="${STAGE:-smoke}"
GPU_ID="${GPU_ID:-6}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"

case "${STAGE}" in
  smoke)
    MAX_TRAIN=16
    MAX_EVAL=16
    EPOCHS=1
    OUTPUT="thesis_exp/outputs/exp56_meanaux/smoke/seed_42"
    CHECKPOINT="thesis_exp/artifacts/exp56_meanaux/smoke/seed_42"
    SEED=42
    ;;
  seed42)
    MAX_TRAIN=""
    MAX_EVAL=""
    EPOCHS=10
    OUTPUT="thesis_exp/outputs/exp56_meanaux/runs/hard_main_mean_aux_lambda1/seed_42"
    CHECKPOINT="thesis_exp/artifacts/exp56_meanaux/hard_main_mean_aux_lambda1/seed_42"
    SEED=42
    ;;
  *)
    echo "STAGE must be smoke or seed42" >&2
    exit 2
    ;;
esac

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${GPU_ID}")"
GPU_USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU_ID}")"
[[ "${GPU_NAME}" == *"3090"* ]] || {
  echo "GPU ${GPU_ID} is not an RTX 3090: ${GPU_NAME}" >&2
  exit 2
}
(( GPU_USED < 2048 )) || {
  echo "GPU ${GPU_ID} is not sufficiently free: ${GPU_USED} MiB used" >&2
  exit 2
}

ARGS=(
  --model_name_or_path "${MODEL_PATH}"
  --output_dir "${OUTPUT}"
  --checkpoint_output_dir "${CHECKPOINT}"
  --max_length 2048
  --num_train_epochs "${EPOCHS}"
  --learning_rate 2e-5
  --weight_decay 0.01
  --warmup_ratio 0.05
  --per_device_train_batch_size 4
  --per_device_eval_batch_size 4
  --gradient_accumulation_steps 32
  --max_grad_norm 1.0
  --seed "${SEED}"
  --bf16 auto
  --gradient_checkpointing
  --local_files_only
  --num_workers 0
)

if [[ -n "${MAX_TRAIN}" ]]; then
  ARGS+=(--max_train_samples "${MAX_TRAIN}")
fi
if [[ -n "${MAX_EVAL}" ]]; then
  ARGS+=(--max_eval_samples "${MAX_EVAL}")
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${PYTHONPATH:-.}"
export EXP56_REQUIRE_SOURCE_LOCK=1
"${PYTHON_BIN}" -m thesis_exp.exp56_meanaux.train "${ARGS[@]}"
