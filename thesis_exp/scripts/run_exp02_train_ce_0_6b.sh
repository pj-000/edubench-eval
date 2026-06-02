#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${MODEL_NAME_OR_PATH:-}" ]]; then
  echo "ERROR: set MODEL_NAME_OR_PATH to the 0.6B base model path or HF id." >&2
  echo "Example: MODEL_NAME_OR_PATH=/path/to/Qwen3-0.6B bash thesis_exp/scripts/run_exp02_train_ce_0_6b.sh" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

python -m thesis_exp.src.edujudge.exp02.build_exp02_dataset --print-summary

python -m thesis_exp.src.edujudge.exp02.train_ce_baseline \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --data_dir thesis_exp/outputs/exp02_ce_baseline/data \
  --output_dir thesis_exp/outputs/exp02_ce_baseline/models/edubench_evaluator_0_6b_ce \
  --max_length "${MAX_LENGTH:-2048}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-3}" \
  --learning_rate "${LEARNING_RATE:-2e-5}" \
  --weight_decay "${WEIGHT_DECAY:-0.01}" \
  --warmup_ratio "${WARMUP_RATIO:-0.05}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-2}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE:-4}" \
  --max_grad_norm "${MAX_GRAD_NORM:-1.0}" \
  --seed "${SEED:-42}" \
  --bf16 "${BF16:-auto}" \
  --gradient_checkpointing \
  --trust_remote_code

