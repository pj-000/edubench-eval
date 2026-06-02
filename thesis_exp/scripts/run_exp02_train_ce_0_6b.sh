#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${MODEL_NAME_OR_PATH:-}" ]]; then
  echo "ERROR: set MODEL_NAME_OR_PATH to the 0.6B base model path or HF id." >&2
  echo "Example: MODEL_NAME_OR_PATH=/path/to/Qwen3-0.6B bash thesis_exp/scripts/run_exp02_train_ce_0_6b.sh" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

OUTPUT_DIR="${OUTPUT_DIR:-thesis_exp/outputs/exp02_ce_baseline}"
CHECKPOINT_OUTPUT_DIR="${CHECKPOINT_OUTPUT_DIR:-thesis_exp/artifacts/exp02_ce_baseline/checkpoints/edubench_evaluator_0_6b_ce}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-10}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-32}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
SEED="${SEED:-42}"
BF16="${BF16:-auto}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LOG_STEPS="${LOG_STEPS:-20}"
EFFECTIVE_BATCH_SIZE=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))

FP16_ARGS=()
if [[ "${FP16:-0}" == "1" ]]; then
  FP16_ARGS+=(--fp16)
fi

GRADIENT_CHECKPOINTING_ARGS=()
if [[ "${GRADIENT_CHECKPOINTING:-1}" == "1" ]]; then
  GRADIENT_CHECKPOINTING_ARGS+=(--gradient_checkpointing)
fi

TRUST_REMOTE_CODE_ARGS=()
if [[ "${TRUST_REMOTE_CODE:-1}" == "1" ]]; then
  TRUST_REMOTE_CODE_ARGS+=(--trust_remote_code)
fi

LOCAL_FILES_ONLY_ARGS=()
if [[ "${LOCAL_FILES_ONLY:-0}" == "1" ]]; then
  LOCAL_FILES_ONLY_ARGS+=(--local_files_only)
fi

OPTIONAL_ARGS=()
if [[ -n "${MAX_TRAIN_SAMPLES:-}" ]]; then
  OPTIONAL_ARGS+=(--max_train_samples "${MAX_TRAIN_SAMPLES}")
fi
if [[ -n "${MAX_EVAL_SAMPLES:-}" ]]; then
  OPTIONAL_ARGS+=(--max_eval_samples "${MAX_EVAL_SAMPLES}")
fi
if [[ "${EVAL_ONLY:-0}" == "1" ]]; then
  OPTIONAL_ARGS+=(--eval_only)
fi
if [[ -n "${CHECKPOINT_DIR:-}" ]]; then
  OPTIONAL_ARGS+=(--checkpoint_dir "${CHECKPOINT_DIR}")
fi

cat <<CONFIG
Exp2 CE baseline config
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
MAX_LENGTH=${MAX_LENGTH}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}
effective batch size=${EFFECTIVE_BATCH_SIZE}
OUTPUT_DIR=${OUTPUT_DIR}
CHECKPOINT_OUTPUT_DIR=${CHECKPOINT_OUTPUT_DIR}
BF16=${BF16}
FP16=${FP16:-0}
CONFIG

python -m thesis_exp.src.edujudge.exp02.build_exp02_dataset --print-summary
python -m thesis_exp.src.edujudge.exp02.sanity_check_exp02_train_setup

python -m thesis_exp.src.edujudge.exp02.train_ce_baseline \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --data_dir thesis_exp/outputs/exp02_ce_baseline/data \
  --output_dir "${OUTPUT_DIR}" \
  --checkpoint_output_dir "${CHECKPOINT_OUTPUT_DIR}" \
  --max_length "${MAX_LENGTH}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
  --max_grad_norm "${MAX_GRAD_NORM}" \
  --seed "${SEED}" \
  --bf16 "${BF16}" \
  --num_workers "${NUM_WORKERS}" \
  --log_steps "${LOG_STEPS}" \
  "${FP16_ARGS[@]}" \
  "${GRADIENT_CHECKPOINTING_ARGS[@]}" \
  "${TRUST_REMOTE_CODE_ARGS[@]}" \
  "${LOCAL_FILES_ONLY_ARGS[@]}" \
  "${OPTIONAL_ARGS[@]}"
