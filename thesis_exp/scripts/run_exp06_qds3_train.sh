#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-llama_factory}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

FORMAL_RUN="${FORMAL_RUN:-1}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
STAGE1_NUM_TRAIN_EPOCHS="${STAGE1_NUM_TRAIN_EPOCHS:-3}"
STAGE2_NUM_TRAIN_EPOCHS="${STAGE2_NUM_TRAIN_EPOCHS:-10}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-32}"
STAGE1_LEARNING_RATE="${STAGE1_LEARNING_RATE:-2e-5}"
STAGE2_LEARNING_RATE="${STAGE2_LEARNING_RATE:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
BF16="${BF16:-auto}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
LOG_STEPS="${LOG_STEPS:-5}"
NO_PROGRESS_BAR="${NO_PROGRESS_BAR:-0}"
RESET_RUN_DIR="${RESET_RUN_DIR:-0}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "${FORMAL_RUN}" != "1" ]]; then
  echo "ERROR: QD-S3 formal train script must run with FORMAL_RUN=1." >&2
  exit 1
fi

if [[ -n "${MAX_TRAIN_SAMPLES:-}" || -n "${MAX_EVAL_SAMPLES:-}" || -n "${STAGE1_MAX_TRAIN_SAMPLES:-}" || -n "${STAGE2_MAX_TRAIN_SAMPLES:-}" ]]; then
  echo "ERROR: FORMAL_RUN=1 cannot use max-sample limits." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${HOME}/miniconda3/bin/activate" ]]; then
  source "${HOME}/miniconda3/bin/activate" "${CONDA_ENV}"
else
  echo "WARNING: ${HOME}/miniconda3/bin/activate was not found; using current shell." >&2
fi

unset MAX_TRAIN_SAMPLES
unset MAX_EVAL_SAMPLES
unset STAGE1_MAX_TRAIN_SAMPLES
unset STAGE2_MAX_TRAIN_SAMPLES
unset FP16
unset EVAL_ONLY
unset CHECKPOINT_DIR

export CUDA_VISIBLE_DEVICES
export MODEL_NAME_OR_PATH
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF
export FORMAL_RUN
export REQUIRE_CUDA

RUN_TAG="exp06_qds3_formal_$(date +%Y%m%d_%H%M%S)"
RUN_ID="QD-S3_synthetic_pretrain_then_human_finetune"
OUTPUT_DIR="thesis_exp/outputs/exp06_synthetic_low_score/training_runs/${RUN_ID}"
CHECKPOINT_DIR="thesis_exp/artifacts/exp06_synthetic_low_score/training_runs/checkpoints/${RUN_ID}"
LOG_DIR="thesis_exp/outputs/exp06_synthetic_low_score/training_runs/logs"
mkdir -p "${LOG_DIR}"

cat <<CONFIG
Exp6 QD-S3 formal training
RUN_TAG=${RUN_TAG}
FORMAL_RUN=${FORMAL_RUN}
REQUIRE_CUDA=${REQUIRE_CUDA}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
RUN_ID=${RUN_ID}
OUTPUT_DIR=${OUTPUT_DIR}
CHECKPOINT_DIR=${CHECKPOINT_DIR}
STAGE1_DATASET=384 synthetic low-score pseudo-label rows
STAGE2_DATASET=3326 question_seed42 human-only rows
DEV_TEST=question_seed42 human-only
LOSS=ordinary ordinal BCEWithLogitsLoss
CLASS_WEIGHTS=disabled
ASYMMETRIC_LOSS=disabled
CHECKPOINT_SELECTION=stage2 dev_MAE_label min
STAGE1_TEST_EVAL=disabled
STAGE1_NUM_TRAIN_EPOCHS=${STAGE1_NUM_TRAIN_EPOCHS}
STAGE2_NUM_TRAIN_EPOCHS=${STAGE2_NUM_TRAIN_EPOCHS}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}
effective batch size=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
STAGE1_LEARNING_RATE=${STAGE1_LEARNING_RATE}
STAGE2_LEARNING_RATE=${STAGE2_LEARNING_RATE}
WEIGHT_DECAY=${WEIGHT_DECAY}
WARMUP_RATIO=${WARMUP_RATIO}
MAX_LENGTH=${MAX_LENGTH}
BF16=${BF16}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING}
RESET_RUN_DIR=${RESET_RUN_DIR}
CONFIG

python - <<'PY'
import os

require_cuda = os.environ.get("REQUIRE_CUDA", "1") == "1"
try:
    import torch
except Exception as exc:
    print(f"torch import error: {type(exc).__name__}: {exc}")
    if require_cuda:
        raise SystemExit(1)
    raise SystemExit(0)

print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
print(f"torch.cuda.device_count()={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"torch.cuda.get_device_name(0)={torch.cuda.get_device_name(0)}")
print(f"torch.version.cuda={torch.version.cuda}")
if require_cuda and not torch.cuda.is_available():
    raise SystemExit("ERROR: REQUIRE_CUDA=1 but CUDA is unavailable.")
PY

progress_args=()
if [[ "${NO_PROGRESS_BAR}" == "1" ]]; then
  progress_args+=(--no_progress_bar)
fi

gc_args=()
if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then
  gc_args+=(--gradient_checkpointing)
fi

if [[ "${RESET_RUN_DIR}" == "1" ]]; then
  echo "Resetting existing QD-S3 formal outputs/checkpoints"
  rm -rf "${OUTPUT_DIR}" "${CHECKPOINT_DIR}"
fi

log_path="${LOG_DIR}/train_${RUN_ID}_${RUN_TAG}.log"
python -m thesis_exp.src.edujudge.exp06_training.train_qds3_curriculum \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --data_dir "thesis_exp/outputs/exp06_synthetic_low_score/training_datasets/${RUN_ID}" \
  --output_dir "${OUTPUT_DIR}" \
  --checkpoint_output_dir "${CHECKPOINT_DIR}" \
  --max_length "${MAX_LENGTH}" \
  --stage1_num_train_epochs "${STAGE1_NUM_TRAIN_EPOCHS}" \
  --stage2_num_train_epochs "${STAGE2_NUM_TRAIN_EPOCHS}" \
  --stage1_learning_rate "${STAGE1_LEARNING_RATE}" \
  --stage2_learning_rate "${STAGE2_LEARNING_RATE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --bf16 "${BF16}" \
  --log_steps "${LOG_STEPS}" \
  --trust_remote_code \
  --local_files_only \
  "${progress_args[@]}" \
  "${gc_args[@]}" 2>&1 | tee "${log_path}"

{
  echo
  echo "===== QD-S3 final run summary ====="
  cat "${OUTPUT_DIR}/run_summary.md"
  echo
  echo "===== QD-S3 metrics_summary.csv ====="
  cat "${OUTPUT_DIR}/tables/metrics_summary.csv"
  echo
  echo "===== QD-S3 low_score_metrics.csv ====="
  cat "${OUTPUT_DIR}/tables/low_score_metrics.csv"
  echo
  echo "===== QD-S3 high_score_metrics.csv ====="
  cat "${OUTPUT_DIR}/tables/high_score_metrics.csv"
  echo
  echo "===== QD-S3 review package ====="
  cat "${OUTPUT_DIR}/review_package.md"
} | tee -a "${log_path}"
