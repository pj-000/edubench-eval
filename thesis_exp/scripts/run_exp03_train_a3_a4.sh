#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-llama_factory}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-10}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-32}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
BF16="${BF16:-auto}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
LOG_STEPS="${LOG_STEPS:-5}"
NO_PROGRESS_BAR="${NO_PROGRESS_BAR:-0}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
FORMAL_RUN="${FORMAL_RUN:-1}"

if [[ "${FORMAL_RUN}" == "1" && ( -n "${MAX_TRAIN_SAMPLES:-}" || -n "${MAX_EVAL_SAMPLES:-}" ) ]]; then
  echo "ERROR: FORMAL_RUN=1 cannot be used with MAX_TRAIN_SAMPLES or MAX_EVAL_SAMPLES." >&2
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
unset FP16
unset EVAL_ONLY
unset CHECKPOINT_DIR

export MODEL_NAME_OR_PATH
export CUDA_VISIBLE_DEVICES
export NUM_TRAIN_EPOCHS
export PER_DEVICE_TRAIN_BATCH_SIZE
export PER_DEVICE_EVAL_BATCH_SIZE
export GRADIENT_ACCUMULATION_STEPS
export LEARNING_RATE
export WEIGHT_DECAY
export WARMUP_RATIO
export MAX_LENGTH
export BF16
export GRADIENT_CHECKPOINTING
export LOG_STEPS
export NO_PROGRESS_BAR
export PYTORCH_CUDA_ALLOC_CONF
export FORMAL_RUN=1
export REQUIRE_CUDA
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

RUN_ID="formal_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="thesis_exp/outputs/exp03_input_ablation/logs"
mkdir -p "${LOG_DIR}"

cat <<CONFIG
Exp3 formal A3/A4 training wrapper
RUN_ID=${RUN_ID}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
TEMPLATES=A3_question_answer_metric_rubric A4_question_answer_metric_rubric_metadata
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}
effective batch size=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
MAX_LENGTH=${MAX_LENGTH}
BF16=${BF16}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING}
FORMAL_RUN=${FORMAL_RUN}
REQUIRE_CUDA=${REQUIRE_CUDA}
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}
CONFIG

python - <<'PY'
import os
import sys

require_cuda = os.environ.get("REQUIRE_CUDA", "1") == "1"
try:
    import torch
except Exception as exc:
    print(f"torch import error: {type(exc).__name__}: {exc}")
    if require_cuda:
        raise SystemExit(1)
    raise SystemExit(0)

cuda_available = torch.cuda.is_available()
print(f"torch.cuda.is_available()={cuda_available}")
print(f"torch.cuda.device_count()={torch.cuda.device_count()}")
if cuda_available:
    print(f"torch.cuda.get_device_name(0)={torch.cuda.get_device_name(0)}")
print(f"torch.version.cuda={torch.version.cuda}")
print(f"selected BF16/FP16 setting=BF16={os.environ.get('BF16')}, FP16=unset")
if require_cuda and not cuda_available:
    raise SystemExit("ERROR: REQUIRE_CUDA=1 but CUDA is unavailable.")
PY

python -m thesis_exp.src.edujudge.exp03.sanity_check_exp03_setup

python -m thesis_exp.src.edujudge.exp03.train_input_ablation \
  --template_name A2_question_answer_metric \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --reuse_exp02 \
  --trust_remote_code \
  --local_files_only

run_template() {
  local template_name="$1"
  local output_dir="thesis_exp/outputs/exp03_input_ablation/runs/${template_name}"
  local checkpoint_dir="thesis_exp/artifacts/exp03_input_ablation/checkpoints/${template_name}"
  local log_path="${LOG_DIR}/train_${template_name}_${RUN_ID}.log"
  local postprocess_log_path="${LOG_DIR}/postprocess_${template_name}_${RUN_ID}.log"

  local progress_args=()
  if [[ "${NO_PROGRESS_BAR}" == "1" ]]; then
    progress_args+=(--no_progress_bar)
  fi

  local gc_args=()
  if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then
    gc_args+=(--gradient_checkpointing)
  fi

  echo "Starting ${template_name}; log=${log_path}"
  python -m thesis_exp.src.edujudge.exp03.train_input_ablation \
    --template_name "${template_name}" \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --data_dir "thesis_exp/outputs/exp03_input_ablation/datasets/${template_name}" \
    --output_dir "${output_dir}" \
    --checkpoint_output_dir "${checkpoint_dir}" \
    --max_length "${MAX_LENGTH}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --learning_rate "${LEARNING_RATE}" \
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

  python -m thesis_exp.src.edujudge.exp03.postprocess_exp03_results 2>&1 | tee "${postprocess_log_path}"
}

run_template A3_question_answer_metric_rubric
run_template A4_question_answer_metric_rubric_metadata

python -m thesis_exp.src.edujudge.exp03.sanity_check_exp03_outputs
cat thesis_exp/outputs/exp03_input_ablation/review_package.md
