#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-llama_factory}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
GPU_L2A="${GPU_L2A:-6}"
GPU_L2B="${GPU_L2B:-7}"
VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_L2A},${GPU_L2B}}"
RUN_MODE="${RUN_MODE:-parallel}"
RESET_RUN_DIRS="${RESET_RUN_DIRS:-0}"

FORMAL_RUN="${FORMAL_RUN:-1}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-10}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-32}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
BF16="${BF16:-auto}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
LOG_STEPS="${LOG_STEPS:-5}"
NO_PROGRESS_BAR="${NO_PROGRESS_BAR:-0}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "${FORMAL_RUN}" != "1" ]]; then
  echo "ERROR: L2 formal train script must run with FORMAL_RUN=1." >&2
  exit 1
fi

if [[ -n "${MAX_TRAIN_SAMPLES:-}" || -n "${MAX_EVAL_SAMPLES:-}" ]]; then
  echo "ERROR: FORMAL_RUN=1 cannot use MAX_TRAIN_SAMPLES or MAX_EVAL_SAMPLES." >&2
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
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF

RUN_ID="exp05_l2_formal_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="thesis_exp/outputs/exp05_low_score_loss/logs"
mkdir -p "${LOG_DIR}"

cat <<CONFIG
Exp5 L2 formal training
RUN_ID=${RUN_ID}
FORMAL_RUN=${FORMAL_RUN}
REQUIRE_CUDA=${REQUIRE_CUDA}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}
RUN_MODE=${RUN_MODE}
GPU_L2A=${GPU_L2A}
GPU_L2B=${GPU_L2B}
VISIBLE_DEVICES=${VISIBLE_DEVICES}
RESET_RUN_DIRS=${RESET_RUN_DIRS}
RUNS=L2a_asymmetric_ordinal_lambda03_margin0 L2b_asymmetric_ordinal_lambda05_margin0
L2a lambda_low=0.3 margin=0.0
L2b lambda_low=0.5 margin=0.0
L0_BASELINE=reuse Exp4 O3 ordinal, no retraining
L1_BASELINE=reuse completed weighted ordinal result, no retraining
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}
effective batch size=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
LEARNING_RATE=${LEARNING_RATE}
WEIGHT_DECAY=${WEIGHT_DECAY}
WARMUP_RATIO=${WARMUP_RATIO}
MAX_LENGTH=${MAX_LENGTH}
BF16=${BF16}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING}
CONFIG

CUDA_VISIBLE_DEVICES="${VISIBLE_DEVICES}" python - <<'PY'
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
for idx in range(torch.cuda.device_count()):
    print(f"torch.cuda.get_device_name({idx})={torch.cuda.get_device_name(idx)}")
print(f"torch.version.cuda={torch.version.cuda}")
if require_cuda and not torch.cuda.is_available():
    raise SystemExit("ERROR: REQUIRE_CUDA=1 but CUDA is unavailable.")
PY

python -m thesis_exp.src.edujudge.exp05.sanity_check_exp05_setup

progress_args=()
if [[ "${NO_PROGRESS_BAR}" == "1" ]]; then
  progress_args+=(--no_progress_bar)
fi

gc_args=()
if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then
  gc_args+=(--gradient_checkpointing)
fi

reset_run_dir() {
  local run_id="$1"
  if [[ "${RESET_RUN_DIRS}" == "1" ]]; then
    echo "Resetting existing formal outputs/checkpoints for ${run_id}"
    rm -rf \
      "thesis_exp/outputs/exp05_low_score_loss/runs/${run_id}" \
      "thesis_exp/artifacts/exp05_low_score_loss/checkpoints/${run_id}"
  fi
}

run_l2() {
  local gpu_id="$1"
  local run_id="$2"
  local lambda_low="$3"
  local margin="$4"
  local log_path="${LOG_DIR}/train_${run_id}_gpu${gpu_id}_${RUN_ID}.log"

  echo "Starting ${run_id} on GPU ${gpu_id}; lambda_low=${lambda_low}; margin=${margin}; log=${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" python -m thesis_exp.src.edujudge.exp05.train_l2_asymmetric_ordinal \
    --run_id "${run_id}" \
    --lambda_low "${lambda_low}" \
    --margin "${margin}" \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --data_dir "thesis_exp/outputs/exp04_target_objectives/datasets/A4_fixed_question_answer_metric_rubric_metadata" \
    --output_dir "thesis_exp/outputs/exp05_low_score_loss/runs/${run_id}" \
    --checkpoint_output_dir "thesis_exp/artifacts/exp05_low_score_loss/checkpoints/${run_id}" \
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
}

reset_run_dir L2a_asymmetric_ordinal_lambda03_margin0
reset_run_dir L2b_asymmetric_ordinal_lambda05_margin0

if [[ "${RUN_MODE}" == "parallel" ]]; then
  run_l2 "${GPU_L2A}" L2a_asymmetric_ordinal_lambda03_margin0 0.3 0.0 &
  pid_l2a=$!
  run_l2 "${GPU_L2B}" L2b_asymmetric_ordinal_lambda05_margin0 0.5 0.0 &
  pid_l2b=$!

  status_l2a=0
  status_l2b=0
  wait "${pid_l2a}" || status_l2a=$?
  wait "${pid_l2b}" || status_l2b=$?
  if [[ "${status_l2a}" != "0" || "${status_l2b}" != "0" ]]; then
    echo "ERROR: L2 parallel training failed: L2a=${status_l2a}, L2b=${status_l2b}" >&2
    exit 1
  fi
elif [[ "${RUN_MODE}" == "sequential" ]]; then
  run_l2 "${GPU_L2A}" L2a_asymmetric_ordinal_lambda03_margin0 0.3 0.0
  run_l2 "${GPU_L2B}" L2b_asymmetric_ordinal_lambda05_margin0 0.5 0.0
else
  echo "ERROR: RUN_MODE must be parallel or sequential; got ${RUN_MODE}" >&2
  exit 1
fi

python -m thesis_exp.src.edujudge.exp05.postprocess_exp05_results --strict
python -m thesis_exp.src.edujudge.exp05.sanity_check_exp05_outputs --strict
python -m thesis_exp.src.edujudge.exp05.write_exp05_report
python -m thesis_exp.src.edujudge.exp05.readability_check_exp05
cat thesis_exp/outputs/exp05_low_score_loss/review_package.md
