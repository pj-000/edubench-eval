#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-llama_factory}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-10}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-32}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
BF16="${BF16:-auto}"
# Same formal budget as Exp3: effective batch size 4 * 32 = 128. Leave
# checkpointing off for 3090 speed; set GRADIENT_CHECKPOINTING=1 if OOM.
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
LOG_STEPS="${LOG_STEPS:-5}"
NO_PROGRESS_BAR="${NO_PROGRESS_BAR:-0}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
FORMAL_RUN="${FORMAL_RUN:-1}"
RUN_REGRESSION="${RUN_REGRESSION:-1}"
RUN_ORDINAL="${RUN_ORDINAL:-1}"

if [[ "${FORMAL_RUN}" == "1" && ( -n "${MAX_TRAIN_SAMPLES:-}" || -n "${MAX_EVAL_SAMPLES:-}" ) ]]; then
  echo "ERROR: FORMAL_RUN=1 cannot be used with MAX_TRAIN_SAMPLES or MAX_EVAL_SAMPLES." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ ! -f "thesis_exp/outputs/exp03_input_ablation/runs/A4_question_answer_metric_rubric_metadata/tables/metrics_summary.csv" ]]; then
  echo "ERROR: Exp3 A4 run output is missing; sync Exp3 results before Exp4." >&2
  exit 1
fi

if [[ ! -d "thesis_exp/outputs/exp03_input_ablation/datasets/A4_question_answer_metric_rubric_metadata" ]]; then
  echo "ERROR: Exp3 A4 dataset is missing; build or sync Exp3 datasets before Exp4." >&2
  exit 1
fi

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

RUN_ID="formal_exp04_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="thesis_exp/outputs/exp04_target_objectives/logs"
mkdir -p "${LOG_DIR}"

cat <<CONFIG
Exp4 formal target-objective wrapper
RUN_ID=${RUN_ID}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
OBJECTIVES=O1_classification O2_regression_smoothl1 O3_ordinal
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}
effective batch size=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
MAX_LENGTH=${MAX_LENGTH}
BF16=${BF16}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING}
LOG_STEPS=${LOG_STEPS}
PROGRESS_BAR=$([[ "${NO_PROGRESS_BAR}" == "1" ]] && echo disabled || echo enabled)
FORMAL_RUN=${FORMAL_RUN}
REQUIRE_CUDA=${REQUIRE_CUDA}
RUN_REGRESSION=${RUN_REGRESSION}
RUN_ORDINAL=${RUN_ORDINAL}
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}
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

python -m thesis_exp.src.edujudge.exp04.build_exp04_dataset --force

python -m thesis_exp.src.edujudge.exp04.train_objective \
  --objective_type classification \
  --objective_id O1_classification \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --reuse_exp03_a4 \
  --trust_remote_code \
  --local_files_only

python -m thesis_exp.src.edujudge.exp04.postprocess_exp04_results

progress_args=()
if [[ "${NO_PROGRESS_BAR}" == "1" ]]; then
  progress_args+=(--no_progress_bar)
fi

gc_args=()
if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then
  gc_args+=(--gradient_checkpointing)
fi

run_objective() {
  local objective_type="$1"
  local objective_id="$2"
  local regression_loss="$3"
  local log_path="${LOG_DIR}/train_${objective_id}_${RUN_ID}.log"
  local postprocess_log_path="${LOG_DIR}/postprocess_${objective_id}_${RUN_ID}.log"

  echo "Starting ${objective_id}; log=${log_path}"
  python -m thesis_exp.src.edujudge.exp04.train_objective \
    --objective_type "${objective_type}" \
    --objective_id "${objective_id}" \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --data_dir "thesis_exp/outputs/exp04_target_objectives/datasets/A4_fixed_question_answer_metric_rubric_metadata" \
    --output_dir "thesis_exp/outputs/exp04_target_objectives/runs/${objective_id}" \
    --checkpoint_output_dir "thesis_exp/artifacts/exp04_target_objectives/checkpoints/${objective_id}" \
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
    --regression_loss "${regression_loss}" \
    --trust_remote_code \
    --local_files_only \
    "${progress_args[@]}" \
    "${gc_args[@]}" 2>&1 | tee "${log_path}"

  python -m thesis_exp.src.edujudge.exp04.postprocess_exp04_results 2>&1 | tee "${postprocess_log_path}"
}

if [[ "${RUN_REGRESSION}" == "1" ]]; then
  run_objective regression O2_regression_smoothl1 smoothl1
else
  echo "Skipping O2_regression_smoothl1 because RUN_REGRESSION=${RUN_REGRESSION}"
fi

if [[ "${RUN_ORDINAL}" == "1" ]]; then
  run_objective ordinal O3_ordinal smoothl1
else
  echo "Skipping O3_ordinal because RUN_ORDINAL=${RUN_ORDINAL}"
fi

python -m thesis_exp.src.edujudge.exp04.postprocess_exp04_results --strict
python -m thesis_exp.src.edujudge.exp04.sanity_check_exp04_outputs --strict
python -m thesis_exp.src.edujudge.exp04.readability_check_exp04
cat thesis_exp/outputs/exp04_target_objectives/review_package.md
