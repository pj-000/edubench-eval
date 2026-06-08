#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-llama_factory}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

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
  echo "ERROR: formal train script must run with FORMAL_RUN=1." >&2
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

export CUDA_VISIBLE_DEVICES
export MODEL_NAME_OR_PATH
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF

RUN_ID="exp05_l1_formal_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="thesis_exp/outputs/exp05_low_score_loss/logs"
mkdir -p "${LOG_DIR}"

cat <<CONFIG
Exp5 L1 formal training
RUN_ID=${RUN_ID}
FORMAL_RUN=${FORMAL_RUN}
REQUIRE_CUDA=${REQUIRE_CUDA}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
LOSS=L1_weighted_ordinal
L0_BASELINE=reuse Exp4 O3 ordinal, no retraining
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

python -m thesis_exp.src.edujudge.exp05.sanity_check_exp05_setup
python -m thesis_exp.src.edujudge.exp05.class_weights
cat thesis_exp/outputs/exp05_low_score_loss/tables/class_weights.csv

progress_args=()
if [[ "${NO_PROGRESS_BAR}" == "1" ]]; then
  progress_args+=(--no_progress_bar)
fi

gc_args=()
if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then
  gc_args+=(--gradient_checkpointing)
fi

python -m thesis_exp.src.edujudge.exp05.train_l1_weighted_ordinal \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --data_dir "thesis_exp/outputs/exp04_target_objectives/datasets/A4_fixed_question_answer_metric_rubric_metadata" \
  --output_dir "thesis_exp/outputs/exp05_low_score_loss/runs/L1_weighted_ordinal" \
  --checkpoint_output_dir "thesis_exp/artifacts/exp05_low_score_loss/checkpoints/L1_weighted_ordinal" \
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
  "${gc_args[@]}" 2>&1 | tee "${LOG_DIR}/train_${RUN_ID}.log"

python -m thesis_exp.src.edujudge.exp05.postprocess_exp05_results --strict
python -m thesis_exp.src.edujudge.exp05.sanity_check_exp05_outputs --strict
python -m thesis_exp.src.edujudge.exp05.write_exp05_report
python -m thesis_exp.src.edujudge.exp05.readability_check_exp05
cat thesis_exp/outputs/exp05_low_score_loss/review_package.md
