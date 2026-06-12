#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-llama_factory}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
ALLOW_EXP09_ENV_OVERRIDES="${ALLOW_EXP09_ENV_OVERRIDES:-0}"

if [[ "${ALLOW_EXP09_ENV_OVERRIDES}" == "1" ]]; then
  FORMAL_RUN="${FORMAL_RUN:-1}"
  REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
  EXP09_EXISTING_EXP0_EXP8_DIRTY_OK="${EXP09_EXISTING_EXP0_EXP8_DIRTY_OK:-1}"
  NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-10}"
  PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-32}"
  PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-8}"
  GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
  LEARNING_RATE="${LEARNING_RATE:-2e-5}"
  WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
  WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
  MAX_LENGTH="${MAX_LENGTH:-2048}"
  BF16="${BF16:-auto}"
  GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
  LOG_STEPS="${LOG_STEPS:-5}"
  NO_PROGRESS_BAR="${NO_PROGRESS_BAR:-0}"
else
  FORMAL_RUN="1"
  REQUIRE_CUDA="1"
  EXP09_EXISTING_EXP0_EXP8_DIRTY_OK="1"
  NUM_TRAIN_EPOCHS="10"
  PER_DEVICE_TRAIN_BATCH_SIZE="32"
  PER_DEVICE_EVAL_BATCH_SIZE="8"
  GRADIENT_ACCUMULATION_STEPS="4"
  LEARNING_RATE="2e-5"
  WEIGHT_DECAY="0.01"
  WARMUP_RATIO="0.05"
  MAX_LENGTH="2048"
  BF16="auto"
  GRADIENT_CHECKPOINTING="1"
  LOG_STEPS="5"
  NO_PROGRESS_BAR="0"
fi
RESET_RUN_DIR="${RESET_RUN_DIR:-0}"
EXP09_SCRIPT_PREFLIGHT_ONLY="${EXP09_SCRIPT_PREFLIGHT_ONLY:-0}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "${FORMAL_RUN}" != "1" ]]; then
  echo "ERROR: QD-PR1 formal train script must run with FORMAL_RUN=1." >&2
  exit 1
fi

if [[ -n "${MAX_TRAIN_SAMPLES:-}" || -n "${MAX_EVAL_SAMPLES:-}" || -n "${MAX_TRAIN_PAIRS:-}" || -n "${MAX_DEV_PAIRS:-}" ]]; then
  echo "ERROR: FORMAL_RUN=1 cannot use max sample or max pair limits." >&2
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
unset MAX_TRAIN_PAIRS
unset MAX_DEV_PAIRS
unset FP16
unset EVAL_ONLY
unset CHECKPOINT_DIR

export CUDA_VISIBLE_DEVICES
export MODEL_NAME_OR_PATH
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF
export FORMAL_RUN
export REQUIRE_CUDA
export EXP09_EXISTING_EXP0_EXP8_DIRTY_OK

RUN_TAG="exp09_qdpr1_formal_$(date +%Y%m%d_%H%M%S)"
RUN_ID="QD-PR1_PairwiseRiskOrdinal_human_only"
OUTPUT_DIR="thesis_exp/outputs/exp09_pairwise_ordinal/runs/${RUN_ID}"
CHECKPOINT_DIR="thesis_exp/artifacts/exp09_pairwise_ordinal/checkpoints/${RUN_ID}"
LOG_DIR="thesis_exp/outputs/exp09_pairwise_ordinal/logs"
CONFIG_PATH="thesis_exp/configs/exp09_pairwise_ordinal/exp09_qdpr1_pairwise_human_only.yaml"
mkdir -p "${LOG_DIR}"
log_path="${LOG_DIR}/train_${RUN_ID}_${RUN_TAG}.log"
exec > >(tee -a "${log_path}") 2>&1

EFFECTIVE_BATCH_SIZE=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
if [[ "${EFFECTIVE_BATCH_SIZE}" != "128" ]]; then
  echo "ERROR: effective batch size must remain 128, got ${EFFECTIVE_BATCH_SIZE}." >&2
  exit 1
fi

cat <<CONFIG
Exp9 QD-PR1 pairwise ordinal formal training
LOG_PATH=${log_path}
RUN_TAG=${RUN_TAG}
ALLOW_EXP09_ENV_OVERRIDES=${ALLOW_EXP09_ENV_OVERRIDES}
FORMAL_RUN=${FORMAL_RUN}
REQUIRE_CUDA=${REQUIRE_CUDA}
EXP09_EXISTING_EXP0_EXP8_DIRTY_OK=${EXP09_EXISTING_EXP0_EXP8_DIRTY_OK}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
RUN_ID=${RUN_ID}
OUTPUT_DIR=${OUTPUT_DIR}
CHECKPOINT_DIR=${CHECKPOINT_DIR}
CONFIG_PATH=${CONFIG_PATH}
DATASET=QD-S0_human_only question_seed42 human-only
INPUT_TEMPLATE=A4 text field
PAIR_DATASET=train:20000 dev:5000
PAIR_TYPE_TARGETS=low_high:0.40 low_mid:0.20 adjacent:0.30 random_ordinal:0.10
HEAD=independent ordinal
POINTWISE_LOSS=QD-B1-style weighted ordinal BCE
PAIRWISE_LOSS=softplus(margin - score_gap)
LAMBDA_PAIR=0.3
LOW_HIGH_MARGIN=0.25
LOW_HIGH_WEIGHT=1.0
GAP_WEIGHT=0.5
SYNTHETIC=disabled
CORAL=disabled
EDURISK=disabled
CHECKPOINT_SELECTION=dev_MAE_label min
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}
effective batch size=${EFFECTIVE_BATCH_SIZE}
effective text sequences per optimizer step=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * 2))
LEARNING_RATE=${LEARNING_RATE}
WEIGHT_DECAY=${WEIGHT_DECAY}
WARMUP_RATIO=${WARMUP_RATIO}
MAX_LENGTH=${MAX_LENGTH}
BF16=${BF16}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING}
RESET_RUN_DIR=${RESET_RUN_DIR}
EXP09_SCRIPT_PREFLIGHT_ONLY=${EXP09_SCRIPT_PREFLIGHT_ONLY}
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
    for idx in range(torch.cuda.device_count()):
        print(f"torch.cuda.get_device_name({idx})={torch.cuda.get_device_name(idx)}")
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
  echo "Resetting existing QD-PR1 formal outputs/checkpoints"
  rm -rf "${OUTPUT_DIR}" "${CHECKPOINT_DIR}"
fi

python -m thesis_exp.src.edujudge.exp09_pairwise_ordinal.sanity_check_exp09_setup

if [[ "${EXP09_SCRIPT_PREFLIGHT_ONLY}" == "1" ]]; then
  echo "EXP09_SCRIPT_PREFLIGHT_ONLY=1; exiting before training."
  exit 0
fi

echo "Launching Exp9 training command"
python -m thesis_exp.src.edujudge.exp09_pairwise_ordinal.train_qdpr1_pairwise \
  --config_path "${CONFIG_PATH}" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --checkpoint_output_dir "${CHECKPOINT_DIR}" \
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
  "${progress_args[@]}" \
  "${gc_args[@]}"

python -m thesis_exp.src.edujudge.exp09_pairwise_ordinal.collect_exp09_results
python -m thesis_exp.src.edujudge.exp09_pairwise_ordinal.readability_check_exp09
cat thesis_exp/outputs/exp09_pairwise_ordinal/review_package.md
