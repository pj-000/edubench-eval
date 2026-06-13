#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-llama_factory}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
QD_B1_CHECKPOINT_DIR="${QD_B1_CHECKPOINT_DIR:-thesis_exp/artifacts/exp06_question_disjoint_baselines/checkpoints/QD-B1_human_only_L1_weighted_ordinal/best}"
ALLOW_EXP09_ENV_OVERRIDES="${ALLOW_EXP09_ENV_OVERRIDES:-0}"

if [[ "${ALLOW_EXP09_ENV_OVERRIDES}" == "1" ]]; then
  FORMAL_RUN="${FORMAL_RUN:-1}"
  REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
  NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
  PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-32}"
  PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-8}"
  GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
  LEARNING_RATE="${LEARNING_RATE:-1e-5}"
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
  NUM_TRAIN_EPOCHS="3"
  PER_DEVICE_TRAIN_BATCH_SIZE="32"
  PER_DEVICE_EVAL_BATCH_SIZE="8"
  GRADIENT_ACCUMULATION_STEPS="4"
  LEARNING_RATE="1e-5"
  WEIGHT_DECAY="0.01"
  WARMUP_RATIO="0.05"
  MAX_LENGTH="2048"
  BF16="auto"
  GRADIENT_CHECKPOINTING="1"
  LOG_STEPS="5"
  NO_PROGRESS_BAR="0"
fi
RESET_RUN_DIR="${RESET_RUN_DIR:-0}"
EXP09_QDPR2_PREFLIGHT_ONLY="${EXP09_QDPR2_PREFLIGHT_ONLY:-0}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "${FORMAL_RUN}" != "1" ]]; then
  echo "ERROR: QD-PR2 formal train script must run with FORMAL_RUN=1." >&2
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

if [[ ! -f "${QD_B1_CHECKPOINT_DIR}/state_dict.pt" ]]; then
  echo "BLOCKED_MISSING_QDB1_CHECKPOINT: ${QD_B1_CHECKPOINT_DIR}" >&2
  exit 1
fi

unset MAX_TRAIN_SAMPLES
unset MAX_EVAL_SAMPLES
unset MAX_TRAIN_PAIRS
unset MAX_DEV_PAIRS
unset FP16
unset EVAL_ONLY

export CUDA_VISIBLE_DEVICES
export MODEL_NAME_OR_PATH
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF
export FORMAL_RUN
export REQUIRE_CUDA

RUN_TAG="exp09_qdpr2_formal_$(date +%Y%m%d_%H%M%S)"
RUN_ID="QD-PR2_AnchoredPairwiseOrdinal_human_only"
OUTPUT_DIR="thesis_exp/outputs/exp09_pairwise_ordinal_qdpr2/runs/${RUN_ID}"
CHECKPOINT_DIR="thesis_exp/artifacts/exp09_pairwise_ordinal_qdpr2/checkpoints/${RUN_ID}"
LOG_DIR="thesis_exp/outputs/exp09_pairwise_ordinal_qdpr2/logs"
CONFIG_PATH="thesis_exp/configs/exp09_pairwise_ordinal/exp09_qdpr2_anchored_pairwise_human_only.yaml"
mkdir -p "${LOG_DIR}"
log_path="${LOG_DIR}/train_${RUN_ID}_${RUN_TAG}.log"
exec > >(tee -a "${log_path}") 2>&1

EFFECTIVE_BATCH_SIZE=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
if [[ "${EFFECTIVE_BATCH_SIZE}" != "128" ]]; then
  echo "ERROR: effective batch size must remain 128, got ${EFFECTIVE_BATCH_SIZE}." >&2
  exit 1
fi

cat <<CONFIG
Exp9 QD-PR2 anchored pairwise formal fine-tuning
LOG_PATH=${log_path}
RUN_TAG=${RUN_TAG}
ALLOW_EXP09_ENV_OVERRIDES=${ALLOW_EXP09_ENV_OVERRIDES}
FORMAL_RUN=${FORMAL_RUN}
REQUIRE_CUDA=${REQUIRE_CUDA}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}
QD_B1_CHECKPOINT_DIR=${QD_B1_CHECKPOINT_DIR}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
RUN_ID=${RUN_ID}
OUTPUT_DIR=${OUTPUT_DIR}
CHECKPOINT_DIR=${CHECKPOINT_DIR}
CONFIG_PATH=${CONFIG_PATH}
DATASET=question_seed42 human-only A4
PAIR_DATASET=train:10000 dev:3000 high-comparability
INITIALIZATION=QD-B1 checkpoint
LOSS=L_point + 0.05*L_pair + 0.5*L_anchor + 0.1*L_mono
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}
effective batch size=${EFFECTIVE_BATCH_SIZE}
LEARNING_RATE=${LEARNING_RATE}
WEIGHT_DECAY=${WEIGHT_DECAY}
WARMUP_RATIO=${WARMUP_RATIO}
MAX_LENGTH=${MAX_LENGTH}
BF16=${BF16}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING}
RESET_RUN_DIR=${RESET_RUN_DIR}
EXP09_QDPR2_PREFLIGHT_ONLY=${EXP09_QDPR2_PREFLIGHT_ONLY}
SYNTHETIC=disabled
CONFIG

python -m thesis_exp.src.edujudge.exp09_pairwise_ordinal.sanity_check_qdpr2_setup

progress_args=()
if [[ "${NO_PROGRESS_BAR}" == "1" ]]; then
  progress_args+=(--no_progress_bar)
fi

gc_args=()
if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then
  gc_args+=(--gradient_checkpointing)
fi

if [[ "${RESET_RUN_DIR}" == "1" ]]; then
  echo "Resetting existing QD-PR2 formal outputs/checkpoints"
  rm -rf "${OUTPUT_DIR}" "${CHECKPOINT_DIR}"
fi

if [[ "${EXP09_QDPR2_PREFLIGHT_ONLY}" == "1" ]]; then
  python -m thesis_exp.src.edujudge.exp09_pairwise_ordinal.train_qdpr2_anchored_pairwise \
    --config_path "${CONFIG_PATH}" \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --qd_b1_checkpoint_dir "${QD_B1_CHECKPOINT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --checkpoint_output_dir "${CHECKPOINT_DIR}" \
    --preflight_only
  echo "EXP09_QDPR2_PREFLIGHT_ONLY=1; exiting before training."
  exit 0
fi

echo "Launching QD-PR2 anchored fine-tuning command"
python -m thesis_exp.src.edujudge.exp09_pairwise_ordinal.train_qdpr2_anchored_pairwise \
  --config_path "${CONFIG_PATH}" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --qd_b1_checkpoint_dir "${QD_B1_CHECKPOINT_DIR}" \
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

python -m thesis_exp.src.edujudge.exp09_pairwise_ordinal.readability_check_qdpr2
cat "${OUTPUT_DIR}/run_summary.md"
