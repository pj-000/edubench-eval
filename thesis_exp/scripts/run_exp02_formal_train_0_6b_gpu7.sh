#!/usr/bin/env bash
set -euo pipefail

# Edit this block if you want to change the formal Exp2 run.
CONDA_ENV="llama_factory"
MODEL_NAME_OR_PATH="/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B"
CUDA_VISIBLE_DEVICES="7"

NUM_TRAIN_EPOCHS="10"
PER_DEVICE_TRAIN_BATCH_SIZE="4"
PER_DEVICE_EVAL_BATCH_SIZE="4"
GRADIENT_ACCUMULATION_STEPS="32"
LEARNING_RATE="2e-5"
WEIGHT_DECAY="0.01"
WARMUP_RATIO="0.05"
MAX_LENGTH="2048"
BF16="auto"
GRADIENT_CHECKPOINTING="1"

OUTPUT_DIR="thesis_exp/outputs/exp02_ce_baseline"
CHECKPOINT_OUTPUT_DIR="thesis_exp/artifacts/exp02_ce_baseline/checkpoints/edubench_evaluator_0_6b_ce"
RUN_ID="formal_$(date +%Y%m%d_%H%M%S)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${HOME}/miniconda3/bin/activate" ]]; then
  # Keep this inside the script so it can be executed directly after SSH login.
  source "${HOME}/miniconda3/bin/activate" "${CONDA_ENV}"
else
  echo "WARNING: ${HOME}/miniconda3/bin/activate was not found; using the current shell environment." >&2
fi

# Formal-run safety: never inherit smoke-test subset variables from the shell.
unset MAX_TRAIN_SAMPLES
unset MAX_EVAL_SAMPLES
unset FP16
unset EVAL_ONLY
unset CHECKPOINT_DIR

export MODEL_NAME_OR_PATH
export CUDA_VISIBLE_DEVICES
export OUTPUT_DIR
export CHECKPOINT_OUTPUT_DIR
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
export FORMAL_RUN=1
export REQUIRE_CUDA=1

mkdir -p "${OUTPUT_DIR}/logs"
LOG_PATH="${OUTPUT_DIR}/logs/train_${RUN_ID}.log"
POSTPROCESS_LOG_PATH="${OUTPUT_DIR}/logs/postprocess_${RUN_ID}.log"

cat <<CONFIG
Exp2 formal training wrapper
RUN_ID=${RUN_ID}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
OUTPUT_DIR=${OUTPUT_DIR}
CHECKPOINT_OUTPUT_DIR=${CHECKPOINT_OUTPUT_DIR}
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
FP16=unset
FORMAL_RUN=${FORMAL_RUN}
REQUIRE_CUDA=${REQUIRE_CUDA}
LOG_PATH=${LOG_PATH}
POSTPROCESS_LOG_PATH=${POSTPROCESS_LOG_PATH}
CONFIG

bash thesis_exp/scripts/run_exp02_train_ce_0_6b.sh 2>&1 | tee "${LOG_PATH}"

python -m thesis_exp.src.edujudge.exp02.postprocess_exp02_results 2>&1 | tee "${POSTPROCESS_LOG_PATH}"
cat "${OUTPUT_DIR}/postprocess_check.md"
