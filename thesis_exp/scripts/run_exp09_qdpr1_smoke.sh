#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-llama_factory}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"

FORMAL_RUN="${FORMAL_RUN:-0}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
EXP09_EXISTING_EXP0_EXP8_DIRTY_OK="${EXP09_EXISTING_EXP0_EXP8_DIRTY_OK:-1}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-8}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-8}"
MAX_TRAIN_PAIRS="${MAX_TRAIN_PAIRS:-8}"
MAX_DEV_PAIRS="${MAX_DEV_PAIRS:-8}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-0.01}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
MAX_LENGTH="${MAX_LENGTH:-512}"
BF16="${BF16:-auto}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
LOG_STEPS="${LOG_STEPS:-1}"
NO_PROGRESS_BAR="${NO_PROGRESS_BAR:-0}"
RESET_RUN_DIR="${RESET_RUN_DIR:-1}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "${FORMAL_RUN}" != "0" ]]; then
  echo "ERROR: QD-PR1 smoke script must run with FORMAL_RUN=0." >&2
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

export CUDA_VISIBLE_DEVICES
export MODEL_NAME_OR_PATH
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF
export FORMAL_RUN
export REQUIRE_CUDA
export EXP09_EXISTING_EXP0_EXP8_DIRTY_OK

RUN_TAG="exp09_qdpr1_smoke_$(date +%Y%m%d_%H%M%S)"
RUN_ID="QD-PR1_PairwiseRiskOrdinal_human_only"
OUTPUT_DIR="thesis_exp/outputs/exp09_pairwise_ordinal/smoke_test/${RUN_ID}"
CHECKPOINT_DIR="thesis_exp/artifacts/exp09_pairwise_ordinal/smoke_test/${RUN_ID}"
LOG_DIR="thesis_exp/outputs/exp09_pairwise_ordinal/logs"
CONFIG_PATH="thesis_exp/configs/exp09_pairwise_ordinal/exp09_qdpr1_pairwise_smoke.yaml"
mkdir -p "${LOG_DIR}"

cat <<CONFIG
Exp9 QD-PR1 pairwise ordinal smoke training
RUN_TAG=${RUN_TAG}
FORMAL_RUN=${FORMAL_RUN}
REQUIRE_CUDA=${REQUIRE_CUDA}
EXP09_EXISTING_EXP0_EXP8_DIRTY_OK=${EXP09_EXISTING_EXP0_EXP8_DIRTY_OK}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
RUN_ID=${RUN_ID}
OUTPUT_DIR=${OUTPUT_DIR}
CHECKPOINT_DIR=${CHECKPOINT_DIR}
CONFIG_PATH=${CONFIG_PATH}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES}
MAX_TRAIN_PAIRS=${MAX_TRAIN_PAIRS}
MAX_DEV_PAIRS=${MAX_DEV_PAIRS}
PAIR_DATASET=train:20000 dev:5000
PAIR_LOSS=softplus(margin - score_gap)
POINTWISE_LOSS=QD-B1-style weighted ordinal BCE
SYNTHETIC=disabled
CORAL=disabled
EDURISK=disabled
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
  rm -rf "${OUTPUT_DIR}" "${CHECKPOINT_DIR}"
fi

python -m thesis_exp.src.edujudge.exp09_pairwise_ordinal.sanity_check_exp09_setup

log_path="${LOG_DIR}/train_${RUN_ID}_${RUN_TAG}.log"
python -m thesis_exp.src.edujudge.exp09_pairwise_ordinal.train_qdpr1_pairwise \
  --config_path "${CONFIG_PATH}" \
  --smoke \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --checkpoint_output_dir "${CHECKPOINT_DIR}" \
  --max_length "${MAX_LENGTH}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --learning_rate "${LEARNING_RATE}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --max_train_samples "${MAX_TRAIN_SAMPLES}" \
  --max_eval_samples "${MAX_EVAL_SAMPLES}" \
  --max_train_pairs "${MAX_TRAIN_PAIRS}" \
  --max_dev_pairs "${MAX_DEV_PAIRS}" \
  --bf16 "${BF16}" \
  --log_steps "${LOG_STEPS}" \
  "${progress_args[@]}" \
  "${gc_args[@]}" 2>&1 | tee "${log_path}"

python -m thesis_exp.src.edujudge.exp09_pairwise_ordinal.collect_exp09_results
python -m thesis_exp.src.edujudge.exp09_pairwise_ordinal.readability_check_exp09
cat "${OUTPUT_DIR}/run_summary.md"
