#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-llama_factory}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

FORMAL_RUN="${FORMAL_RUN:-0}"
REQUIRE_CUDA="${REQUIRE_CUDA:-0}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-8}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-8}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-0.01}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.0}"
MAX_LENGTH="${MAX_LENGTH:-512}"
BF16="${BF16:-auto}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
LOG_STEPS="${LOG_STEPS:-1}"
NO_PROGRESS_BAR="${NO_PROGRESS_BAR:-0}"
MU_THR="${MU_THR:-0.3}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "${FORMAL_RUN}" != "0" ]]; then
  echo "ERROR: L3b smoke script must run with FORMAL_RUN=0." >&2
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

RUN_TAG="exp05_l3b_smoke_$(date +%Y%m%d_%H%M%S)"
RUN_ID="L3b_weighted_threshold_mu03"
LOG_DIR="thesis_exp/outputs/exp05_low_score_loss/logs"
mkdir -p "${LOG_DIR}"

cat <<CONFIG
Exp5 L3b smoke test
RUN_TAG=${RUN_TAG}
FORMAL_RUN=${FORMAL_RUN}
REQUIRE_CUDA=${REQUIRE_CUDA}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
RUN_ID=${RUN_ID}
MU_THR=${MU_THR}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}
MAX_LENGTH=${MAX_LENGTH}
BF16=${BF16}
CONFIG

python - <<'PY'
import os

require_cuda = os.environ.get("REQUIRE_CUDA", "0") == "1"
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

progress_args=()
if [[ "${NO_PROGRESS_BAR}" == "1" ]]; then
  progress_args+=(--no_progress_bar)
fi

gc_args=()
if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then
  gc_args+=(--gradient_checkpointing)
fi

log_path="${LOG_DIR}/train_${RUN_ID}_${RUN_TAG}.log"
echo "Starting ${RUN_ID} smoke; mu_thr=${MU_THR}; log=${log_path}"
python -m thesis_exp.src.edujudge.exp05.train_l3b_threshold_ordinal \
  --smoke \
  --mu_thr "${MU_THR}" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --data_dir "thesis_exp/outputs/exp04_target_objectives/datasets/A4_fixed_question_answer_metric_rubric_metadata" \
  --output_dir "thesis_exp/outputs/exp05_low_score_loss/smoke_test/${RUN_ID}" \
  --checkpoint_output_dir "thesis_exp/artifacts/exp05_low_score_loss/smoke_test/${RUN_ID}" \
  --max_length "${MAX_LENGTH}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --max_train_samples "${MAX_TRAIN_SAMPLES}" \
  --max_eval_samples "${MAX_EVAL_SAMPLES}" \
  --bf16 "${BF16}" \
  --log_steps "${LOG_STEPS}" \
  --trust_remote_code \
  --local_files_only \
  "${progress_args[@]}" \
  "${gc_args[@]}" 2>&1 | tee "${log_path}"

python -m thesis_exp.src.edujudge.exp05.sanity_check_exp05_outputs --smoke --l3b-only --strict
python -m thesis_exp.src.edujudge.exp05.write_exp05_report
python -m thesis_exp.src.edujudge.exp05.readability_check_exp05

rm -rf "thesis_exp/artifacts/exp05_low_score_loss/smoke_test/${RUN_ID}"

cat > thesis_exp/outputs/exp05_low_score_loss/smoke_test/smoke_test_l3b_report.md <<REPORT
# Exp5 L3b Smoke Test Report

Status: completed

Run tag: ${RUN_TAG}

Output:

- thesis_exp/outputs/exp05_low_score_loss/smoke_test/${RUN_ID}

This smoke test uses max_train_samples=${MAX_TRAIN_SAMPLES}, max_eval_samples=${MAX_EVAL_SAMPLES}, num_train_epochs=${NUM_TRAIN_EPOCHS}, and mu_thr=${MU_THR}.
REPORT

cat thesis_exp/outputs/exp05_low_score_loss/smoke_test/smoke_test_l3b_report.md
