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
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "${FORMAL_RUN}" != "0" ]]; then
  echo "ERROR: L2 smoke script must run with FORMAL_RUN=0." >&2
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

RUN_ID="exp05_l2_smoke_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="thesis_exp/outputs/exp05_low_score_loss/logs"
mkdir -p "${LOG_DIR}"

cat <<CONFIG
Exp5 L2 smoke test
RUN_ID=${RUN_ID}
FORMAL_RUN=${FORMAL_RUN}
REQUIRE_CUDA=${REQUIRE_CUDA}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
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

run_l2_smoke() {
  local run_id="$1"
  local lambda_low="$2"
  local margin="$3"
  local log_path="${LOG_DIR}/train_${run_id}_${RUN_ID}.log"

  echo "Starting ${run_id} smoke; lambda_low=${lambda_low}; margin=${margin}; log=${log_path}"
  python -m thesis_exp.src.edujudge.exp05.train_l2_asymmetric_ordinal \
    --smoke \
    --run_id "${run_id}" \
    --lambda_low "${lambda_low}" \
    --margin "${margin}" \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --data_dir "thesis_exp/outputs/exp04_target_objectives/datasets/A4_fixed_question_answer_metric_rubric_metadata" \
    --output_dir "thesis_exp/outputs/exp05_low_score_loss/smoke_test/${run_id}" \
    --checkpoint_output_dir "thesis_exp/artifacts/exp05_low_score_loss/smoke_test/${run_id}" \
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
}

run_l2_smoke L2a_asymmetric_ordinal_lambda03_margin0 0.3 0.0
run_l2_smoke L2b_asymmetric_ordinal_lambda05_margin0 0.5 0.0

python -m thesis_exp.src.edujudge.exp05.sanity_check_exp05_outputs --smoke --l2-only --strict
python -m thesis_exp.src.edujudge.exp05.write_exp05_report
python -m thesis_exp.src.edujudge.exp05.readability_check_exp05

cat > thesis_exp/outputs/exp05_low_score_loss/smoke_test/smoke_test_l2_report.md <<REPORT
# Exp5 L2 Smoke Test Report

Status: completed

Run id: ${RUN_ID}

Outputs:

- thesis_exp/outputs/exp05_low_score_loss/smoke_test/L2a_asymmetric_ordinal_lambda03_margin0
- thesis_exp/outputs/exp05_low_score_loss/smoke_test/L2b_asymmetric_ordinal_lambda05_margin0

This smoke test uses max_train_samples=${MAX_TRAIN_SAMPLES}, max_eval_samples=${MAX_EVAL_SAMPLES}, and num_train_epochs=${NUM_TRAIN_EPOCHS}.
REPORT

cat thesis_exp/outputs/exp05_low_score_loss/smoke_test/smoke_test_l2_report.md
