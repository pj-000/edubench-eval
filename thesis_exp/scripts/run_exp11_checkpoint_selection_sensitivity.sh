#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-llama_factory}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
QD_B1_CHECKPOINT_DIR="${QD_B1_CHECKPOINT_DIR:-thesis_exp/artifacts/exp06_question_disjoint_baselines/checkpoints/QD-B1_human_only_L1_weighted_ordinal/best}"
CONFIG_PATH="${CONFIG_PATH:-thesis_exp/configs/exp11_checkpoint_selection_sensitivity/exp11_full_qdpr2_checkpoint_selection.yaml}"
SEEDS="${SEEDS:-42 43 44}"
GPU_LIST="${GPU_LIST:-6 7}"
EPOCHS="${EPOCHS:-3}"
FORMAL_RUN="${FORMAL_RUN:-1}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
SMOKE="${SMOKE:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
RESET_RUN_DIR="${RESET_RUN_DIR:-0}"
EXP11_PREFLIGHT_ONLY="${EXP11_PREFLIGHT_ONLY:-0}"
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
GAMMA="${GAMMA:-4.0}"
MAE_GUARD_DELTA="${MAE_GUARD_DELTA:-0.005}"
MONO_BETA="${MONO_BETA:-0.2}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${SMOKE}" == "1" ]]; then
  FORMAL_RUN="0"
  REQUIRE_CUDA="${REQUIRE_CUDA:-0}"
fi

if [[ "${FORMAL_RUN}" != "1" && "${EXP11_PREFLIGHT_ONLY}" != "1" && "${SMOKE}" != "1" ]]; then
  echo "ERROR: Exp11 formal script must run with FORMAL_RUN=1 unless SMOKE=1 or EXP11_PREFLIGHT_ONLY=1." >&2
  exit 1
fi

if [[ "${EXP11_PREFLIGHT_ONLY}" != "1" && "${SMOKE}" != "1" && ( -n "${MAX_TRAIN_SAMPLES:-}" || -n "${MAX_EVAL_SAMPLES:-}" || -n "${MAX_TRAIN_PAIRS:-}" || -n "${MAX_DEV_PAIRS:-}" ) ]]; then
  echo "ERROR: Exp11 formal runs cannot use max sample or max pair limits." >&2
  exit 1
fi

if [[ -f "${HOME}/miniconda3/bin/activate" ]]; then
  if [[ "${CONDA_ENV}" != "base" && ! -d "${HOME}/miniconda3/envs/${CONDA_ENV}" ]]; then
    echo "WARNING: conda env ${CONDA_ENV} was not found; using current shell." >&2
  elif ! source "${HOME}/miniconda3/bin/activate" "${CONDA_ENV}"; then
    echo "WARNING: could not activate conda env ${CONDA_ENV}; using current shell." >&2
  fi
else
  echo "WARNING: ${HOME}/miniconda3/bin/activate was not found; using current shell." >&2
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF
export FORMAL_RUN
export REQUIRE_CUDA

read -r -a SEED_ARRAY <<< "${SEEDS}"
read -r -a GPU_ARRAY <<< "${GPU_LIST//,/ }"
if [[ "${#SEED_ARRAY[@]}" -lt 1 ]]; then
  echo "ERROR: SEEDS must contain at least one seed." >&2
  exit 1
fi
if [[ "${#GPU_ARRAY[@]}" -lt 1 ]]; then
  echo "ERROR: GPU_LIST must contain at least one GPU id." >&2
  exit 1
fi

EFFECTIVE_BATCH_SIZE=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
if [[ "${SMOKE}" != "1" && "${EFFECTIVE_BATCH_SIZE}" != "128" ]]; then
  echo "ERROR: Exp11 formal effective batch size must remain 128, got ${EFFECTIVE_BATCH_SIZE}." >&2
  exit 1
fi

cat <<CONFIG
Exp11 checkpoint selection sensitivity
FORMAL_RUN=${FORMAL_RUN}
SMOKE=${SMOKE}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}
QD_B1_CHECKPOINT_DIR=${QD_B1_CHECKPOINT_DIR}
CONFIG_PATH=${CONFIG_PATH}
SEEDS=${SEEDS}
GPU_LIST=${GPU_LIST}
EPOCHS=${EPOCHS}
SKIP_COMPLETED=${SKIP_COMPLETED}
RESET_RUN_DIR=${RESET_RUN_DIR}
EXP11_PREFLIGHT_ONLY=${EXP11_PREFLIGHT_ONLY}
GAMMA=${GAMMA}
MAE_GUARD_DELTA=${MAE_GUARD_DELTA}
MONO_BETA=${MONO_BETA}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}
effective batch size=${EFFECTIVE_BATCH_SIZE}
CONFIG

if [[ "${EXP11_PREFLIGHT_ONLY}" == "1" ]]; then
  python -m thesis_exp.src.edujudge.exp11_checkpoint_selection_sensitivity.preflight_exp11 \
    --config_path "${CONFIG_PATH}" \
    --seeds "${SEEDS}" \
    --gpu_list "${GPU_LIST}" \
    --epochs "${EPOCHS}" \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --qd_b1_checkpoint_dir "${QD_B1_CHECKPOINT_DIR}" \
    --gamma "${GAMMA}" \
    --delta "${MAE_GUARD_DELTA}" \
    --beta "${MONO_BETA}"
  cat thesis_exp/outputs/exp11_checkpoint_selection_sensitivity/reports/exp11_preflight_report.md
  exit 0
fi

if [[ ! -f "${QD_B1_CHECKPOINT_DIR}/state_dict.pt" ]]; then
  echo "BLOCKED_MISSING_QDB1_CHECKPOINT: ${QD_B1_CHECKPOINT_DIR}" >&2
  exit 1
fi

progress_args=()
if [[ "${NO_PROGRESS_BAR}" == "1" ]]; then
  progress_args+=(--no_progress_bar)
fi

gc_args=()
if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then
  gc_args+=(--gradient_checkpointing)
fi

smoke_args=()
seed_prefix="seed"
if [[ "${SMOKE}" == "1" ]]; then
  smoke_args+=(--smoke)
  seed_prefix="smoke_seed"
fi

run_one_seed() {
  local seed="$1"
  local gpu="$2"
  local base_dir="thesis_exp/runs/exp11_checkpoint_selection_sensitivity/${seed_prefix}_${seed}"
  local output_dir="${base_dir}/run"
  local checkpoint_dir="${base_dir}/checkpoints"
  local log_dir="${base_dir}/logs"
  local log_path="${log_dir}/train_exp11_seed_${seed}_gpu${gpu}.log"
  mkdir -p "${log_dir}"
  if [[ "${SKIP_COMPLETED}" == "1" && "${RESET_RUN_DIR}" != "1" && -f "${output_dir}/run_metadata.json" ]]; then
    if grep -q '"status": "completed"' "${output_dir}/run_metadata.json"; then
      echo "Skipping seed ${seed}: completed run found at ${output_dir}"
      return 0
    fi
  fi
  if [[ "${RESET_RUN_DIR}" == "1" ]]; then
    rm -rf "${base_dir}"
    mkdir -p "${log_dir}"
  fi
  echo "Starting Exp11 seed ${seed} on GPU ${gpu}; log=${log_path}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    echo "seed=${seed}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "OUTPUT_DIR=${output_dir}"
    echo "CHECKPOINT_DIR=${checkpoint_dir}"
    python -m thesis_exp.src.edujudge.exp09_pairwise_ordinal.train_qdpr2_anchored_pairwise \
      --config_path "${CONFIG_PATH}" \
      --model_name_or_path "${MODEL_NAME_OR_PATH}" \
      --qd_b1_checkpoint_dir "${QD_B1_CHECKPOINT_DIR}" \
      --output_dir "${output_dir}" \
      --checkpoint_output_dir "${checkpoint_dir}" \
      --max_length "${MAX_LENGTH}" \
      --num_train_epochs "${EPOCHS}" \
      --learning_rate "${LEARNING_RATE}" \
      --weight_decay "${WEIGHT_DECAY}" \
      --warmup_ratio "${WARMUP_RATIO}" \
      --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
      --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
      --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
      --seed "${seed}" \
      --bf16 "${BF16}" \
      --log_steps "${LOG_STEPS}" \
      --save_each_epoch \
      --eval_each_epoch \
      --keep_epoch_checkpoints_local \
      --selection_rules_enabled \
      --soft_risk_gamma "${GAMMA}" \
      "${progress_args[@]}" \
      "${gc_args[@]}" \
      "${smoke_args[@]}"
  ) 2>&1 | tee "${log_path}"
}

run_gpu_queue() {
  local gpu="$1"
  shift
  local queue=("$@")
  if [[ "${#queue[@]}" -eq 0 ]]; then
    echo "GPU ${gpu} queue is empty."
    return 0
  fi
  echo "GPU ${gpu} queue: ${queue[*]}"
  for seed in "${queue[@]}"; do
    run_one_seed "${seed}" "${gpu}"
  done
}

pids=()
names=()
for gpu_index in "${!GPU_ARRAY[@]}"; do
  gpu="${GPU_ARRAY[gpu_index]}"
  queue=()
  for ((seed_index=gpu_index; seed_index<${#SEED_ARRAY[@]}; seed_index+=${#GPU_ARRAY[@]})); do
    queue+=("${SEED_ARRAY[seed_index]}")
  done
  run_gpu_queue "${gpu}" "${queue[@]}" &
  pids+=("$!")
  names+=("gpu_${gpu}")
done

for idx in "${!pids[@]}"; do
  if ! wait "${pids[idx]}"; then
    echo "Exp11 GPU queue failed: ${names[idx]}" >&2
    exit 1
  fi
done

if [[ "${SMOKE}" == "1" ]]; then
  mkdir -p thesis_exp/outputs/exp11_checkpoint_selection_sensitivity/reports
  {
    echo "# Exp11 Smoke Check"
    echo
    echo "Status: \`PASS\`"
    echo
    echo "Smoke run artifacts were written under \`thesis_exp/runs/exp11_checkpoint_selection_sensitivity/smoke_seed_*\`."
  } > thesis_exp/outputs/exp11_checkpoint_selection_sensitivity/reports/exp11_smoke_check.md
  cat thesis_exp/outputs/exp11_checkpoint_selection_sensitivity/reports/exp11_smoke_check.md
  exit 0
fi

python -m thesis_exp.src.edujudge.exp11_checkpoint_selection_sensitivity.collect_exp11_results \
  --delta "${MAE_GUARD_DELTA}" \
  --beta "${MONO_BETA}" \
  --gamma "${GAMMA}"
python -m thesis_exp.src.edujudge.exp11_checkpoint_selection_sensitivity.readability_check_exp11
cat thesis_exp/outputs/exp11_checkpoint_selection_sensitivity/reports/exp11_checkpoint_selection_sensitivity_report.md
