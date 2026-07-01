#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-llama_factory}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-thesis_exp/outputs/exp16_boundary_linking/runs/qmr/seed_42/checkpoint_best/state_dict.pt}"
SEED="${SEED:-42}"
GPU_LIST="${GPU_LIST:-6 7}"
EPOCHS="${EPOCHS:-3}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-32}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MAX_LENGTH_QUALITY="${MAX_LENGTH_QUALITY:-2048}"
MAX_LENGTH_BOUNDARY="${MAX_LENGTH_BOUNDARY:-768}"
BF16="${BF16:-auto}"
FP16="${FP16:-}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
LOG_STEPS="${LOG_STEPS:-5}"
SAVE_BEST_BY="${SAVE_BEST_BY:-dev_mae}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
RESET_RUN_DIR="${RESET_RUN_DIR:-0}"
EXP17_A1_CONFIGS="${EXP17_A1_CONFIGS:-A1F_2_frozen_probe_lr1em3_gradaccum1_epochs20 A1F_3_frozen_probe_lr3em4_gradaccum1_epochs20 A1F_4_frozen_probe_lr1em4_gradaccum1_epochs30 A1_5a_all_low_downsample76_same_neg_pool A1_5b_all_low111_same_clean_high_controls A1_1b_a0_weak_random_high_negatives}"
OUTPUT_DIR="${OUTPUT_DIR:-thesis_exp/exp17_low_score_evidence/outputs/exp17_a1_diagnostic_controls_seed42}"
A0_DIR="${A0_DIR:-thesis_exp/exp17_low_score_evidence/outputs/exp17_a0_train_hidden_failure_signals_seed42}"
D1_DIR="${D1_DIR:-thesis_exp/exp17_low_score_evidence/outputs/d1_hidden_failure_audit_seed42_dev}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ -f "${HOME}/miniconda3/bin/activate" ]]; then
  if [[ "${CONDA_ENV}" != "base" && ! -d "${HOME}/miniconda3/envs/${CONDA_ENV}" ]]; then
    echo "WARNING: conda env ${CONDA_ENV} was not found; using current shell." >&2
  elif ! source "${HOME}/miniconda3/bin/activate" "${CONDA_ENV}"; then
    echo "WARNING: could not activate conda env ${CONDA_ENV}; using current shell." >&2
  fi
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

read -r -a CONFIG_ARRAY <<< "${EXP17_A1_CONFIGS}"
read -r -a GPU_ARRAY <<< "${GPU_LIST//,/ }"
if [[ "${#CONFIG_ARRAY[@]}" -lt 1 ]]; then
  echo "ERROR: EXP17_A1_CONFIGS must contain at least one config." >&2
  exit 1
fi
if [[ "${#GPU_ARRAY[@]}" -lt 1 ]]; then
  echo "ERROR: GPU_LIST must contain at least one GPU id." >&2
  exit 1
fi
for config in "${CONFIG_ARRAY[@]}"; do
  case "${config}" in
    A1F_2_frozen_probe_lr1em3_gradaccum1_epochs20|\
    A1F_3_frozen_probe_lr3em4_gradaccum1_epochs20|\
    A1F_4_frozen_probe_lr1em4_gradaccum1_epochs30|\
    A1_5a_all_low_downsample76_same_neg_pool|\
    A1_5b_all_low111_same_clean_high_controls|\
    A1_1b_a0_weak_random_high_negatives|\
    A1_0_baseline|A1_1|A1_2|A1_3|A1_4|A1_5_all_low_aux_baseline|A1_6_random_positive_control|A1F_1_frozen_base_beta_0p10) ;;
    *)
      echo "ERROR: unknown Exp17-A1 diagnostic config '${config}'" >&2
      exit 1
      ;;
  esac
done

is_truthy() {
  [[ "$1" =~ ^(1|true|TRUE|yes|YES)$ ]]
}

if is_truthy "${FP16}" && is_truthy "${BF16}"; then
  echo "ERROR: enable only one of FP16 or BF16." >&2
  exit 1
fi

precision_from_flags() {
  local bf16_value="$1"
  local fp16_value="$2"
  if [[ "${bf16_value}" == "auto" ]]; then
    echo "auto"
  elif is_truthy "${bf16_value}"; then
    echo "bf16"
  elif is_truthy "${fp16_value}"; then
    echo "fp16"
  else
    echo "fp32"
  fi
}

PRECISION="$(precision_from_flags "${BF16}" "${FP16}")"

if [[ ! -f "${INIT_CHECKPOINT}" ]]; then
  cat >&2 <<MSG
ERROR: missing Exp16A qmr init checkpoint:
  ${INIT_CHECKPOINT}

Exp17-A1 diagnostics must start from the Exp16A qmr boundary-linking checkpoint.
Run/sync Exp16A qmr seed42 first, or set INIT_CHECKPOINT explicitly.
MSG
  exit 1
fi

cat <<CONFIG
Exp17-A1 diagnostic controls
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}
INIT_CHECKPOINT=${INIT_CHECKPOINT}
SEED=${SEED}
GPU_LIST=${GPU_LIST}
BASE_EPOCHS=${EPOCHS}
BASE_PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}
BASE_PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE}
BASE_GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}
BASE_LEARNING_RATE=${LEARNING_RATE}
MAX_LENGTH_QUALITY=${MAX_LENGTH_QUALITY}
MAX_LENGTH_BOUNDARY=${MAX_LENGTH_BOUNDARY}
BF16=${BF16}
PRECISION=${PRECISION}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING}
LOG_STEPS=${LOG_STEPS}
EXP17_A1_CONFIGS=${EXP17_A1_CONFIGS}
OUTPUT_DIR=${OUTPUT_DIR}

Note: A1F_2/3/4 override learning rate, epochs, and grad accumulation inside
train_exp17_a1_evidence_head.py for a real frozen linear-probe diagnostic.
CONFIG

run_one() {
  local config="$1"
  local gpu="$2"
  local run_dir="${OUTPUT_DIR}/runs/${config}/seed_${SEED}"
  local log_dir="${run_dir}/logs"
  local log_path="${log_dir}/train_exp17_a1_diag_${config}_seed_${SEED}_gpu${gpu}.log"
  if [[ "${SKIP_COMPLETED}" == "1" && "${RESET_RUN_DIR}" != "1" && -f "${run_dir}/metrics_dev.json" && -f "${run_dir}/evidence_eval.json" ]]; then
    echo "Skipping Exp17-A1 diagnostic ${config} seed ${SEED}: completed outputs found at ${run_dir}"
    return 0
  fi
  if [[ "${RESET_RUN_DIR}" == "1" ]]; then
    rm -rf "${run_dir}"
  fi
  mkdir -p "${log_dir}"
  local gc_args=()
  if is_truthy "${GRADIENT_CHECKPOINTING}"; then
    gc_args+=(--gradient_checkpointing)
  fi
  echo "Starting Exp17-A1 diagnostic ${config} seed ${SEED} on GPU ${gpu}; log=${log_path}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    python thesis_exp/exp17_low_score_evidence/train_exp17_a1_evidence_head.py \
      --config_name "${config}" \
      --model_name_or_path "${MODEL_NAME_OR_PATH}" \
      --init_checkpoint "${INIT_CHECKPOINT}" \
      --train_path thesis_exp/data/splits/question_seed42/train.jsonl \
      --dev_path thesis_exp/data/splits/question_seed42/dev.jsonl \
      --a0_dir "${A0_DIR}" \
      --d1_dir "${D1_DIR}" \
      --output_dir "${OUTPUT_DIR}" \
      --max_length_quality "${MAX_LENGTH_QUALITY}" \
      --max_length_boundary "${MAX_LENGTH_BOUNDARY}" \
      --batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
      --eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
      --grad_accum_steps "${GRADIENT_ACCUMULATION_STEPS}" \
      --epochs "${EPOCHS}" \
      --learning_rate "${LEARNING_RATE}" \
      --weight_decay "${WEIGHT_DECAY}" \
      --seed "${SEED}" \
      --save_best_by "${SAVE_BEST_BY}" \
      --precision "${PRECISION}" \
      --log_steps "${LOG_STEPS}" \
      --trust_remote_code \
      "${gc_args[@]}"
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
  for config in "${queue[@]}"; do
    run_one "${config}" "${gpu}"
  done
}

declare -a pids=()
for gpu_idx in "${!GPU_ARRAY[@]}"; do
  gpu="${GPU_ARRAY[$gpu_idx]}"
  queue=()
  for config_idx in "${!CONFIG_ARRAY[@]}"; do
    if (( config_idx % ${#GPU_ARRAY[@]} == gpu_idx )); then
      queue+=("${CONFIG_ARRAY[$config_idx]}")
    fi
  done
  run_gpu_queue "${gpu}" "${queue[@]}" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

python thesis_exp/exp17_low_score_evidence/collect_exp17_a1_results.py \
  --output_dir "${OUTPUT_DIR}" \
  --seed "${SEED}" \
  --configs "${CONFIG_ARRAY[@]}"

echo "Exp17-A1 diagnostic controls completed for configs: ${EXP17_A1_CONFIGS}"
