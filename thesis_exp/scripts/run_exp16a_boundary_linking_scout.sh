#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-llama_factory}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
SEED="${SEED:-42}"
GPU_LIST="${GPU_LIST:-6 7}"
EPOCHS="${EPOCHS:-3}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-${BATCH_SIZE:-4}}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-${GRAD_ACCUM_STEPS:-32}}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MAX_LENGTH_QUALITY="${MAX_LENGTH_QUALITY:-2048}"
MAX_LENGTH_BOUNDARY="${MAX_LENGTH_BOUNDARY:-768}"
BF16="${BF16:-auto}"
FP16="${FP16:-}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
AUTO_RETRY_OOM="${AUTO_RETRY_OOM:-1}"
OOM_FALLBACK_BATCH_SIZE="${OOM_FALLBACK_BATCH_SIZE:-2}"
OOM_FALLBACK_GRAD_ACCUM_STEPS="${OOM_FALLBACK_GRAD_ACCUM_STEPS:-64}"
OOM_FALLBACK_GRADIENT_CHECKPOINTING="${OOM_FALLBACK_GRADIENT_CHECKPOINTING:-1}"
LOG_STEPS="${LOG_STEPS:-5}"
SAVE_BEST_BY="${SAVE_BEST_BY:-dev_mae}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
RESET_RUN_DIR="${RESET_RUN_DIR:-0}"
EXP16A_VARIANTS="${EXP16A_VARIANTS:-global metric_rubric qmr qmr_meta}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ "$#" -gt 0 ]]; then
  EXP16A_VARIANTS="$*"
fi

if [[ -f "${HOME}/miniconda3/bin/activate" ]]; then
  if [[ "${CONDA_ENV}" != "base" && ! -d "${HOME}/miniconda3/envs/${CONDA_ENV}" ]]; then
    echo "WARNING: conda env ${CONDA_ENV} was not found; using current shell." >&2
  elif ! source "${HOME}/miniconda3/bin/activate" "${CONDA_ENV}"; then
    echo "WARNING: could not activate conda env ${CONDA_ENV}; using current shell." >&2
  fi
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

read -r -a VARIANT_ARRAY <<< "${EXP16A_VARIANTS}"
read -r -a GPU_ARRAY <<< "${GPU_LIST//,/ }"
if [[ "${#VARIANT_ARRAY[@]}" -lt 1 ]]; then
  echo "ERROR: EXP16A_VARIANTS must contain at least one variant." >&2
  exit 1
fi
if [[ "${#GPU_ARRAY[@]}" -lt 1 ]]; then
  echo "ERROR: GPU_LIST must contain at least one GPU id." >&2
  exit 1
fi
for variant in "${VARIANT_ARRAY[@]}"; do
  case "${variant}" in
    global|metric_rubric|qmr|qmr_meta) ;;
    *)
      echo "ERROR: unknown Exp16A variant '${variant}'" >&2
      echo "Allowed: global metric_rubric qmr qmr_meta" >&2
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
EFFECTIVE_BATCH_SIZE=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
FALLBACK_EFFECTIVE_BATCH_SIZE=$((OOM_FALLBACK_BATCH_SIZE * OOM_FALLBACK_GRAD_ACCUM_STEPS))
if [[ "${EFFECTIVE_BATCH_SIZE}" != "128" ]]; then
  echo "ERROR: Exp16A effective batch size must remain 128, got ${EFFECTIVE_BATCH_SIZE}." >&2
  exit 1
fi
if [[ "${FALLBACK_EFFECTIVE_BATCH_SIZE}" != "128" ]]; then
  echo "ERROR: Exp16A OOM fallback effective batch size must remain 128, got ${FALLBACK_EFFECTIVE_BATCH_SIZE}." >&2
  exit 1
fi

cat <<CONFIG
Exp16A boundary linking scout
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}
SEED=${SEED}
GPU_LIST=${GPU_LIST}
EPOCHS=${EPOCHS}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}
effective batch size=${EFFECTIVE_BATCH_SIZE}
MAX_LENGTH_QUALITY=${MAX_LENGTH_QUALITY}
MAX_LENGTH_BOUNDARY=${MAX_LENGTH_BOUNDARY}
BF16=${BF16}
PRECISION=${PRECISION}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING}
AUTO_RETRY_OOM=${AUTO_RETRY_OOM}
OOM_FALLBACK_BATCH_SIZE=${OOM_FALLBACK_BATCH_SIZE}
OOM_FALLBACK_GRAD_ACCUM_STEPS=${OOM_FALLBACK_GRAD_ACCUM_STEPS}
OOM fallback effective batch size=${FALLBACK_EFFECTIVE_BATCH_SIZE}
OOM_FALLBACK_GRADIENT_CHECKPOINTING=${OOM_FALLBACK_GRADIENT_CHECKPOINTING}
LOG_STEPS=${LOG_STEPS}
EXP16A_VARIANTS=${EXP16A_VARIANTS}
SKIP_COMPLETED=${SKIP_COMPLETED}
RESET_RUN_DIR=${RESET_RUN_DIR}
CONFIG

run_train_attempt() {
  local variant="$1"
  local output_dir="$2"
  local batch_size="$3"
  local grad_accum_steps="$4"
  local gradient_checkpointing="$5"
  local attempt_name="$6"
  local attempt_args=()
  if is_truthy "${gradient_checkpointing}"; then
    attempt_args+=(--gradient_checkpointing)
  fi
  echo "Exp16A ${variant} attempt=${attempt_name} batch_size=${batch_size} grad_accum=${grad_accum_steps} gradient_checkpointing=${gradient_checkpointing}"
  python -m thesis_exp.src.edujudge.exp16_boundary_linking.train_boundary_linking \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --train_path thesis_exp/data/splits/question_seed42/train.jsonl \
    --dev_path thesis_exp/data/splits/question_seed42/dev.jsonl \
    --test_path thesis_exp/data/splits/question_seed42/test.jsonl \
    --output_dir "${output_dir}" \
    --variant "${variant}" \
    --max_length_quality "${MAX_LENGTH_QUALITY}" \
    --max_length_boundary "${MAX_LENGTH_BOUNDARY}" \
    --batch_size "${batch_size}" \
    --eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
    --grad_accum_steps "${grad_accum_steps}" \
    --epochs "${EPOCHS}" \
    --learning_rate "${LEARNING_RATE}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --seed "${SEED}" \
    --freeze_encoder false \
    --eval_every_epoch \
    --save_best_by "${SAVE_BEST_BY}" \
    --precision "${PRECISION}" \
    --log_steps "${LOG_STEPS}" \
    --trust_remote_code \
    "${attempt_args[@]}"
}

run_one() {
  local variant="$1"
  local gpu="$2"
  local output_dir="thesis_exp/outputs/exp16_boundary_linking/runs/${variant}/seed_${SEED}"
  local log_dir="${output_dir}/logs"
  local log_path="${log_dir}/train_exp16a_${variant}_seed_${SEED}_gpu${gpu}.log"
  if [[ "${SKIP_COMPLETED}" == "1" && "${RESET_RUN_DIR}" != "1" && -f "${output_dir}/metrics_dev.json" && -f "${output_dir}/predictions_dev.jsonl" ]]; then
    echo "Skipping Exp16A ${variant} seed ${SEED}: completed outputs found at ${output_dir}"
    return 0
  fi
  if [[ "${RESET_RUN_DIR}" == "1" ]]; then
    rm -rf "${output_dir}"
  fi
  mkdir -p "${log_dir}"
  echo "Starting Exp16A ${variant} seed ${SEED} on GPU ${gpu}; log=${log_path}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    set +e
    run_train_attempt "${variant}" "${output_dir}" "${PER_DEVICE_TRAIN_BATCH_SIZE}" "${GRADIENT_ACCUMULATION_STEPS}" "${GRADIENT_CHECKPOINTING}" "standard"
    status="$?"
    set -e
    if [[ "${status}" -ne 0 ]] && is_truthy "${AUTO_RETRY_OOM}"; then
      echo "Exp16A ${variant} standard attempt failed with exit ${status}; retrying with OOM fallback profile."
      rm -rf "${output_dir}/checkpoint_best" \
        "${output_dir}/metrics_dev.json" \
        "${output_dir}/metrics_test.json" \
        "${output_dir}/predictions_dev.jsonl" \
        "${output_dir}/predictions_test.jsonl" \
        "${output_dir}/threshold_stats_dev.json" \
        "${output_dir}/threshold_stats_test.json" \
        "${output_dir}/threshold_by_metric_dev.csv" \
        "${output_dir}/threshold_by_metric_test.csv" \
        "${output_dir}/training_history.json" \
        "${output_dir}/config.json"
      set +e
      run_train_attempt "${variant}" "${output_dir}" "${OOM_FALLBACK_BATCH_SIZE}" "${OOM_FALLBACK_GRAD_ACCUM_STEPS}" "${OOM_FALLBACK_GRADIENT_CHECKPOINTING}" "oom_fallback"
      status="$?"
      set -e
    fi
    if [[ "${status}" -ne 0 ]]; then
      exit "${status}"
    fi

    python -m thesis_exp.src.edujudge.exp16_boundary_linking.analyze_boundaries \
      --predictions_path "${output_dir}/predictions_dev.jsonl" \
      --output_dir "${output_dir}/analysis_dev"
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
  for variant in "${queue[@]}"; do
    run_one "${variant}" "${gpu}"
  done
}

declare -a pids=()
for gpu_idx in "${!GPU_ARRAY[@]}"; do
  gpu="${GPU_ARRAY[$gpu_idx]}"
  queue=()
  for variant_idx in "${!VARIANT_ARRAY[@]}"; do
    if (( variant_idx % ${#GPU_ARRAY[@]} == gpu_idx )); then
      queue+=("${VARIANT_ARRAY[$variant_idx]}")
    fi
  done
  run_gpu_queue "${gpu}" "${queue[@]}" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

echo "Exp16A scout completed for variants: ${EXP16A_VARIANTS}"
