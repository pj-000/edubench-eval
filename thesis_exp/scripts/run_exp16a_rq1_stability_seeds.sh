#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-llama_factory}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
SEEDS="${SEEDS:-43 44}"
GPU_LIST="${GPU_LIST:-6 7}"
EXP16A_VARIANTS="${EXP16A_VARIANTS:-qmr metric_rubric}"
EPOCHS="${EPOCHS:-3}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-32}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MAX_LENGTH_QUALITY="${MAX_LENGTH_QUALITY:-2048}"
MAX_LENGTH_BOUNDARY="${MAX_LENGTH_BOUNDARY:-768}"
BF16="${BF16:-auto}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
SAVE_BEST_BY="${SAVE_BEST_BY:-dev_mae}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
RESET_RUN_DIR="${RESET_RUN_DIR:-0}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'USAGE'
Run Exp16A RQ1 stability seeds.

Defaults:
  SEEDS="43 44"
  EXP16A_VARIANTS="qmr metric_rubric"
  GPU_LIST="6 7"

Outputs:
  thesis_exp/outputs/exp16_boundary_linking/scout_seed43/{qmr,metric_rubric}/
  thesis_exp/outputs/exp16_boundary_linking/scout_seed44/{qmr,metric_rubric}/

Environment overrides are supported for SEEDS, GPU_LIST, EPOCHS,
PER_DEVICE_TRAIN_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS, LEARNING_RATE,
BF16, and GRADIENT_CHECKPOINTING.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -f "${HOME}/miniconda3/bin/activate" ]]; then
  if [[ "${CONDA_ENV}" != "base" && ! -d "${HOME}/miniconda3/envs/${CONDA_ENV}" ]]; then
    echo "WARNING: conda env ${CONDA_ENV} was not found; using current shell." >&2
  elif ! source "${HOME}/miniconda3/bin/activate" "${CONDA_ENV}"; then
    echo "WARNING: could not activate conda env ${CONDA_ENV}; using current shell." >&2
  fi
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF

read -r -a SEED_ARRAY <<< "${SEEDS}"
read -r -a GPU_ARRAY <<< "${GPU_LIST//,/ }"
read -r -a VARIANT_ARRAY <<< "${EXP16A_VARIANTS}"
if [[ "${#SEED_ARRAY[@]}" -lt 1 || "${#GPU_ARRAY[@]}" -lt 1 || "${#VARIANT_ARRAY[@]}" -lt 1 ]]; then
  echo "ERROR: SEEDS, GPU_LIST, and EXP16A_VARIANTS must be non-empty." >&2
  exit 1
fi
for variant in "${VARIANT_ARRAY[@]}"; do
  case "${variant}" in
    qmr|metric_rubric) ;;
    *)
      echo "ERROR: RQ1 stability only supports qmr and metric_rubric, got ${variant}" >&2
      exit 1
      ;;
  esac
done

EFFECTIVE_BATCH_SIZE=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
if [[ "${EFFECTIVE_BATCH_SIZE}" != "128" ]]; then
  echo "ERROR: Exp16A stability effective batch size must remain 128, got ${EFFECTIVE_BATCH_SIZE}." >&2
  exit 1
fi

precision_args=()
case "${BF16}" in
  1|true|TRUE|True|auto|AUTO|Auto|bf16|BF16)
    precision_args+=(--bf16)
    ;;
esac
gc_args=()
if [[ "${GRADIENT_CHECKPOINTING}" == "1" || "${GRADIENT_CHECKPOINTING}" == "true" || "${GRADIENT_CHECKPOINTING}" == "TRUE" || "${GRADIENT_CHECKPOINTING}" == "True" ]]; then
  gc_args+=(--gradient_checkpointing)
fi

cat <<CONFIG
Exp16A RQ1 stability seeds
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}
SEEDS=${SEEDS}
GPU_LIST=${GPU_LIST}
EXP16A_VARIANTS=${EXP16A_VARIANTS}
EPOCHS=${EPOCHS}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}
effective batch size=${EFFECTIVE_BATCH_SIZE}
BF16=${BF16}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING}
SAVE_BEST_BY=${SAVE_BEST_BY}
SKIP_COMPLETED=${SKIP_COMPLETED}
RESET_RUN_DIR=${RESET_RUN_DIR}
CONFIG

run_one() {
  local variant="$1"
  local seed="$2"
  local gpu="$3"
  local output_dir="thesis_exp/outputs/exp16_boundary_linking/scout_seed${seed}/${variant}"
  local log_dir="${output_dir}/logs"
  local log_path="${log_dir}/train_exp16a_rq1_${variant}_seed_${seed}_gpu${gpu}.log"
  if [[ "${SKIP_COMPLETED}" == "1" && "${RESET_RUN_DIR}" != "1" && -f "${output_dir}/metrics_dev.json" && -f "${output_dir}/predictions_dev.jsonl" ]]; then
    echo "Skipping Exp16A RQ1 ${variant} seed ${seed}: completed outputs found at ${output_dir}"
    return 0
  fi
  if [[ "${RESET_RUN_DIR}" == "1" ]]; then
    rm -rf "${output_dir}"
  fi
  mkdir -p "${log_dir}"
  echo "Starting Exp16A RQ1 ${variant} seed ${seed} on GPU ${gpu}; log=${log_path}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    python -m thesis_exp.src.edujudge.exp16_boundary_linking.train_boundary_linking \
      --model_name_or_path "${MODEL_NAME_OR_PATH}" \
      --train_path thesis_exp/data/splits/question_seed42/train.jsonl \
      --dev_path thesis_exp/data/splits/question_seed42/dev.jsonl \
      --test_path thesis_exp/data/splits/question_seed42/test.jsonl \
      --output_dir "${output_dir}" \
      --variant "${variant}" \
      --max_length_quality "${MAX_LENGTH_QUALITY}" \
      --max_length_boundary "${MAX_LENGTH_BOUNDARY}" \
      --batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
      --grad_accum_steps "${GRADIENT_ACCUMULATION_STEPS}" \
      --epochs "${EPOCHS}" \
      --learning_rate "${LEARNING_RATE}" \
      --weight_decay "${WEIGHT_DECAY}" \
      --seed "${seed}" \
      --freeze_encoder false \
      --eval_every_epoch \
      --save_best_by "${SAVE_BEST_BY}" \
      --trust_remote_code \
      "${gc_args[@]}" \
      "${precision_args[@]}"
  ) 2>&1 | tee "${log_path}"
}

jobs=()
for seed in "${SEED_ARRAY[@]}"; do
  for variant in "${VARIANT_ARRAY[@]}"; do
    jobs+=("${variant}:${seed}")
  done
done

run_gpu_queue() {
  local gpu="$1"
  shift
  local queue=("$@")
  if [[ "${#queue[@]}" -eq 0 ]]; then
    echo "GPU ${gpu} queue is empty."
    return 0
  fi
  echo "GPU ${gpu} queue: ${queue[*]}"
  for item in "${queue[@]}"; do
    run_one "${item%%:*}" "${item##*:}" "${gpu}"
  done
}

pids=()
for gpu_idx in "${!GPU_ARRAY[@]}"; do
  gpu="${GPU_ARRAY[$gpu_idx]}"
  queue=()
  for job_idx in "${!jobs[@]}"; do
    if (( job_idx % ${#GPU_ARRAY[@]} == gpu_idx )); then
      queue+=("${jobs[$job_idx]}")
    fi
  done
  run_gpu_queue "${gpu}" "${queue[@]}" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

python -m thesis_exp.src.edujudge.exp16_boundary_linking.summarize_stability
