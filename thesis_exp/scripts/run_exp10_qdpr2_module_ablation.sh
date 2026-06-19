#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-llama_factory}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
QD_B1_CHECKPOINT_DIR="${QD_B1_CHECKPOINT_DIR:-thesis_exp/artifacts/exp06_question_disjoint_baselines/checkpoints/QD-B1_human_only_L1_weighted_ordinal/best}"
EXP10_GPU_LIST="${EXP10_GPU_LIST:-6,7}"
EXP10_ABLATIONS="${EXP10_ABLATIONS:-full_qdpr2 no_pair no_pair_same_pair_batches no_anchor no_mono point_only no_point_diagnostic}"
INCLUDE_DIAGNOSTIC="${INCLUDE_DIAGNOSTIC:-0}"
RESET_RUN_DIR="${RESET_RUN_DIR:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
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
EXP10_PREFLIGHT_ONLY="${EXP10_PREFLIGHT_ONLY:-0}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "${FORMAL_RUN}" != "1" && "${EXP10_PREFLIGHT_ONLY}" != "1" ]]; then
  echo "ERROR: Exp10 formal ablation script must run with FORMAL_RUN=1." >&2
  exit 1
fi

if [[ "${EXP10_PREFLIGHT_ONLY}" != "1" && ( -n "${MAX_TRAIN_SAMPLES:-}" || -n "${MAX_EVAL_SAMPLES:-}" || -n "${MAX_TRAIN_PAIRS:-}" || -n "${MAX_DEV_PAIRS:-}" ) ]]; then
  echo "ERROR: Exp10 formal runs cannot use max sample or max pair limits." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${HOME}/miniconda3/bin/activate" ]]; then
  if [[ "${CONDA_ENV}" != "base" && ! -d "${HOME}/miniconda3/envs/${CONDA_ENV}" ]]; then
    echo "WARNING: conda env ${CONDA_ENV} was not found; using current shell." >&2
  elif ! source "${HOME}/miniconda3/bin/activate" "${CONDA_ENV}"; then
    echo "WARNING: could not activate conda env ${CONDA_ENV}; using current shell." >&2
  fi
else
  echo "WARNING: ${HOME}/miniconda3/bin/activate was not found; using current shell." >&2
fi

if [[ "${INCLUDE_DIAGNOSTIC}" == "1" && "${EXP10_ABLATIONS}" != *"no_point_diagnostic"* ]]; then
  EXP10_ABLATIONS="${EXP10_ABLATIONS} no_point_diagnostic"
fi

export MODEL_NAME_OR_PATH
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF
export FORMAL_RUN
export REQUIRE_CUDA

IFS=',' read -r -a GPUS <<< "${EXP10_GPU_LIST}"
read -r -a ABLATIONS <<< "${EXP10_ABLATIONS}"
if [[ "${#GPUS[@]}" -lt 1 ]]; then
  echo "ERROR: EXP10_GPU_LIST must contain at least one GPU id." >&2
  exit 1
fi

EFFECTIVE_BATCH_SIZE=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
if [[ "${EFFECTIVE_BATCH_SIZE}" != "128" ]]; then
  echo "ERROR: effective batch size must remain 128, got ${EFFECTIVE_BATCH_SIZE}." >&2
  exit 1
fi

cat <<CONFIG
Exp10 QD-PR2 module ablation
FORMAL_RUN=${FORMAL_RUN}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}
QD_B1_CHECKPOINT_DIR=${QD_B1_CHECKPOINT_DIR}
EXP10_GPU_LIST=${EXP10_GPU_LIST}
EXP10_ABLATIONS=${EXP10_ABLATIONS}
INCLUDE_DIAGNOSTIC=${INCLUDE_DIAGNOSTIC}
RESET_RUN_DIR=${RESET_RUN_DIR}
SKIP_COMPLETED=${SKIP_COMPLETED}
EXP10_PREFLIGHT_ONLY=${EXP10_PREFLIGHT_ONLY}
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
CONFIG

python -m thesis_exp.src.edujudge.exp10_qdpr2_module_ablation.sanity_check_exp10_setup

if [[ "${EXP10_PREFLIGHT_ONLY}" == "1" ]]; then
  python -m thesis_exp.src.edujudge.exp10_qdpr2_module_ablation.preflight_exp10_ablation_matrix
  python -m thesis_exp.src.edujudge.exp10_qdpr2_module_ablation.collect_exp10_results
  python -m thesis_exp.src.edujudge.exp10_qdpr2_module_ablation.readability_check_exp10
  cat thesis_exp/outputs/exp10_qdpr2_module_ablation/reports/exp10_preflight_ablation_matrix.md
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

run_one() {
  local ablation="$1"
  local gpu="$2"
  local config_path="thesis_exp/configs/exp10_qdpr2_module_ablation/exp10_${ablation}.yaml"
  local output_dir="thesis_exp/outputs/exp10_qdpr2_module_ablation/runs/${ablation}"
  local checkpoint_dir="thesis_exp/artifacts/exp10_qdpr2_module_ablation/checkpoints/${ablation}"
  local log_dir="thesis_exp/outputs/exp10_qdpr2_module_ablation/logs"
  local run_tag="exp10_${ablation}_$(date +%Y%m%d_%H%M%S)"
  local log_path="${log_dir}/train_EXP10_${ablation}_${run_tag}_gpu${gpu}.log"
  mkdir -p "${log_dir}"
  if [[ ! -f "${config_path}" ]]; then
    echo "Missing config: ${config_path}" >&2
    return 1
  fi
  if [[ "${SKIP_COMPLETED}" == "1" && "${RESET_RUN_DIR}" != "1" && -f "${output_dir}/run_metadata.json" ]]; then
    if grep -q '"status": "completed"' "${output_dir}/run_metadata.json"; then
      echo "Skipping ${ablation}: existing completed run found at ${output_dir}"
      return 0
    fi
  fi
  if [[ "${RESET_RUN_DIR}" == "1" ]]; then
    rm -rf "${output_dir}" "${checkpoint_dir}"
  fi
  echo "Starting ${ablation} on GPU ${gpu}; log=${log_path}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    echo "Ablation=${ablation}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "CONFIG_PATH=${config_path}"
    echo "OUTPUT_DIR=${output_dir}"
    echo "CHECKPOINT_DIR=${checkpoint_dir}"
    echo "RUN_TAG=${run_tag}"
    if [[ "${EXP10_PREFLIGHT_ONLY}" == "1" ]]; then
      python -m thesis_exp.src.edujudge.exp09_pairwise_ordinal.train_qdpr2_anchored_pairwise \
        --config_path "${config_path}" \
        --model_name_or_path "${MODEL_NAME_OR_PATH}" \
        --qd_b1_checkpoint_dir "${QD_B1_CHECKPOINT_DIR}" \
        --output_dir "${output_dir}" \
        --checkpoint_output_dir "${checkpoint_dir}" \
        --preflight_only
      exit 0
    fi
    python -m thesis_exp.src.edujudge.exp09_pairwise_ordinal.train_qdpr2_anchored_pairwise \
      --config_path "${config_path}" \
      --model_name_or_path "${MODEL_NAME_OR_PATH}" \
      --qd_b1_checkpoint_dir "${QD_B1_CHECKPOINT_DIR}" \
      --output_dir "${output_dir}" \
      --checkpoint_output_dir "${checkpoint_dir}" \
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
  for ablation in "${queue[@]}"; do
    run_one "${ablation}" "${gpu}"
  done
}

pids=()
names=()
for gpu_index in "${!GPUS[@]}"; do
  gpu="${GPUS[gpu_index]}"
  queue=()
  for ((task_index=gpu_index; task_index<${#ABLATIONS[@]}; task_index+=${#GPUS[@]})); do
    queue+=("${ABLATIONS[task_index]}")
  done
  run_gpu_queue "${gpu}" "${queue[@]}" &
  pids+=("$!")
  names+=("gpu_${gpu}")
done

for idx in "${!pids[@]}"; do
  if ! wait "${pids[idx]}"; then
    echo "Exp10 GPU queue failed: ${names[idx]}" >&2
    exit 1
  fi
done

python -m thesis_exp.src.edujudge.exp10_qdpr2_module_ablation.collect_exp10_results
python -m thesis_exp.src.edujudge.exp10_qdpr2_module_ablation.readability_check_exp10
cat thesis_exp/outputs/exp10_qdpr2_module_ablation/reports/exp10_ablation_summary.md
