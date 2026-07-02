#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-llama_factory}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-thesis_exp/outputs/exp16_boundary_linking/runs/qmr/seed_42/checkpoint_best/state_dict.pt}"
SEED="${SEED:-42}"
GPU_LIST="${GPU_LIST:-6 7}"
EPOCHS="${EPOCHS:-3}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
PAIR_BATCH_SIZE="${PAIR_BATCH_SIZE:-4}"
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
RUN_C0_D1_DIAG="${RUN_C0_D1_DIAG:-0}"
C0B_CONFIGS="${C0B_CONFIGS:-C0b_0_init_no_train_eval C0b_1_all_pairs_raw_s_gamma0p01_freeze_boundary C0b_2_all_pairs_raw_s_gamma0p02_freeze_boundary C0b_3_evidence_pairs_raw_s_gamma0p02_freeze_boundary C0b_4_random_pair_raw_s_gamma0p02_freeze_boundary C0b_5_all_pairs_g3detach_gamma0p01 C0b_6_all_pairs_g3detach_gamma0p02 C0b_7_random_matched_g3detach_gamma0p02}"
OUTPUT_DIR="${OUTPUT_DIR:-thesis_exp/exp17_low_score_evidence/outputs/exp17_c0b_pairwise_diagnostics_seed42}"
A0_DIR="${A0_DIR:-thesis_exp/exp17_low_score_evidence/outputs/exp17_a0_train_hidden_failure_signals_seed42}"
D1_DIR="${D1_DIR:-thesis_exp/exp17_low_score_evidence/outputs/d1_hidden_failure_audit_seed42_dev}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ -f "${HOME}/miniconda3/bin/activate" ]]; then
  if [[ "${CONDA_ENV}" != "base" && -d "${HOME}/miniconda3/envs/${CONDA_ENV}" ]]; then
    source "${HOME}/miniconda3/bin/activate" "${CONDA_ENV}"
  fi
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

read -r -a CONFIG_ARRAY <<< "${C0B_CONFIGS}"
read -r -a GPU_ARRAY <<< "${GPU_LIST//,/ }"
if [[ "${#GPU_ARRAY[@]}" -lt 1 ]]; then
  echo "ERROR: GPU_LIST must contain at least one GPU id." >&2
  exit 1
fi

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
  echo "ERROR: missing Exp16A qmr init checkpoint: ${INIT_CHECKPOINT}" >&2
  exit 1
fi

cat <<CONFIG
Exp17-C0b pairwise diagnostics
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}
INIT_CHECKPOINT=${INIT_CHECKPOINT}
SEED=${SEED}
GPU_LIST=${GPU_LIST}
EPOCHS=${EPOCHS}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}
PAIR_BATCH_SIZE=${PAIR_BATCH_SIZE}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}
LEARNING_RATE=${LEARNING_RATE}
PRECISION=${PRECISION}
C0B_CONFIGS=${C0B_CONFIGS}
OUTPUT_DIR=${OUTPUT_DIR}
CONFIG

run_one() {
  local config="$1"
  local gpu="$2"
  local run_dir="${OUTPUT_DIR}/runs/${config}/seed_${SEED}"
  local log_dir="${run_dir}/logs"
  local log_path="${log_dir}/train_exp17_c0b_${config}_seed_${SEED}_gpu${gpu}.log"
  if [[ "${SKIP_COMPLETED}" == "1" && "${RESET_RUN_DIR}" != "1" && -f "${run_dir}/metrics_dev.json" && -f "${run_dir}/diagnostic_predictions_dev.csv" ]]; then
    echo "Skipping Exp17-C0b ${config} seed ${SEED}: completed outputs found at ${run_dir}"
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
  echo "Starting Exp17-C0b ${config} seed ${SEED} on GPU ${gpu}; log=${log_path}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    python thesis_exp/exp17_low_score_evidence/train_exp17_c0_pairwise_separation.py \
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
      --pair_batch_size "${PAIR_BATCH_SIZE}" \
      --eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
      --grad_accum_steps "${GRADIENT_ACCUMULATION_STEPS}" \
      --epochs "${EPOCHS}" \
      --learning_rate "${LEARNING_RATE}" \
      --weight_decay "${WEIGHT_DECAY}" \
      --seed "${SEED}" \
      --save_best_by "${SAVE_BEST_BY}" \
      --precision "${PRECISION}" \
      --log_steps "${LOG_STEPS}" \
      --export_diagnostic_predictions \
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

python thesis_exp/exp17_low_score_evidence/collect_exp17_c0_results.py \
  --output_dir "${OUTPUT_DIR}" \
  --seed "${SEED}" \
  --configs "${CONFIG_ARRAY[@]}"

if is_truthy "${RUN_C0_D1_DIAG}"; then
  python thesis_exp/exp17_low_score_evidence/diagnose_exp17_c0_outcomes.py \
    --c0-output-dir "${OUTPUT_DIR}" \
    --exp16a-init-predictions thesis_exp/outputs/exp16_boundary_linking/runs/qmr/seed_42/predictions_dev.jsonl \
    --dev-jsonl thesis_exp/data/splits/question_seed42/dev.jsonl \
    --d1-dir "${D1_DIR}" \
    --out-dir thesis_exp/exp17_low_score_evidence/outputs/exp17_c0b_d1_diagnostics_seed42 \
    --seed "${SEED}"
fi

echo "Exp17-C0b diagnostics completed for configs: ${C0B_CONFIGS}"
