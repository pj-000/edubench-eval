#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-llama_factory}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
OUT_DIR="${OUT_DIR:-thesis_exp/exp17_low_score_evidence/outputs/exp19_r5f2_rejection_mining_seed42}"
PREDICTION_ROOT="${PREDICTION_ROOT:-${OUT_DIR}/predictions}"
CONFIG_ROOT="${CONFIG_ROOT:-${OUT_DIR}/predict_configs}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/predict_logs}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/jpang/models/modelscope/Qwen/Qwen3-4B}"
EVAL_DATASET="${EVAL_DATASET:-edubench_exp19_r5f2_generation_train}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
LOW_TEMP_SAMPLES="${LOW_TEMP_SAMPLES:-8}"
HIGH_TEMP_SAMPLES="${HIGH_TEMP_SAMPLES:-4}"
LOW_TEMPERATURE="${LOW_TEMPERATURE:-0.7}"
LOW_TOP_P="${LOW_TOP_P:-0.9}"
HIGH_TEMPERATURE="${HIGH_TEMPERATURE:-1.0}"
HIGH_TOP_P="${HIGH_TOP_P:-0.95}"
SEED="${SEED:-42}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

GENERATOR_NAMES=(
  "r0a_direct_qwen3_4b"
  "r1b_score_only_balanced"
  "r2n_reason_score_natural"
  "r2c_clean_reason_score_balanced"
  "r4b_shuffled_reason_balanced"
  "r5c_from_r1b"
)
GENERATOR_ADAPTERS=(
  ""
  "saves/edubench/qwen3-4b/r1_score_only_balanced_lora"
  "saves/edubench/qwen3-4b/r2_reason_score_natural_lora"
  "saves/edubench/qwen3-4b/r2_clean_reason_score_balanced_lora"
  "saves/edubench/qwen3-4b/r4_shuffled_reason_balanced_lora"
  "saves/edubench/qwen3-4b/r5c_dpo_scout_from_r1b_maxsteps100_lora"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

if [[ -f "${HOME}/miniconda3/bin/activate" ]]; then
  source "${HOME}/miniconda3/bin/activate" "${CONDA_ENV}"
fi

mkdir -p "${PREDICTION_ROOT}" "${CONFIG_ROOT}" "${LOG_DIR}"

if ! command -v llamafactory-cli >/dev/null 2>&1; then
  echo "ERROR: llamafactory-cli not found in CONDA_ENV=${CONDA_ENV}" >&2
  exit 1
fi

python -m py_compile thesis_exp/exp17_low_score_evidence/prepare_exp19_r5f2_rejection_mining.py

python thesis_exp/exp17_low_score_evidence/prepare_exp19_r5f2_rejection_mining.py \
  --stage prepare_inputs \
  --out-dir "${OUT_DIR}" \
  --seed "${SEED}"

IFS=' ' read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "ERROR: GPU_LIST is empty" >&2
  exit 1
fi

TOTAL_SAMPLES_PER_GENERATOR=$((LOW_TEMP_SAMPLES + HIGH_TEMP_SAMPLES))
if [[ "${TOTAL_SAMPLES_PER_GENERATOR}" -le 0 ]]; then
  echo "ERROR: LOW_TEMP_SAMPLES + HIGH_TEMP_SAMPLES must be positive" >&2
  exit 1
fi

cat <<CONFIG
Exp19-R5F2 expanded train-only rejection mining
CONDA_ENV=${CONDA_ENV}
GPU_LIST=${GPU_LIST}
OUT_DIR=${OUT_DIR}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}
EVAL_DATASET=${EVAL_DATASET}
LOW_TEMP_SAMPLES=${LOW_TEMP_SAMPLES}
HIGH_TEMP_SAMPLES=${HIGH_TEMP_SAMPLES}
LOW_TEMPERATURE=${LOW_TEMPERATURE}
LOW_TOP_P=${LOW_TOP_P}
HIGH_TEMPERATURE=${HIGH_TEMPERATURE}
HIGH_TOP_P=${HIGH_TOP_P}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS}
TOTAL_SAMPLES_PER_GENERATOR=${TOTAL_SAMPLES_PER_GENERATOR}
SKIP_COMPLETED=${SKIP_COMPLETED}
CONFIG

sample_temperature() {
  local sample_idx="$1"
  if (( sample_idx < LOW_TEMP_SAMPLES )); then
    echo "${LOW_TEMPERATURE}"
  else
    echo "${HIGH_TEMPERATURE}"
  fi
}

sample_top_p() {
  local sample_idx="$1"
  if (( sample_idx < LOW_TEMP_SAMPLES )); then
    echo "${LOW_TOP_P}"
  else
    echo "${HIGH_TOP_P}"
  fi
}

write_predict_config() {
  local config_path="$1"
  local adapter_path="$2"
  local output_dir="$3"
  local seed="$4"
  local temperature="$5"
  local top_p="$6"
  cat >"${config_path}" <<YAML
model_name_or_path: ${MODEL_NAME_OR_PATH}
trust_remote_code: true
stage: sft
do_predict: true
finetuning_type: lora
infer_backend: huggingface
dataset_dir: ${OUT_DIR}
eval_dataset: ${EVAL_DATASET}
template: qwen3_nothink
cutoff_len: 4096
overwrite_cache: true
preprocessing_num_workers: 16
output_dir: ${output_dir}
overwrite_output_dir: true
predict_with_generate: true
per_device_eval_batch_size: ${PER_DEVICE_EVAL_BATCH_SIZE}
max_new_tokens: ${MAX_NEW_TOKENS}
do_sample: true
temperature: ${temperature}
top_p: ${top_p}
num_beams: 1
repetition_penalty: 1.05
seed: ${seed}
bf16: true
bf16_full_eval: true
report_to: none
YAML
  if [[ -n "${adapter_path}" ]]; then
    printf 'adapter_name_or_path: %s\n' "${adapter_path}" >>"${config_path}"
  fi
}

run_one() {
  local generator_name="$1"
  local adapter_path="$2"
  local sample_idx="$3"
  local gpu_id="$4"
  local run_out="${PREDICTION_ROOT}/${generator_name}/sample_${sample_idx}"
  local config_path="${CONFIG_ROOT}/${generator_name}_sample_${sample_idx}.yaml"
  local log_path="${LOG_DIR}/predict_${generator_name}_sample_${sample_idx}_gpu${gpu_id}.log"
  local run_seed=$((SEED + sample_idx))
  local temperature
  local top_p
  temperature="$(sample_temperature "${sample_idx}")"
  top_p="$(sample_top_p "${sample_idx}")"

  if [[ -n "${adapter_path}" && ! -f "${adapter_path}/adapter_config.json" ]]; then
    echo "ERROR: Missing adapter for ${generator_name}: ${adapter_path}/adapter_config.json" >&2
    exit 1
  fi
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${run_out}/generated_predictions.jsonl" ]]; then
    echo "Skipping ${generator_name} sample_${sample_idx}: generated_predictions.jsonl exists"
    return 0
  fi

  rm -rf "${run_out}"
  mkdir -p "${run_out}"
  write_predict_config "${config_path}" "${adapter_path}" "${run_out}" "${run_seed}" "${temperature}" "${top_p}"
  echo "Starting ${generator_name} sample_${sample_idx} temp=${temperature} top_p=${top_p} on GPU ${gpu_id}; log=${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" llamafactory-cli train "${config_path}" 2>&1 | tee "${log_path}"
  echo "Completed ${generator_name} sample_${sample_idx}"
}

TASK_GENERATORS=()
TASK_ADAPTERS=()
TASK_SAMPLE_IDXS=()
for generator_idx in "${!GENERATOR_NAMES[@]}"; do
  for ((sample_idx = 0; sample_idx < TOTAL_SAMPLES_PER_GENERATOR; sample_idx++)); do
    TASK_GENERATORS+=("${GENERATOR_NAMES[$generator_idx]}")
    TASK_ADAPTERS+=("${GENERATOR_ADAPTERS[$generator_idx]}")
    TASK_SAMPLE_IDXS+=("${sample_idx}")
  done
done

run_queue() {
  local gpu_id="$1"
  shift
  local idx
  for idx in "$@"; do
    run_one "${TASK_GENERATORS[$idx]}" "${TASK_ADAPTERS[$idx]}" "${TASK_SAMPLE_IDXS[$idx]}" "${gpu_id}"
  done
}

queues=()
for _ in "${GPUS[@]}"; do
  queues+=("")
done
for idx in "${!TASK_GENERATORS[@]}"; do
  gpu_slot=$((idx % ${#GPUS[@]}))
  queues[$gpu_slot]="${queues[$gpu_slot]} ${idx}"
done

pids=()
for slot in "${!GPUS[@]}"; do
  read -r -a queue_indices <<< "${queues[$slot]}"
  if [[ "${#queue_indices[@]}" -eq 0 ]]; then
    continue
  fi
  echo "GPU ${GPUS[$slot]} queue:${queues[$slot]}"
  run_queue "${GPUS[$slot]}" "${queue_indices[@]}" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

python thesis_exp/exp17_low_score_evidence/prepare_exp19_r5f2_rejection_mining.py \
  --stage build_dpo \
  --out-dir "${OUT_DIR}" \
  --prediction-root "${PREDICTION_ROOT}" \
  --seed "${SEED}" \
  --total-samples-per-generator "${TOTAL_SAMPLES_PER_GENERATOR}"

echo "Exp19-R5F2 rejection mining completed."
