#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-llama_factory}"
GPU_LIST="${GPU_LIST:-0 1 2}"
DATASET_DIR="${DATASET_DIR:-thesis_exp/exp17_low_score_evidence/outputs/exp19_sft_dpo_datasets_seed42}"
OUT_DIR="${OUT_DIR:-thesis_exp/exp17_low_score_evidence/outputs/exp19_r5_next_scout}"
PREDICTION_ROOT="${PREDICTION_ROOT:-${OUT_DIR}/dev_predictions}"
CONFIG_ROOT="${CONFIG_ROOT:-${OUT_DIR}/dev_predict_configs}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/dev_predict_logs}"
DEV_JSONL="${DEV_JSONL:-thesis_exp/data/splits/question_seed42/dev.jsonl}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/jpang/models/modelscope/Qwen/Qwen3-4B}"
EVAL_DATASET="${EVAL_DATASET:-edubench_exp19_dev_score_eval}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
SCOUT_MAX_STEPS="${SCOUT_MAX_STEPS:-100}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

RUN_NAMES=(
  "r5e_from_r1b"
  "r5c_no_mid_from_r1b"
  "r5c_no_mid_from_r2c"
)
RUN_ADAPTERS=(
  "saves/edubench/qwen3-4b/r5e_dpo_control_scout_from_r1b_maxsteps${SCOUT_MAX_STEPS}_lora"
  "saves/edubench/qwen3-4b/r5c_no_mid_dpo_scout_from_r1b_maxsteps${SCOUT_MAX_STEPS}_lora"
  "saves/edubench/qwen3-4b/r5c_no_mid_dpo_scout_from_r2c_maxsteps${SCOUT_MAX_STEPS}_lora"
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

python -m py_compile \
  thesis_exp/exp17_low_score_evidence/prepare_exp19_sft_eval_datasets.py \
  thesis_exp/exp17_low_score_evidence/collect_exp19_r5_next_scout_dev_results.py

python thesis_exp/exp17_low_score_evidence/prepare_exp19_sft_eval_datasets.py \
  --dev-jsonl "${DEV_JSONL}" \
  --dataset-dir "${DATASET_DIR}" \
  --split-name dev \
  --max-examples "${MAX_EXAMPLES}"

IFS=' ' read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "ERROR: GPU_LIST is empty" >&2
  exit 1
fi

cat <<CONFIG
Exp19-R5 next scout dev prediction with LLaMA-Factory
CONDA_ENV=${CONDA_ENV}
GPU_LIST=${GPU_LIST}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}
DATASET_DIR=${DATASET_DIR}
EVAL_DATASET=${EVAL_DATASET}
DEV_JSONL=${DEV_JSONL}
OUT_DIR=${OUT_DIR}
PREDICTION_ROOT=${PREDICTION_ROOT}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS}
MAX_EXAMPLES=${MAX_EXAMPLES}
SCOUT_MAX_STEPS=${SCOUT_MAX_STEPS}
SKIP_COMPLETED=${SKIP_COMPLETED}
CONFIG

write_predict_config() {
  local config_path="$1"
  local adapter_path="$2"
  local output_dir="$3"
  cat >"${config_path}" <<YAML
model_name_or_path: ${MODEL_NAME_OR_PATH}
adapter_name_or_path: ${adapter_path}
trust_remote_code: true
stage: sft
do_predict: true
finetuning_type: lora
infer_backend: huggingface
dataset_dir: ${DATASET_DIR}
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
do_sample: false
temperature: 0.0
top_p: 1.0
num_beams: 1
repetition_penalty: 1.05
bf16: true
bf16_full_eval: true
report_to: none
YAML
}

run_one() {
  local run_name="$1"
  local adapter_path="$2"
  local gpu_id="$3"
  local run_out="${PREDICTION_ROOT}/${run_name}"
  local config_path="${CONFIG_ROOT}/${run_name}.yaml"
  local log_path="${LOG_DIR}/predict_${run_name}_gpu${gpu_id}.log"

  if [[ ! -f "${adapter_path}/adapter_config.json" ]]; then
    echo "ERROR: Missing DPO scout adapter for ${run_name}: ${adapter_path}/adapter_config.json" >&2
    exit 1
  fi
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${run_out}/generated_predictions.jsonl" ]]; then
    echo "Skipping ${run_name}: generated_predictions.jsonl exists (${run_out})"
    return 0
  fi

  rm -rf "${run_out}"
  mkdir -p "${run_out}"
  write_predict_config "${config_path}" "${adapter_path}" "${run_out}"
  echo "Starting ${run_name} dev prediction on GPU ${gpu_id}; log=${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" llamafactory-cli train "${config_path}" 2>&1 | tee "${log_path}"
  echo "Completed ${run_name} dev prediction on GPU ${gpu_id}"
}

run_queue() {
  local gpu_id="$1"
  shift
  local idx
  for idx in "$@"; do
    run_one "${RUN_NAMES[$idx]}" "${RUN_ADAPTERS[$idx]}" "${gpu_id}"
  done
}

queues=()
for _ in "${GPUS[@]}"; do
  queues+=("")
done
for idx in "${!RUN_NAMES[@]}"; do
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

python thesis_exp/exp17_low_score_evidence/collect_exp19_r5_next_scout_dev_results.py \
  --out-dir "${OUT_DIR}" \
  --prediction-root "${PREDICTION_ROOT}" \
  --reference-csv "${DATASET_DIR}/tables/exp19_dev_reference.csv"

echo "Exp19-R5 next scout dev prediction completed."
