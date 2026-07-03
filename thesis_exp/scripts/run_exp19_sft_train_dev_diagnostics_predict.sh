#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-llama_factory}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
OUT_DIR="${OUT_DIR:-thesis_exp/exp17_low_score_evidence/outputs/exp19_sft_train_dev_diagnostics_seed42}"
PREDICTION_ROOT="${PREDICTION_ROOT:-${OUT_DIR}/predictions}"
CONFIG_ROOT="${CONFIG_ROOT:-${OUT_DIR}/predict_configs}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/predict_logs}"
TRAIN_JSONL="${TRAIN_JSONL:-thesis_exp/data/splits/question_seed42/train.jsonl}"
DEV_JSONL="${DEV_JSONL:-thesis_exp/data/splits/question_seed42/dev.jsonl}"
A0_CANDIDATES="${A0_CANDIDATES:-thesis_exp/exp17_low_score_evidence/outputs/exp17_a0_train_hidden_failure_signals_seed42/train_hidden_failure_candidates.csv}"
A0_HIGH_CONTROLS="${A0_HIGH_CONTROLS:-thesis_exp/exp17_low_score_evidence/outputs/exp17_a0_train_hidden_failure_signals_seed42/train_clean_high_controls.csv}"
D1_DIR="${D1_DIR:-thesis_exp/exp17_low_score_evidence/outputs/d1_hidden_failure_audit_seed42_dev/summary_human_rationale_recovered}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/jpang/models/modelscope/Qwen/Qwen3-4B}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MAX_TRAIN_CLEAN_HIGH="${MAX_TRAIN_CLEAN_HIGH:-0}"
MAX_DEV_HIGH="${MAX_DEV_HIGH:-400}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

SUBSETS=(
  "train_low_reasoned_subset"
  "train_clean_high_subset"
  "dev_label2_subset"
  "dev_d1_hidden_subset"
  "dev_d1_matched_controls"
  "dev_high_subset"
)
ADAPTER_NAMES=(
  "r2n_reason_score_natural"
  "r2c_clean_reason_score_balanced"
  "r4b_shuffled_reason_balanced"
  "r1b_score_only_balanced"
)
ADAPTER_PATHS=(
  "saves/edubench/qwen3-4b/r2_reason_score_natural_lora"
  "saves/edubench/qwen3-4b/r2_clean_reason_score_balanced_lora"
  "saves/edubench/qwen3-4b/r4_shuffled_reason_balanced_lora"
  "saves/edubench/qwen3-4b/r1_score_only_balanced_lora"
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

python -m py_compile thesis_exp/exp17_low_score_evidence/run_exp19_sft_train_dev_diagnostics.py

python thesis_exp/exp17_low_score_evidence/run_exp19_sft_train_dev_diagnostics.py prepare \
  --train-jsonl "${TRAIN_JSONL}" \
  --dev-jsonl "${DEV_JSONL}" \
  --a0-candidates "${A0_CANDIDATES}" \
  --a0-high-controls "${A0_HIGH_CONTROLS}" \
  --d1-dir "${D1_DIR}" \
  --out-dir "${OUT_DIR}" \
  --seed 42 \
  --max-train-clean-high "${MAX_TRAIN_CLEAN_HIGH}" \
  --max-dev-high "${MAX_DEV_HIGH}"

IFS=' ' read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "ERROR: GPU_LIST is empty" >&2
  exit 1
fi

cat <<CONFIG
Exp19-SFT-D2 train-vs-dev structured failure diagnostic
CONDA_ENV=${CONDA_ENV}
GPU_LIST=${GPU_LIST}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}
OUT_DIR=${OUT_DIR}
PREDICTION_ROOT=${PREDICTION_ROOT}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS}
MAX_TRAIN_CLEAN_HIGH=${MAX_TRAIN_CLEAN_HIGH}
MAX_DEV_HIGH=${MAX_DEV_HIGH}
SKIP_COMPLETED=${SKIP_COMPLETED}

Adapters:
  r2n_reason_score_natural
  r2c_clean_reason_score_balanced
  r4b_shuffled_reason_balanced
  r1b_score_only_balanced

Subsets:
  ${SUBSETS[*]}
CONFIG

write_predict_config() {
  local config_path="$1"
  local adapter_path="$2"
  local dataset_name="$3"
  local output_dir="$4"
  cat >"${config_path}" <<YAML
model_name_or_path: ${MODEL_NAME_OR_PATH}
adapter_name_or_path: ${adapter_path}
trust_remote_code: true
stage: sft
do_predict: true
finetuning_type: lora
infer_backend: huggingface
dataset_dir: ${OUT_DIR}
eval_dataset: ${dataset_name}
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
  local adapter_name="$1"
  local adapter_path="$2"
  local subset="$3"
  local gpu_id="$4"
  local dataset_name="edubench_exp19_d2_${subset}"
  local run_out="${PREDICTION_ROOT}/${adapter_name}/${subset}"
  local config_path="${CONFIG_ROOT}/${adapter_name}_${subset}.yaml"
  local log_path="${LOG_DIR}/predict_${adapter_name}_${subset}_gpu${gpu_id}.log"

  if [[ ! -f "${adapter_path}/adapter_config.json" ]]; then
    echo "ERROR: Missing adapter for ${adapter_name}: ${adapter_path}/adapter_config.json" >&2
    exit 1
  fi

  if [[ "${SKIP_COMPLETED}" == "1" && -f "${run_out}/generated_predictions.jsonl" ]]; then
    echo "Skipping ${adapter_name}/${subset}: generated_predictions.jsonl exists (${run_out})"
    return 0
  fi

  rm -rf "${run_out}"
  mkdir -p "${run_out}"
  write_predict_config "${config_path}" "${adapter_path}" "${dataset_name}" "${run_out}"
  echo "Starting ${adapter_name}/${subset} on GPU ${gpu_id}; log=${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" llamafactory-cli train "${config_path}" 2>&1 | tee "${log_path}"
  echo "Completed ${adapter_name}/${subset} on GPU ${gpu_id}"
}

run_adapter_queue() {
  local gpu_id="$1"
  shift
  local adapter_idx subset
  for adapter_idx in "$@"; do
    for subset in "${SUBSETS[@]}"; do
      run_one "${ADAPTER_NAMES[$adapter_idx]}" "${ADAPTER_PATHS[$adapter_idx]}" "${subset}" "${gpu_id}"
    done
  done
}

queues=()
for _ in "${GPUS[@]}"; do
  queues+=("")
done
for idx in "${!ADAPTER_NAMES[@]}"; do
  gpu_slot=$((idx % ${#GPUS[@]}))
  queues[$gpu_slot]="${queues[$gpu_slot]} ${idx}"
done

pids=()
for slot in "${!GPUS[@]}"; do
  read -r -a queue_indices <<< "${queues[$slot]}"
  if [[ "${#queue_indices[@]}" -eq 0 ]]; then
    continue
  fi
  echo "GPU ${GPUS[$slot]} adapter queue:${queues[$slot]}"
  run_adapter_queue "${GPUS[$slot]}" "${queue_indices[@]}" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

python thesis_exp/exp17_low_score_evidence/run_exp19_sft_train_dev_diagnostics.py collect \
  --out-dir "${OUT_DIR}" \
  --prediction-root "${PREDICTION_ROOT}"

echo "Exp19-SFT-D2 train-vs-dev structured failure diagnostic completed."
