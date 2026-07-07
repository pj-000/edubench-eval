#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-llama_factory}"
GPU_LIST="${GPU_LIST:-0 1 2}"
OUT_DIR="${OUT_DIR:-thesis_exp/exp17_low_score_evidence/outputs/exp24_orc_dpo_seed42}"
DATASET_DIR="${DATASET_DIR:-thesis_exp/exp17_low_score_evidence/outputs/exp19_sft_dpo_datasets_seed42}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/jpang/models/modelscope/Qwen/Qwen3-4B}"
INIT_ADAPTER="${INIT_ADAPTER:-saves/edubench/qwen3-4b/r2_clean_reason_score_balanced_lora}"
REF_ADAPTER="${REF_ADAPTER:-${INIT_ADAPTER}}"
SAVE_ROOT="${SAVE_ROOT:-saves/edubench/qwen3-4b}"
DEV_JSONL="${DEV_JSONL:-thesis_exp/data/splits/question_seed42/dev.jsonl}"
TEST_JSONL="${TEST_JSONL:-thesis_exp/data/splits/question_seed42/test.jsonl}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"
PREDICTION_ROOT="${PREDICTION_ROOT:-${OUT_DIR}/dev_predictions}"
SUMMARY_DIR="${SUMMARY_DIR:-${OUT_DIR}/training_summaries}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
TRAIN_ONLY="${TRAIN_ONLY:-0}"
PREDICT_ONLY="${PREDICT_ONLY:-0}"
COLLECT_ONLY="${COLLECT_ONLY:-0}"
MAX_STEPS="${MAX_STEPS:-100}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-0}"
MAX_PREDICT_EXAMPLES="${MAX_PREDICT_EXAMPLES:-0}"

RUN_NAMES=(
  "exp24_orc_a_r2c"
  "exp24_orc_b_r2c"
  "exp24_orc_c_r2c"
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

mkdir -p "${LOG_DIR}" "${PREDICTION_ROOT}" "${SUMMARY_DIR}"

python -m py_compile \
  thesis_exp/exp17_low_score_evidence/prepare_exp24_orc_score_channel_data.py \
  thesis_exp/exp17_low_score_evidence/validate_exp24_orc_metadata.py \
  thesis_exp/exp17_low_score_evidence/train_exp24_orc_dpo.py \
  thesis_exp/exp17_low_score_evidence/collect_exp24_orc_dpo_dev.py \
  thesis_exp/exp17_low_score_evidence/prepare_exp19_sft_eval_datasets.py

if [[ "${COLLECT_ONLY}" != "1" && "${PREDICT_ONLY}" != "1" ]]; then
  python thesis_exp/exp17_low_score_evidence/prepare_exp24_orc_score_channel_data.py \
    --dev-jsonl "${DEV_JSONL}" \
    --test-jsonl "${TEST_JSONL}" \
    --out-dir "${OUT_DIR}"

  python thesis_exp/exp17_low_score_evidence/validate_exp24_orc_metadata.py \
    --data "${OUT_DIR}/data/edubench_r7g_orc_score_channel_reason_aux_train.json" \
    --out-dir "${OUT_DIR}"
fi

python thesis_exp/exp17_low_score_evidence/prepare_exp19_sft_eval_datasets.py \
  --dev-jsonl "${DEV_JSONL}" \
  --dataset-dir "${DATASET_DIR}" \
  --split-name dev \
  --max-examples "${MAX_PREDICT_EXAMPLES}"

IFS=' ' read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "ERROR: GPU_LIST is empty" >&2
  exit 1
fi

cat <<CONFIG
Exp24 score-channel ORC-DPO
CONDA_ENV=${CONDA_ENV}
GPU_LIST=${GPU_LIST}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}
INIT_ADAPTER=${INIT_ADAPTER}
REF_ADAPTER=${REF_ADAPTER}
OUT_DIR=${OUT_DIR}
SAVE_ROOT=${SAVE_ROOT}
MAX_STEPS=${MAX_STEPS}
LEARNING_RATE=${LEARNING_RATE}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}
SKIP_COMPLETED=${SKIP_COMPLETED}
TRAIN_ONLY=${TRAIN_ONLY}
PREDICT_ONLY=${PREDICT_ONLY}
COLLECT_ONLY=${COLLECT_ONLY}

Runs:
  A: mild low/high risk weighting
  B: stronger low-to-high risk weighting
  C: balanced low/high risk weighting + stronger reason auxiliary
CONFIG

run_args() {
  local run_name="$1"
  case "${run_name}" in
    exp24_orc_a_r2c)
      echo "--alpha-lh 1.0 --alpha-hl 0.75 --alpha-lm 0.25 --alpha-hm 0.25 --alpha-d 0.15 --margin-lh 0.05 --margin-hl 0.05 --margin-d 0.03 --lambda-reason 0.03"
      ;;
    exp24_orc_b_r2c)
      echo "--alpha-lh 1.5 --alpha-hl 1.0 --alpha-lm 0.25 --alpha-hm 0.25 --alpha-d 0.20 --margin-lh 0.10 --margin-hl 0.05 --margin-d 0.05 --lambda-reason 0.03"
      ;;
    exp24_orc_c_r2c)
      echo "--alpha-lh 1.0 --alpha-hl 1.0 --alpha-lm 0.25 --alpha-hm 0.25 --alpha-d 0.20 --margin-lh 0.05 --margin-hl 0.10 --margin-d 0.05 --lambda-reason 0.05"
      ;;
    *)
      echo "ERROR: unknown run ${run_name}" >&2
      exit 1
      ;;
  esac
}

train_one() {
  local run_name="$1"
  local gpu_id="$2"
  local output_dir="${SAVE_ROOT}/${run_name}"
  local log_path="${LOG_DIR}/train_${run_name}_gpu${gpu_id}.log"
  if [[ ! -f "${INIT_ADAPTER}/adapter_config.json" ]]; then
    echo "ERROR: missing init adapter: ${INIT_ADAPTER}/adapter_config.json" >&2
    exit 1
  fi
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${output_dir}/adapter_config.json" ]]; then
    echo "Skipping train ${run_name}: completed adapter exists (${output_dir}/adapter_config.json)"
    return 0
  fi
  read -r -a extra_args <<< "$(run_args "${run_name}")"
  echo "Starting Exp24 train ${run_name} on GPU ${gpu_id}; log=${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" python thesis_exp/exp17_low_score_evidence/train_exp24_orc_dpo.py \
    --mode train \
    --run-name "${run_name}" \
    --model-name-or-path "${MODEL_NAME_OR_PATH}" \
    --adapter-name-or-path "${INIT_ADAPTER}" \
    --ref-adapter-name-or-path "${REF_ADAPTER}" \
    --data "${OUT_DIR}/data/edubench_r7g_orc_score_channel_reason_aux_train.json" \
    --output-dir "${output_dir}" \
    --summary-dir "${SUMMARY_DIR}" \
    --max-steps "${MAX_STEPS}" \
    --learning-rate "${LEARNING_RATE}" \
    --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --max-train-examples "${MAX_TRAIN_EXAMPLES}" \
    "${extra_args[@]}" \
    2>&1 | tee "${log_path}"
  echo "Completed Exp24 train ${run_name}"
}

predict_one() {
  local run_name="$1"
  local gpu_id="$2"
  local output_dir="${SAVE_ROOT}/${run_name}"
  local pred_dir="${PREDICTION_ROOT}/${run_name}"
  local log_path="${LOG_DIR}/predict_${run_name}_gpu${gpu_id}.log"
  if [[ ! -f "${output_dir}/adapter_config.json" ]]; then
    echo "ERROR: missing trained adapter for prediction: ${output_dir}/adapter_config.json" >&2
    exit 1
  fi
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${pred_dir}/generated_predictions.jsonl" ]]; then
    echo "Skipping predict ${run_name}: generated_predictions.jsonl exists (${pred_dir})"
    return 0
  fi
  rm -rf "${pred_dir}"
  mkdir -p "${pred_dir}"
  echo "Starting Exp24 dev prediction ${run_name} on GPU ${gpu_id}; log=${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" python thesis_exp/exp17_low_score_evidence/train_exp24_orc_dpo.py \
    --mode predict \
    --run-name "${run_name}" \
    --model-name-or-path "${MODEL_NAME_OR_PATH}" \
    --adapter-name-or-path "${output_dir}" \
    --dev-jsonl "${DEV_JSONL}" \
    --prediction-dir "${pred_dir}" \
    --max-predict-examples "${MAX_PREDICT_EXAMPLES}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    2>&1 | tee "${log_path}"
  echo "Completed Exp24 dev prediction ${run_name}"
}

run_parallel() {
  local phase="$1"
  pids=()
  for idx in "${!RUN_NAMES[@]}"; do
    local gpu_slot=$((idx % ${#GPUS[@]}))
    local gpu_id="${GPUS[$gpu_slot]}"
    local run_name="${RUN_NAMES[$idx]}"
    echo "GPU ${gpu_id} ${phase} queue: ${run_name}"
    if [[ "${phase}" == "train" ]]; then
      train_one "${run_name}" "${gpu_id}" &
    else
      predict_one "${run_name}" "${gpu_id}" &
    fi
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
}

if [[ "${COLLECT_ONLY}" != "1" && "${PREDICT_ONLY}" != "1" ]]; then
  run_parallel train
fi

if [[ "${COLLECT_ONLY}" != "1" && "${TRAIN_ONLY}" != "1" ]]; then
  run_parallel predict
fi

if [[ "${TRAIN_ONLY}" != "1" ]]; then
  python thesis_exp/exp17_low_score_evidence/collect_exp24_orc_dpo_dev.py \
    --out-dir "${OUT_DIR}" \
    --prediction-root "${PREDICTION_ROOT}" \
    --reference-csv "${DATASET_DIR}/tables/exp19_dev_reference.csv"
fi

echo "Exp24 score-channel ORC-DPO workflow completed."
