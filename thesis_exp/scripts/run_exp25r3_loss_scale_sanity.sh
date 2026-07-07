#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ENV="${CONDA_ENV:-llama_factory}"
GPU_ID="${GPU_ID:-0}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/jpang/models/modelscope/Qwen/Qwen3-4B}"
INIT_ADAPTER="${INIT_ADAPTER:-saves/edubench/qwen3-4b/r2_clean_reason_score_balanced_lora}"
OUT_DIR="${OUT_DIR:-thesis_exp/exp17_low_score_evidence/outputs/exp25r3_loss_scale_sanity_seed42}"
SCORE_DATA="${SCORE_DATA:-thesis_exp/exp17_low_score_evidence/outputs/exp25_structured_src_dpo_seed42/data/edubench_r7h_score_mismatch_only_train.json}"
MIXED_DATA="${MIXED_DATA:-thesis_exp/exp17_low_score_evidence/outputs/exp25_structured_src_dpo_seed42/data/edubench_r7h_structured_src_dpo_train.json}"
MAX_STEPS="${MAX_STEPS:-20}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

if [[ -f "${HOME}/miniconda3/bin/activate" ]]; then
  source "${HOME}/miniconda3/bin/activate" "${CONDA_ENV}"
fi

python -m py_compile \
  thesis_exp/exp17_low_score_evidence/train_exp25r3_field_mask_src_dpo.py \
  thesis_exp/exp17_low_score_evidence/diagnose_exp25r3_loss_scale.py

mkdir -p "${OUT_DIR}/logs" "${OUT_DIR}/run_summaries"

cat <<CONFIG
Exp25R3 loss-scale + field-mask sanity
CONDA_ENV=${CONDA_ENV}
GPU_ID=${GPU_ID}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}
INIT_ADAPTER=${INIT_ADAPTER}
OUT_DIR=${OUT_DIR}
MAX_STEPS=${MAX_STEPS}
LEARNING_RATE=${LEARNING_RATE}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}
CONFIG

CONFIGS=(
  "score_full_b0p03_ftx0p05|${SCORE_DATA}|full|0.03|0.05|32|0"
  "score_full_b1_ftx0|${SCORE_DATA}|full|1.0|0.0|32|0"
  "score_field_b1_ftx0|${SCORE_DATA}|field|1.0|0.0|32|0"
  "score_field_b3_ftx0|${SCORE_DATA}|field|3.0|0.0|32|0"
  "mixed_field_b1_ftx0|${MIXED_DATA}|field|1.0|0.0|0|16"
  "mixed_field_b3_ftx0|${MIXED_DATA}|field|3.0|0.0|0|16"
)

for spec in "${CONFIGS[@]}"; do
  IFS='|' read -r config_name data_path logp_mode beta pref_ftx max_examples examples_per_type <<< "${spec}"
  summary_path="${OUT_DIR}/run_summaries/${config_name}.json"
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${summary_path}" ]]; then
    echo "Skipping ${config_name}: ${summary_path} exists"
    continue
  fi
  echo "Starting Exp25R3 ${config_name}"
  python thesis_exp/exp17_low_score_evidence/train_exp25r3_field_mask_src_dpo.py \
    --config-name "${config_name}" \
    --model-name-or-path "${MODEL_NAME_OR_PATH}" \
    --adapter-name-or-path "${INIT_ADAPTER}" \
    --ref-adapter-name-or-path "${INIT_ADAPTER}" \
    --data "${data_path}" \
    --out-dir "${OUT_DIR}" \
    --logp-mode "${logp_mode}" \
    --beta "${beta}" \
    --pref-ftx "${pref_ftx}" \
    --max-steps "${MAX_STEPS}" \
    --learning-rate "${LEARNING_RATE}" \
    --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --logging-steps "${LOGGING_STEPS}" \
    --max-train-examples "${max_examples}" \
    --examples-per-negative-type "${examples_per_type}" \
    2>&1 | tee "${OUT_DIR}/logs/${config_name}.log"
done

python thesis_exp/exp17_low_score_evidence/diagnose_exp25r3_loss_scale.py --out-dir "${OUT_DIR}"

echo "Exp25R3 loss-scale sanity completed."
