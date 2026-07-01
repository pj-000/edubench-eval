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
C0_CONFIGS="${C0_CONFIGS:-C0_0_ordinal_continue C0_1_all_pairs_gamma0p02_m0p2 C0_2_all_pairs_gamma0p05_m0p2 C0_3_all_pairs_gamma0p10_m0p2 C0_7_same_subject_only_gamma0p05_m0p2 C0_8_high_weight_only_gamma0p05_m0p2 C0_9_same_subject_high_weight_gamma0p05_m0p2 C0_10_exclude_format_auxiliary_gamma0p05_m0p2 C0_11_exclude_answer_key_dependent_gamma0p05_m0p2 C0_12_random_matched_metric_rubric_gamma0p05_m0p2 C0_13_random_matched_metric_rubric_subject_gamma0p05_m0p2 C0_6_random_pair_control_gamma0p05_m0p2 C0_14_same_question_group_upper_bound_gamma0p05_m0p2}"
OUTPUT_DIR="${OUTPUT_DIR:-thesis_exp/exp17_low_score_evidence/outputs/exp17_c0_pairwise_separation_seed42}"
A0_DIR="${A0_DIR:-thesis_exp/exp17_low_score_evidence/outputs/exp17_a0_train_hidden_failure_signals_seed42}"
D1_DIR="${D1_DIR:-thesis_exp/exp17_low_score_evidence/outputs/d1_hidden_failure_audit_seed42_dev}"
PAIR_AUDIT_DIR="${PAIR_AUDIT_DIR:-thesis_exp/exp17_low_score_evidence/outputs/exp17_c0_pair_noise_audit_seed42}"

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

read -r -a CONFIG_ARRAY <<< "${C0_CONFIGS}"
read -r -a GPU_ARRAY <<< "${GPU_LIST//,/ }"
if [[ "${#CONFIG_ARRAY[@]}" -lt 1 ]]; then
  echo "ERROR: C0_CONFIGS must contain at least one config." >&2
  exit 1
fi
if [[ "${#GPU_ARRAY[@]}" -lt 1 ]]; then
  echo "ERROR: GPU_LIST must contain at least one GPU id." >&2
  exit 1
fi
for config in "${CONFIG_ARRAY[@]}"; do
  case "${config}" in
    C0_0_ordinal_continue|\
    C0_1_all_pairs_gamma0p02_m0p2|\
    C0_2_all_pairs_gamma0p05_m0p2|\
    C0_3_all_pairs_gamma0p10_m0p2|\
    C0_4_pairwise_low_only_gamma0p05_m0p2|\
    C0_5_evidence_positive_plus_pairwise_low_gamma0p05_m0p2|\
    C0_6_random_pair_control_gamma0p05_m0p2|\
    C0_7_same_subject_only_gamma0p05_m0p2|\
    C0_8_high_weight_only_gamma0p05_m0p2|\
    C0_9_same_subject_high_weight_gamma0p05_m0p2|\
    C0_10_exclude_format_auxiliary_gamma0p05_m0p2|\
    C0_11_exclude_answer_key_dependent_gamma0p05_m0p2|\
    C0_12_random_matched_metric_rubric_gamma0p05_m0p2|\
    C0_13_random_matched_metric_rubric_subject_gamma0p05_m0p2|\
    C0_14_same_question_group_upper_bound_gamma0p05_m0p2) ;;
    *)
      echo "ERROR: unknown Exp17-C0 config '${config}'" >&2
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
  echo "ERROR: missing Exp16A qmr init checkpoint: ${INIT_CHECKPOINT}" >&2
  exit 1
fi

python thesis_exp/exp17_low_score_evidence/analyze_exp17_c0_pair_noise.py \
  --pairs "${A0_DIR}/train_hidden_failure_pairs.csv" \
  --train-jsonl thesis_exp/data/splits/question_seed42/train.jsonl \
  --out-dir "${PAIR_AUDIT_DIR}" \
  --seed "${SEED}"

SAME_QUESTION_PAIR_COUNT="$(python - <<PY
import csv
from pathlib import Path
path = Path("${PAIR_AUDIT_DIR}") / "pair_source_candidate_counts.csv"
count = 0
if path.exists():
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("pair_source") == "same_question_group_upper_bound":
                count = int(float(row.get("available_pair_count") or 0))
                break
print(count)
PY
)"
if [[ "${SAME_QUESTION_PAIR_COUNT}" -lt 1 ]]; then
  filtered=()
  for config in "${CONFIG_ARRAY[@]}"; do
    if [[ "${config}" == "C0_14_same_question_group_upper_bound_gamma0p05_m0p2" ]]; then
      echo "WARNING: skipping C0_14_same_question_group_upper_bound_gamma0p05_m0p2 because no same-question upper-bound pairs are available." >&2
    else
      filtered+=("${config}")
    fi
  done
  CONFIG_ARRAY=("${filtered[@]}")
fi

cat <<CONFIG
Exp17-C0 pairwise-low quality separation scout
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
MAX_LENGTH_QUALITY=${MAX_LENGTH_QUALITY}
MAX_LENGTH_BOUNDARY=${MAX_LENGTH_BOUNDARY}
PRECISION=${PRECISION}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING}
LOG_STEPS=${LOG_STEPS}
C0_CONFIGS=${C0_CONFIGS}
OUTPUT_DIR=${OUTPUT_DIR}
PAIR_AUDIT_DIR=${PAIR_AUDIT_DIR}
CONFIG

run_one() {
  local config="$1"
  local gpu="$2"
  local run_dir="${OUTPUT_DIR}/runs/${config}/seed_${SEED}"
  local log_dir="${run_dir}/logs"
  local log_path="${log_dir}/train_exp17_c0_${config}_seed_${SEED}_gpu${gpu}.log"
  if [[ "${SKIP_COMPLETED}" == "1" && "${RESET_RUN_DIR}" != "1" && -f "${run_dir}/metrics_dev.json" && -f "${run_dir}/exp17_c0_pair_eval.csv" ]]; then
    echo "Skipping Exp17-C0 ${config} seed ${SEED}: completed outputs found at ${run_dir}"
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
  echo "Starting Exp17-C0 ${config} seed ${SEED} on GPU ${gpu}; log=${log_path}"
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

echo "Exp17-C0 scout completed for configs: ${C0_CONFIGS}"
