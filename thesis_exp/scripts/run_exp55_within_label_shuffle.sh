#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/jpang/miniconda3/envs/llama_factory/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
GPU_LIST="${GPU_LIST:-4 6 7}"
OUTPUT_ROOT="thesis_exp/outputs/exp55_within_label_shuffle"
ARTIFACT_ROOT="thesis_exp/artifacts/exp55_within_label_shuffle"
LOG_DIR="${OUTPUT_ROOT}/logs_private"

read -r -a GPUS <<<"${GPU_LIST}"
SEEDS=(42 43 44)
[[ "${#GPUS[@]}" -eq "${#SEEDS[@]}" ]] || {
  echo "Expected exactly three GPUs for seeds 42/43/44; got: ${GPU_LIST}" >&2
  exit 2
}
for gpu_id in "${GPUS[@]}"; do
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${gpu_id}")"
  gpu_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu_id}")"
  [[ "${gpu_name}" == *"3090"* ]] || {
    echo "GPU ${gpu_id} is not a 3090: ${gpu_name}" >&2
    exit 2
  }
  [[ "${gpu_used}" -lt 2000 ]] || {
    echo "GPU ${gpu_id} is not free: ${gpu_used} MiB used" >&2
    exit 2
  }
done

mkdir -p "${LOG_DIR}"
PYTHONPATH=. "${PYTHON_BIN}" -m thesis_exp.exp55_within_label_shuffle.audit \
  2>&1 | tee "${LOG_DIR}/shuffle_audit.log"

for seed in "${SEEDS[@]}"; do
  output_dir="${OUTPUT_ROOT}/runs/within_label_shuffled_soft/seed_${seed}"
  checkpoint_best="${ARTIFACT_ROOT}/within_label_shuffled_soft/seed_${seed}/best"
  [[ ! -e "${output_dir}" ]] || { echo "Refusing to overwrite ${output_dir}" >&2; exit 2; }
  [[ ! -e "${checkpoint_best}" ]] || { echo "Refusing to overwrite ${checkpoint_best}" >&2; exit 2; }
done

run_seed() {
  local seed="$1"
  local gpu_id="$2"
  local output_dir="${OUTPUT_ROOT}/runs/within_label_shuffled_soft/seed_${seed}"
  local checkpoint_best="${ARTIFACT_ROOT}/within_label_shuffled_soft/seed_${seed}/best"
  echo "Starting within-label shuffled-soft seed ${seed} on GPU ${gpu_id}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONPATH=. "${PYTHON_BIN}" \
    -m thesis_exp.exp55_within_label_shuffle.train \
    --model_name_or_path "${MODEL_PATH}" \
    --output_dir "${output_dir}" \
    --checkpoint_output_dir "${checkpoint_best}" \
    --seed "${seed}" \
    --gradient_checkpointing \
    --local_files_only \
    >"${LOG_DIR}/seed_${seed}.log" 2>&1
}

pids=()
for index in "${!SEEDS[@]}"; do
  run_seed "${SEEDS[$index]}" "${GPUS[$index]}" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
[[ "${failed}" -eq 0 ]] || {
  echo "At least one shuffled-soft training run failed; collection skipped" >&2
  exit 1
}

PYTHONPATH=. "${PYTHON_BIN}" -m thesis_exp.exp55_within_label_shuffle.collect \
  2>&1 | tee "${LOG_DIR}/collect.log"
