#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

GPU_ID="${GPU_ID:?Set GPU_ID to an available RTX 3090}"
SCHEDULE="${SCHEDULE:?Set space-separated seed:variant jobs}"
PYTHON_BIN="${PYTHON_BIN:-/home/jpang/miniconda3/envs/llama_factory/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
ANNOTATIONS="${ANNOTATIONS:-private_data/exp62_summeval/model_annotations.aligned.jsonl}"
OUTPUT_ROOT="thesis_exp/outputs/exp62_summeval_routing_confirmation"
ARTIFACT_ROOT="thesis_exp/artifacts/exp62_summeval_routing_confirmation"

GPU_NAME="$(nvidia-smi --id="${GPU_ID}" --query-gpu=name --format=csv,noheader,nounits)"
if [[ "${GPU_NAME}" != *"3090"* ]]; then
  echo "GPU ${GPU_ID} is not an RTX 3090: ${GPU_NAME}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}/logs_private" "${OUTPUT_ROOT}/campaign"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for job in ${SCHEDULE}; do
  seed="${job%%:*}"
  variant="${job#*:}"
  case "${seed}" in 62|63|64|65|66) ;; *) echo "invalid seed ${seed}" >&2; exit 2 ;; esac
  case "${variant}" in
    direct_residual_blocked|routed_hmsa|orthogonal_only|parallel_only) ;;
    *) echo "invalid variant ${variant}" >&2; exit 2 ;;
  esac
  output_dir="${OUTPUT_ROOT}/runs/${variant}/seed_${seed}"
  checkpoint_dir="${ARTIFACT_ROOT}/${variant}/seed_${seed}/epoch10"
  log_path="${OUTPUT_ROOT}/logs_private/${variant}_seed_${seed}.log"
  if [[ -f "${checkpoint_dir}/checkpoint.json" && -f "${output_dir}/dev_summary.json" ]]; then
    echo "[exp62 queue gpu${GPU_ID}] already complete ${seed}:${variant}"
    continue
  fi
  echo "[exp62 queue gpu${GPU_ID}] start ${seed}:${variant} $(date -u +%FT%TZ)"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" \
    -m thesis_exp.exp62_summeval_routing_confirmation.train \
    --annotations "${ANNOTATIONS}" \
    --model_name_or_path "${MODEL_PATH}" \
    --variant "${variant}" \
    --seed "${seed}" \
    --output_dir "${output_dir}" \
    --checkpoint_dir "${checkpoint_dir}" \
    2>&1 | tee "${log_path}"
  echo "[exp62 queue gpu${GPU_ID}] complete ${seed}:${variant} $(date -u +%FT%TZ)"
done

printf 'COMPLETE\n' > "${OUTPUT_ROOT}/campaign/queue_gpu_${GPU_ID}.status"
