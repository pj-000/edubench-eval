#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${EDUBENCH_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"
PYTHON="${PYTHON:-/home/jpang/miniconda3/envs/llama_factory/bin/python}"
[[ -x "${PYTHON}" ]] || PYTHON="$(command -v python3)"
MODEL="${RUBIMOR_MODEL_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
OUT="thesis_exp/exp45_dopr_head/outputs/exp45a_dopr_seed42"
mkdir -p "${OUT}/logs_private"
read -r -a GPUS <<<"${GPU_LIST}"
(( ${#GPUS[@]} > 0 )) || { echo "GPU_LIST is empty" >&2; exit 2; }

worker() {
  local worker_index="$1" gpu="$2" fold skip=()
  [[ "${SKIP_COMPLETED}" == "1" ]] && skip=(--skip-completed)
  for fold in 0 1 2 3 4; do
    (( fold % ${#GPUS[@]} == worker_index )) || continue
    echo "[exp45a] training/restoring E4 fold ${fold} on physical GPU ${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m thesis_exp.exp45_dopr_head.train_or_restore_exp45a_e4_encoders \
      --fold "${fold}" --model-name-or-path "${MODEL}" "${skip[@]}" \
      >"${OUT}/logs_private/encoder_fold_${fold}.log" 2>&1
    echo "[exp45a] extracting frozen embeddings fold ${fold} on physical GPU ${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m thesis_exp.exp45_dopr_head.extract_exp45a_frozen_embeddings \
      --fold "${fold}" --model-name-or-path "${MODEL}" "${skip[@]}" \
      >"${OUT}/logs_private/embedding_fold_${fold}.log" 2>&1
  done
}

pids=()
for index in "${!GPUS[@]}"; do
  worker "${index}" "${GPUS[$index]}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
(( status == 0 )) || { echo "Exp45A encoder/embedding worker failed; inspect logs_private" >&2; exit 1; }
