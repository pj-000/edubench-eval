#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 6 ]]; then
  echo "usage: $0 SEED GPU_UUID REPO ARTIFACT_ROOT OUTPUT_ROOT ARM..." >&2
  exit 2
fi

seed="$1"
gpu_uuid="$2"
repo="$3"
artifact_root="$4"
output_root="$5"
shift 5
arms=("$@")
python_bin="/home/jpang/miniconda3/envs/llama_factory/bin/python"

cd "$repo"
export PYTHONPATH="$repo"
export PYTHONHASHSEED="$seed"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export CUDA_VISIBLE_DEVICES="$gpu_uuid"

for arm in "${arms[@]}"; do
  "$python_bin" -m thesis_exp.exp54_rar_sft.train_mechanism_controls_formal \
    --arm "$arm" \
    --seed "$seed" \
    --cuda-device-uuid "$gpu_uuid" \
    --artifact-root "$artifact_root" \
    --output-root "$output_root" \
    --execute
done
