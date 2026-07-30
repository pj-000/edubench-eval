#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 7 ]]; then
  echo "usage: $0 ARM GPU_UUID REPO ARTIFACT_ROOT TEST_PATH OUTPUT_ROOT PLAN" >&2
  exit 2
fi

arm="$1"
gpu_uuid="$2"
repo="$3"
artifact_root="$4"
test_path="$5"
output_root="$6"
plan="$7"
python_bin="/home/jpang/exp54_vllm063_shared/bin/python"

cd "$repo"
export PYTHONPATH="$repo"
export CUDA_VISIBLE_DEVICES="$gpu_uuid"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_USE_V1=1
export TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_CACHE_DIR="/tmp/exp54_mechanism_test_${arm}_${gpu_uuid}"
export TRITON_CACHE_DIR="/tmp/exp54_mechanism_test_triton_${arm}_${gpu_uuid}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

for seed in 42 43 44; do
  "$python_bin" -u -m \
    thesis_exp.exp54_rar_sft.run_mechanism_control_test_inference_vllm \
    --arm "$arm" \
    --seed "$seed" \
    --repo-root "$repo" \
    --artifact-root "$artifact_root" \
    --test-path "$test_path" \
    --output-root "$output_root" \
    --cuda-device-uuid "$gpu_uuid" \
    --plan "$plan"
done
