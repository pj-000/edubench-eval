#!/usr/bin/env bash
# Run CBRD's no-training tensor and real-model parity checks on one approved GPU.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

GPU_ID="${GPU_ID:?Set GPU_ID to one explicitly approved RTX 3090 index}"
PYTHON_BIN="${PYTHON_BIN:-/home/jpang/miniconda3/envs/llama_factory/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"

GPU_NAME="$(nvidia-smi --id="${GPU_ID}" --query-gpu=name --format=csv,noheader,nounits)"
if [[ "${GPU_NAME}" != *"3090"* ]]; then
  echo "Refusing Stage 0: GPU ${GPU_ID} is not an RTX 3090 (${GPU_NAME})" >&2
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model path not found: ${MODEL_PATH}" >&2
  exit 2
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

# These commands use two development rows only.  They never construct a
# dataloader over test and never execute optimizer.step().
# Source archive lives only inside the new Exp57 directory; it gives the
# server an immutable, hash-verified HMSA source snapshot without touching any
# legacy experiment path.
"${PYTHON_BIN}" -m thesis_exp.exp57_cbrd.archive_legacy_sources
"${PYTHON_BIN}" -m thesis_exp.exp57_cbrd.data_audit
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -m thesis_exp.exp57_cbrd.torch_audit
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -m thesis_exp.exp57_cbrd.preflight \
  --model-name-or-path "${MODEL_PATH}" --bf16 false

# Run the BF16 real-model route only when the installed PyTorch runtime exposes
# BF16 support.  FP32 parity above is the strict source-equivalence gate.
BF16_READY="$({ CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" - <<'PY'
import torch
print(int(torch.cuda.is_available() and torch.cuda.is_bf16_supported()))
PY
} | tail -n 1)"
if [[ "${BF16_READY}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -m thesis_exp.exp57_cbrd.preflight \
    --model-name-or-path "${MODEL_PATH}" --bf16 true
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -m thesis_exp.exp57_cbrd.mechanism_audit \
    --model-name-or-path "${MODEL_PATH}"
else
  echo "BF16 unsupported on approved GPU/runtime; FP32 route parity and AMP scalar audit remain recorded."
fi
