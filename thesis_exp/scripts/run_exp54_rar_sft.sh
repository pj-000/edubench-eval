#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-llama_factory}"
GPU_ID="${GPU_ID:?Set one audited RTX A6000 GPU ID (0-3)}"
ARM="${ARM:?Set ARM to S0, R1, R2, or R3}"
SEED="${SEED:?Set SEED to 42, 43, or 44}"
AUTHORIZATION_LOCK="${AUTHORIZATION_LOCK:?A reviewed authorization lock is required}"
OUTPUT_DIR="${OUTPUT_DIR:?Set a new, empty per-arm/per-seed output path}"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

python -m thesis_exp.exp54_rar_sft.train_rar_sft \
  --arm "${ARM}" \
  --seed "${SEED}" \
  --authorization-lock "${AUTHORIZATION_LOCK}" \
  --output-dir "${OUTPUT_DIR}"
