#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
if [[ -f "${HOME}/miniconda3/bin/activate" ]]; then source "${HOME}/miniconda3/bin/activate" "${CONDA_ENV:-llama_factory}"; fi

[[ "${RUN_FORMAL:-0}" == "1" ]] || { echo "Set RUN_FORMAL=1 to launch the locked Exp49 scout" >&2; exit 2; }
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
GPU_LIST="${GPU_LIST:-0 1}"
SEED=42
OUT_ROOT="thesis_exp/outputs/exp49_cphce"
LOG_ROOT="${OUT_ROOT}/logs_private/seed_${SEED}"
mkdir -p "${LOG_ROOT}"
read -r -a GPUS <<<"${GPU_LIST}"
[[ "${#GPUS[@]}" -ge 2 ]] || { echo "GPU_LIST must contain at least two GPUs" >&2; exit 2; }

run_arm() {
  local variant="$1" gpu="$2"
  local summary="${OUT_ROOT}/runs/${variant}/seed_${SEED}/run_summary.json"
  if [[ -f "${summary}" ]] && "${PYTHON_BIN}" -c 'import json,sys;sys.exit(0 if json.load(open(sys.argv[1])).get("status")=="COMPLETED" else 1)' "${summary}"; then
    echo "Skipping completed ${variant} seed ${SEED}"
    return 0
  fi
  echo "Starting Exp49 ${variant} seed ${SEED} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m thesis_exp.exp49_cphce.train \
    --variant "${variant}" \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --seed "${SEED}" \
    --max_length 2048 \
    --num_train_epochs 10 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --max_grad_norm 1.0 \
    --bf16 auto \
    --gradient_checkpointing \
    --local_files_only \
    2>&1 | tee "${LOG_ROOT}/${variant}.log"
}

run_arm b0_hard_ce "${GPUS[0]}" & p0=$!
run_arm m1_human_soft "${GPUS[1]}" & p1=$!
failed=0
wait "${p0}" || failed=1
wait "${p1}" || failed=1
[[ "${failed}" == "0" ]] || { echo "Exp49 seed42 training failed" >&2; exit 1; }
"${PYTHON_BIN}" -m thesis_exp.exp49_cphce.formal_gate --mode seed42 --bootstrap-resamples 10000
