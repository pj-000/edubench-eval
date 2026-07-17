#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
if [[ -f "${HOME}/miniconda3/bin/activate" ]]; then source "${HOME}/miniconda3/bin/activate" "${CONDA_ENV:-llama_factory}"; fi
[[ "${RUN_FORMAL:-0}" == "1" ]] || { echo "Set RUN_FORMAL=1 to launch Exp50 seed42" >&2; exit 2; }
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
GPU_ID="${GPU_ID:-0}"
BASELINE="thesis_exp/outputs/exp49_cphce/runs/b0_hard_ce/seed_42/run_summary.json"
"${PYTHON_BIN}" - "${BASELINE}" <<'PY'
import json,sys
v=json.load(open(sys.argv[1]))
assert v['status']=='COMPLETED' and v['test_access_count']==0
assert v['scheduler']=='cosine_with_warmup'
assert v['checkpoint_rule']=='highest Exact_rounded; ties keep earlier epoch'
PY
OUT="thesis_exp/outputs/exp50_cahs/runs/c1_cahs_0p5/seed_42"
SUMMARY="${OUT}/run_summary.json"
mkdir -p thesis_exp/outputs/exp50_cahs/logs_private
if [[ ! -f "${SUMMARY}" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -m thesis_exp.exp50_cahs.train \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" --seed 42 \
    --max_length 2048 --num_train_epochs 10 --learning_rate 2e-5 --weight_decay 0.01 --warmup_ratio 0.05 \
    --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 32 \
    --max_grad_norm 1.0 --bf16 auto --gradient_checkpointing --local_files_only \
    2>&1 | tee thesis_exp/outputs/exp50_cahs/logs_private/seed42_cahs.log
fi
"${PYTHON_BIN}" -m thesis_exp.exp50_cahs.gate
