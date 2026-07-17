#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
if [[ -f "${HOME}/miniconda3/bin/activate" ]]; then source "${HOME}/miniconda3/bin/activate" "${CONDA_ENV:-llama_factory}"; fi
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
DECISION="thesis_exp/outputs/exp49_cphce/decision/seed42_decision.json"
"${PYTHON_BIN}" -c 'import json,sys;v=json.load(open(sys.argv[1]));sys.exit(0 if v.get("status")=="SEED42_PASS" else 2)' "${DECISION}"
read -r -a GPUS <<<"${GPU_LIST}"
[[ "${#GPUS[@]}" -ge 4 ]] || { echo "Formal run requires four GPU entries" >&2; exit 2; }
jobs=("43|b0_hard_ce" "43|m1_human_soft" "44|b0_hard_ce" "44|m1_human_soft")
pids=()
for index in "${!jobs[@]}"; do
  IFS='|' read -r seed variant <<<"${jobs[$index]}"
  log_dir="thesis_exp/outputs/exp49_cphce/logs_private/seed_${seed}"
  mkdir -p "${log_dir}"
  (
    echo "Starting Exp49 ${variant} seed ${seed} on GPU ${GPUS[$index]}"
    CUDA_VISIBLE_DEVICES="${GPUS[$index]}" "${PYTHON_BIN}" -m thesis_exp.exp49_cphce.train \
      --variant "${variant}" --model_name_or_path "${MODEL_NAME_OR_PATH}" --seed "${seed}" \
      --max_length 2048 --num_train_epochs 10 --learning_rate 2e-5 --weight_decay 0.01 --warmup_ratio 0.05 \
      --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 32 \
      --max_grad_norm 1.0 --bf16 auto --gradient_checkpointing --local_files_only \
      2>&1 | tee "${log_dir}/${variant}.log"
  ) &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
[[ "${failed}" == "0" ]] || { echo "Exp49 formal training failed" >&2; exit 1; }
"${PYTHON_BIN}" -m thesis_exp.exp49_cphce.formal_gate --mode formal --bootstrap-resamples 10000
