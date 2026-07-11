#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${HOME}/miniconda3/bin/activate" ]]; then
  source "${HOME}/miniconda3/bin/activate" "${CONDA_ENV:-llama_factory}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_FORMAL="${RUN_FORMAL:-1}"
SEEDS="${SEEDS:-42 43 44}"
GPU_LIST="${GPU_LIST:-0 1 2}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
RESET_RUN_DIR="${RESET_RUN_DIR:-0}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
VARIANT="v3_safe16_original_low_anchor"
OUT_DIR="thesis_exp/exp17_low_score_evidence/outputs/exp27q_safe16_multiseed_seed42_44"
TRAIN_JSONL="${OUT_DIR}/private/data/exp27q_v3_safe16_original_low_anchor_train.jsonl"
RUN_ROOT="thesis_exp/runs/exp27q_safe16"
ARTIFACT_ROOT="thesis_exp/artifacts/exp27q_safe16"

if [[ "${RUN_FORMAL}" != "1" ]]; then
  echo "Exp27Q formal gate is closed. Set RUN_FORMAL=1." >&2
  exit 2
fi
if [[ "${SEEDS}" != "42 43 44" ]]; then
  echo 'Exp27Q is locked to SEEDS="42 43 44".' >&2
  exit 2
fi

read -r -a SEED_ARRAY <<<"${SEEDS}"
read -r -a GPUS <<<"${GPU_LIST}"
if [[ "${#GPUS[@]}" -lt 3 ]]; then
  echo "Exp27Q formal requires three GPU IDs." >&2
  exit 2
fi

"${PYTHON_BIN}" thesis_exp/exp17_low_score_evidence/prepare_exp27q_v3_safe16_dataset.py
"${PYTHON_BIN}" thesis_exp/exp17_low_score_evidence/validate_exp27q_safe16_dataset.py --out-dir "${OUT_DIR}"

run_seed() {
  local seed="$1"
  local gpu="$2"
  local run_dir="${RUN_ROOT}/${VARIANT}/seed_${seed}"
  local artifact_dir="${ARTIFACT_ROOT}/${VARIANT}/seed_${seed}"
  local summary="${run_dir}/run_summary.json"
  local log_dir="${OUT_DIR}/logs_private/seed_${seed}"
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${summary}" ]] && \
    "${PYTHON_BIN}" -c 'import json,sys;sys.exit(0 if json.load(open(sys.argv[1])).get("status")=="COMPLETED" else 1)' "${summary}"; then
    echo "Skipping completed Exp27Q seed ${seed}"
    return
  fi
  if [[ "${RESET_RUN_DIR}" == "1" ]]; then
    rm -rf "${run_dir}" "${artifact_dir}"
  fi
  mkdir -p "${log_dir}"
  echo "Starting Exp27Q ${VARIANT} seed ${seed} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m thesis_exp.src.edujudge.exp27q.train_exp27q_safe16 \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --train_jsonl "${TRAIN_JSONL}" \
    --run_dir "${run_dir}" \
    --output_dir "${OUT_DIR}" \
    --checkpoint_output_dir "${artifact_dir}" \
    --seed "${seed}" \
    --max_length 2048 \
    --num_train_epochs 10 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --max_grad_norm 1.0 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --log_steps 5 2>&1 | tee "${log_dir}/${VARIANT}.log"
}

pids=()
for index in "${!SEED_ARRAY[@]}"; do
  run_seed "${SEED_ARRAY[$index]}" "${GPUS[$index]}" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
if [[ "${failed}" != "0" ]]; then
  echo "At least one Exp27Q seed failed." >&2
  exit 1
fi

"${PYTHON_BIN}" -m thesis_exp.src.edujudge.exp27q.collect_exp27q_safe16_multiseed \
  --run-root "${RUN_ROOT}" \
  --output-dir "${OUT_DIR}" \
  --seeds 42 43 44 \
  --cluster-resamples 2000 \
  --two-level-resamples 5000

echo "Exp27Q locked Safe16 multiseed training and dev-only collection completed."

