#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
if [[ -f "${HOME}/miniconda3/bin/activate" ]]; then source "${HOME}/miniconda3/bin/activate" "${CONDA_ENV:-llama_factory}"; fi

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_FORMAL="${RUN_FORMAL:-0}"
SEED="${SEED:-42}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
RUN_VARIANTS="${RUN_VARIANTS:-v0_original_hard v0h_human_soft v1_qwen_hard v2_qwen_range_soft v3_naive_human_qwen v4_human_soft_logit_adjustment v5_safer_score v7_shuffled_teacher_control}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
RESET_RUN_DIR="${RESET_RUN_DIR:-0}"
OUT_DIR="thesis_exp/exp36_safer_score/outputs/exp36a_safer_score_seed42"
RUN_ROOT="thesis_exp/runs/exp36_safer_score"
ARTIFACT_ROOT="thesis_exp/artifacts/exp36_safer_score"
LOG_DIR="${OUT_DIR}/private/logs/seed_${SEED}"

if [[ "${RUN_FORMAL}" != "1" || "${SEED}" != "42" ]]; then
  echo "Set RUN_FORMAL=1 and keep SEED=42 for the locked scout." >&2; exit 2
fi
"${PYTHON_BIN}" thesis_exp/exp36_safer_score/validate_exp36a_supervision.py --out-dir "${OUT_DIR}" --require-private
"${PYTHON_BIN}" -c 'import json,sys;sys.exit(0 if json.load(open(sys.argv[1])).get("status")=="PASS" else 1)' "${OUT_DIR}/decision/exp36a_smoke_decision.json"
mkdir -p "${LOG_DIR}"
read -r -a GPUS <<<"${GPU_LIST}"
read -r -a VARIANTS <<<"${RUN_VARIANTS}"

run_variant() {
  local variant="$1"
  local gpu="$2"
  local summary="${RUN_ROOT}/${variant}/seed_${SEED}/run_summary.json"
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${summary}" ]] && "${PYTHON_BIN}" -c 'import json,sys;sys.exit(0 if json.load(open(sys.argv[1])).get("status")=="COMPLETED" else 1)' "${summary}"; then
    echo "Skipping completed Exp36A ${variant}"; return 0
  fi
  if [[ "${RESET_RUN_DIR}" == "1" ]]; then rm -rf "${RUN_ROOT:?}/${variant}" "${ARTIFACT_ROOT:?}/${variant}"; fi
  echo "Starting Exp36A ${variant} seed ${SEED} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" thesis_exp/exp36_safer_score/train_exp36a_safer_score.py \
    --variant "${variant}" --model-name-or-path "${MODEL_NAME_OR_PATH}" --out-dir "${OUT_DIR}" \
    --run-dir "${RUN_ROOT}/${variant}/seed_${SEED}" --artifact-dir "${ARTIFACT_ROOT}/${variant}/seed_${SEED}" \
    --epochs 10 --learning-rate 2e-5 --weight-decay 0.01 --warmup-ratio 0.05 \
    --batch-size 4 --eval-batch-size 4 --gradient-accumulation-steps 32 --max-length 2048 \
    2>&1 | tee "${LOG_DIR}/${variant}.log"
}

pids=()
for worker in "${!GPUS[@]}"; do
  ( for index in "${!VARIANTS[@]}"; do if (( index % ${#GPUS[@]} == worker )); then run_variant "${VARIANTS[$index]}" "${GPUS[$worker]}"; fi; done ) &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do if ! wait "${pid}"; then failed=1; fi; done
if [[ "${failed}" != "0" ]]; then echo "At least one Exp36A queue failed" >&2; exit 1; fi
"${PYTHON_BIN}" thesis_exp/exp36_safer_score/collect_exp36a_seed42.py --out-dir "${OUT_DIR}" --run-root "${RUN_ROOT}" --bootstrap-resamples 2000
echo "Exp36A seed42 scout completed."
