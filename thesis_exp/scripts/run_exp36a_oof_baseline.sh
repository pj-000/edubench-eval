#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${HOME}/miniconda3/bin/activate" ]]; then
  source "${HOME}/miniconda3/bin/activate" "${CONDA_ENV:-llama_factory}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
OUT_DIR="thesis_exp/exp36_safer_score/outputs/exp36a_safer_score_seed42"
LOG_DIR="${OUT_DIR}/private/logs/oof"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

"${PYTHON_BIN}" thesis_exp/exp36_safer_score/prepare_exp36a_oof_folds.py --out-dir "${OUT_DIR}"
mkdir -p "${LOG_DIR}"
read -r -a GPUS <<<"${GPU_LIST}"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "GPU_LIST must not be empty" >&2
  exit 2
fi

run_fold() {
  local fold="$1"
  local gpu="$2"
  local summary="${OUT_DIR}/private/oof_folds/fold_${fold}_summary.json"
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${summary}" ]] && \
    "${PYTHON_BIN}" -c 'import json,sys;sys.exit(0 if json.load(open(sys.argv[1])).get("status")=="COMPLETED" else 1)' "${summary}"; then
    echo "Skipping completed Exp36A OOF fold ${fold}"
    return 0
  fi
  echo "Starting Exp36A OOF fold ${fold} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" thesis_exp/exp36_safer_score/train_exp36a_oof_human_baseline.py \
    --fold "${fold}" \
    --model-name-or-path "${MODEL_NAME_OR_PATH}" \
    --out-dir "${OUT_DIR}" \
    --epochs 10 --learning-rate 2e-5 --weight-decay 0.01 --warmup-ratio 0.05 \
    --batch-size 4 --eval-batch-size 4 --gradient-accumulation-steps 32 --max-length 2048 \
    2>&1 | tee "${LOG_DIR}/fold_${fold}.log"
}

pids=()
for worker in "${!GPUS[@]}"; do
  (
    for fold in 0 1 2 3 4; do
      if (( fold % ${#GPUS[@]} == worker )); then
        run_fold "${fold}" "${GPUS[$worker]}"
      fi
    done
  ) &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
if [[ "${failed}" != "0" ]]; then
  echo "At least one Exp36A OOF queue failed" >&2
  exit 1
fi

"${PYTHON_BIN}" thesis_exp/exp36_safer_score/build_exp36a_safer_supervision.py --out-dir "${OUT_DIR}"
"${PYTHON_BIN}" thesis_exp/exp36_safer_score/validate_exp36a_supervision.py --out-dir "${OUT_DIR}" --require-private
echo "Exp36A OOF baseline and supervision construction completed."

