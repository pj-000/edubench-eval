#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${EDUBENCH_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-/home/jpang/miniconda3/envs/llama_factory/bin/python}"
MODEL="${RUBIMOR_MODEL_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
MODE="${MODE:-groupcv}"
VARIANTS_STRING="${VARIANTS:?VARIANTS is required}"
SEEDS_STRING="${SEEDS:-42}"
FOLDS_STRING="${FOLDS:-0 1 2 3 4}"
OUT_DIR="thesis_exp/exp43_rubimor/outputs/exp43_rubimor_preregistered"
RUN_ROOT="thesis_exp/runs/exp43_rubimor"
ARTIFACT_ROOT="thesis_exp/artifacts/exp43_rubimor"
read -r -a GPUS <<< "${GPU_LIST}"
read -r -a RUN_VARIANTS <<< "${VARIANTS_STRING}"
read -r -a RUN_SEEDS <<< "${SEEDS_STRING}"
read -r -a RUN_FOLDS <<< "${FOLDS_STRING}"
[[ -d "${MODEL}" ]] || { echo "Missing local model: ${MODEL}" >&2; exit 2; }
[[ ${#GPUS[@]} -gt 0 ]] || { echo "GPU_LIST is empty" >&2; exit 2; }

JOBS=()
for variant in "${RUN_VARIANTS[@]}"; do
  for seed in "${RUN_SEEDS[@]}"; do
    if [[ "${MODE}" == "headline" ]]; then
      JOBS+=("${variant}:${seed}:0")
    else
      for fold in "${RUN_FOLDS[@]}"; do JOBS+=("${variant}:${seed}:${fold}"); done
    fi
  done
done
echo "Exp43 ${MODE} matrix: ${#JOBS[@]} runs; variants=${VARIANTS_STRING}; seeds=${SEEDS_STRING}; GPUs=${GPU_LIST}"

run_queue() {
  local gpu="$1"; shift
  local job variant rest seed fold run_dir summary prediction log extra
  for job in "$@"; do
    variant="${job%%:*}"; rest="${job#*:}"; seed="${rest%%:*}"; fold="${rest##*:}"
    if [[ "${MODE}" == "headline" ]]; then
      run_dir="${RUN_ROOT}/headline/${variant}/seed_${seed}"
      prediction="${run_dir}/best_dev_predictions.jsonl"
    else
      run_dir="${RUN_ROOT}/${MODE}/${variant}/seed_${seed}/fold_${fold}"
      prediction="${run_dir}/heldout_predictions.jsonl"
    fi
    summary="${run_dir}/run_summary.json"
    if [[ "${SKIP_COMPLETED:-1}" == "1" && -s "${summary}" && -s "${prediction}" ]] && "${PYTHON}" - "${summary}" <<'PY'
import json,pathlib,sys
row=json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if row.get("status")=="COMPLETED" and row.get("nan_count")==0 and row.get("oom_count")==0 and row.get("test_access_count")==0 else 1)
PY
    then echo "Skipping completed ${MODE} ${variant} seed=${seed} fold=${fold}"; continue; fi
    log="${OUT_DIR}/logs_private/${MODE}_${variant}_seed${seed}_fold${fold}_gpu${gpu}.log"; mkdir -p "$(dirname "${log}")"
    extra=()
    if [[ "${MODE}" == "smoke" ]]; then extra+=(--epochs 1 --batch-size 1 --eval-batch-size 1 --gradient-accumulation 1 --max-train-rows 32 --max-eval-rows 32 --max-updates 1); fi
    echo "Starting Exp43 ${MODE} ${variant} seed=${seed} fold=${fold} GPU=${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m thesis_exp.exp43_rubimor.train_exp43_groupcv \
      --variant "${variant}" --mode "${MODE}" --seed "${seed}" --fold "${fold}" \
      --model-name-or-path "${MODEL}" --out-dir "${OUT_DIR}" --run-root "${RUN_ROOT}" --artifact-root "${ARTIFACT_ROOT}" \
      --epochs 10 --learning-rate 2e-5 --weight-decay .01 --warmup-ratio .05 --batch-size 4 --eval-batch-size 4 --gradient-accumulation 32 --max-length 2048 \
      --resume "${extra[@]}" 2>&1 | tee "${log}"
  done
}

PIDS=()
for gpu_index in "${!GPUS[@]}"; do
  queue=()
  for job_index in "${!JOBS[@]}"; do (( job_index % ${#GPUS[@]} == gpu_index )) && queue+=("${JOBS[$job_index]}"); done
  run_queue "${GPUS[$gpu_index]}" "${queue[@]}" & PIDS+=("$!")
done
failed=0
for pid in "${PIDS[@]}"; do wait "${pid}" || failed=1; done
[[ ${failed} -eq 0 ]] || { echo "Exp43 ${MODE} matrix failed" >&2; exit 1; }
