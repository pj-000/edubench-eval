#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${EDUBENCH_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-/home/jpang/miniconda3/envs/llama_factory/bin/python}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
VARIANTS_STRING="${EXP46_VARIANTS:?EXP46_VARIANTS is required}"
FOLDS_STRING="${FOLDS:-0 1 2 3 4}"
TEACHER_MODEL="${EXP46_TEACHER_MODEL_PATH:-/home/jpang/models/modelscope/Qwen/Qwen3-Reranker-4B}"
STUDENT_MODEL="${EXP46_STUDENT_MODEL_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
OUT="thesis_exp/exp46_hato_kd/outputs/exp46a_hato_seed42"
RUN_ROOT="thesis_exp/runs/exp46_hato_kd"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
read -r -a GPUS <<< "${GPU_LIST}"
read -r -a VARIANTS <<< "${VARIANTS_STRING}"
read -r -a FOLDS_ARRAY <<< "${FOLDS_STRING}"
[[ ${#GPUS[@]} -gt 0 ]] || { echo "GPU_LIST is empty" >&2; exit 2; }

JOBS=()
for variant in "${VARIANTS[@]}"; do
  for fold in "${FOLDS_ARRAY[@]}"; do JOBS+=("${variant}:${fold}"); done
done
echo "Exp46A matrix: ${#JOBS[@]} runs; variants=${VARIANTS_STRING}; GPUs=${GPU_LIST}"

run_queue() {
  local gpu="$1"; shift
  local job variant fold model lr run summary prediction log fingerprint
  for job in "$@"; do
    variant="${job%%:*}"; fold="${job##*:}"
    if [[ "${variant}" == "T1_4B_teacher" ]]; then model="${TEACHER_MODEL}"; lr="1e-4"; else model="${STUDENT_MODEL}"; lr="2e-5"; fi
    [[ -d "${model}" ]] || { echo "Missing local model: ${model}" >&2; exit 2; }
    run="${RUN_ROOT}/groupcv/${variant}/seed_42/fold_${fold}"
    summary="${run}/run_summary.json"; prediction="${run}/heldout_predictions.jsonl"
    log="${OUT}/logs_private/${variant}_fold${fold}_gpu${gpu}.log"; mkdir -p "$(dirname "${log}")"
    fingerprint=$("${PYTHON}" -m thesis_exp.exp46_hato_kd.train_exp46_groupcv \
      --variant "${variant}" --fold "${fold}" --model-name-or-path "${model}" --out-dir "${OUT}" --run-root "${RUN_ROOT}" \
      --epochs 10 --learning-rate "${lr}" --weight-decay .01 --warmup-ratio .05 --batch-size 4 --eval-batch-size 4 --gradient-accumulation 32 --max-length 2048 \
      --fingerprint-only | "${PYTHON}" -c 'import json,sys; print(json.load(sys.stdin)["run_fingerprint"])')
    if [[ "${SKIP_COMPLETED}" == "1" && -s "${summary}" && -s "${prediction}" ]] && "${PYTHON}" - "${summary}" "${fingerprint}" <<'PY'
import json,pathlib,sys
row=json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
valid=(row.get("status")=="COMPLETED" and row.get("fixed_final_epoch")==10 and row.get("question_key_overlap")==0 and row.get("nan_count")==0 and row.get("oom_count")==0 and row.get("dev_access_count")==0 and row.get("test_access_count")==0 and row.get("run_fingerprint")==sys.argv[2])
raise SystemExit(0 if valid else 1)
PY
    then
      echo "Skipping hash-matched Exp46A ${variant} fold=${fold}"
      continue
    fi
    echo "Starting Exp46A ${variant} fold=${fold} GPU=${gpu}; log=${log}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m thesis_exp.exp46_hato_kd.train_exp46_groupcv \
      --variant "${variant}" --fold "${fold}" --model-name-or-path "${model}" --out-dir "${OUT}" --run-root "${RUN_ROOT}" \
      --epochs 10 --learning-rate "${lr}" --weight-decay .01 --warmup-ratio .05 --batch-size 4 --eval-batch-size 4 --gradient-accumulation 32 --max-length 2048 \
      >"${log}" 2>&1
  done
}

PIDS=()
for gpu_index in "${!GPUS[@]}"; do
  queue=()
  for job_index in "${!JOBS[@]}"; do
    (( job_index % ${#GPUS[@]} == gpu_index )) && queue+=("${JOBS[$job_index]}")
  done
  run_queue "${GPUS[$gpu_index]}" "${queue[@]}" & PIDS+=("$!")
done
failed=0
for pid in "${PIDS[@]}"; do wait "${pid}" || failed=1; done
[[ ${failed} -eq 0 ]] || { echo "Exp46A matrix failed" >&2; exit 1; }
