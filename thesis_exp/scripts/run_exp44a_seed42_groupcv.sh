#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${EDUBENCH_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-/home/jpang/miniconda3/envs/llama_factory/bin/python}"
MODEL="${RUBIMOR_MODEL_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
OUT="thesis_exp/exp44_taco_score/outputs/exp44a_taco_seed42"
RUN_ROOT="thesis_exp/runs/exp44_taco_score"
ARTIFACT_ROOT="thesis_exp/artifacts/exp44_taco_score"
read -r -a GPUS <<< "${GPU_LIST}"
VARIANTS=(C0_E4_baseline C1_balanced_plain_contrastive C2_TACO C3_shuffled_margin_control)
JOBS=()
for variant in "${VARIANTS[@]}"; do
  for fold in 0 1 2 3 4; do JOBS+=("${variant}:${fold}"); done
done
echo "Exp44A formal matrix: ${#JOBS[@]} runs; GPUs=${GPU_LIST}"

run_queue() {
  local gpu="$1"; shift
  local job variant fold run summary prediction log fingerprint
  for job in "$@"; do
    variant="${job%%:*}"; fold="${job##*:}"
    run="${RUN_ROOT}/groupcv/${variant}/seed_42/fold_${fold}"
    summary="${run}/run_summary.json"; prediction="${run}/heldout_predictions.jsonl"
    log="${OUT}/logs_private/groupcv_${variant}_seed42_fold${fold}_gpu${gpu}.log"; mkdir -p "$(dirname "${log}")"
    fingerprint=$("${PYTHON}" -m thesis_exp.exp44_taco_score.train_exp44a_groupcv \
      --variant "${variant}" --mode groupcv --fold "${fold}" --seed 42 \
      --model-name-or-path "${MODEL}" --out-dir "${OUT}" --run-root "${RUN_ROOT}" --artifact-root "${ARTIFACT_ROOT}" \
      --fingerprint-only | "${PYTHON}" -c 'import json,sys; print(json.load(sys.stdin)["run_fingerprint"])')
    if [[ "${SKIP_COMPLETED}" == 1 && -s "${summary}" && -s "${prediction}" ]] && "${PYTHON}" - "${summary}" "${fingerprint}" <<'PY'
import json,pathlib,sys
row=json.loads(pathlib.Path(sys.argv[1]).read_text())
valid=row.get("status")=="COMPLETED" and row.get("fixed_final_epoch")==10 and row.get("run_fingerprint")==sys.argv[2] and row.get("nan_count")==row.get("oom_count")==row.get("dev_access_count")==row.get("test_access_count")==0
raise SystemExit(0 if valid else 1)
PY
    then echo "Skipping hash-matched ${variant} fold=${fold}"; continue; fi
    echo "Starting Exp44A ${variant} fold=${fold} GPU=${gpu}; log=${log}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m thesis_exp.exp44_taco_score.train_exp44a_groupcv \
      --variant "${variant}" --mode groupcv --fold "${fold}" --seed 42 \
      --model-name-or-path "${MODEL}" --out-dir "${OUT}" --run-root "${RUN_ROOT}" --artifact-root "${ARTIFACT_ROOT}" \
      --epochs 10 --learning-rate 2e-5 --weight-decay .01 --warmup-ratio .05 \
      --batch-size 4 --eval-batch-size 4 --gradient-accumulation 32 --max-length 2048 --resume \
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
[[ ${failed} -eq 0 ]] || { echo "Exp44A formal matrix failed" >&2; exit 1; }

