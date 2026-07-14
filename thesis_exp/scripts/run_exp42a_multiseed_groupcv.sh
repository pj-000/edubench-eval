#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${RUN_FORMAL:-0}" != "1" ]]; then
  echo "Blocked: set RUN_FORMAL=1 after Exp42A prepare and smoke PASS." >&2
  exit 2
fi

PYTHON="${PYTHON:-python}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
SEEDS_STRING="${SEEDS:-42 43 44}"
OUT_DIR="thesis_exp/exp42_rubidist/outputs/exp42a_rubidist_multiseed"
RUN_ROOT="thesis_exp/runs/exp42_rubidist"
ARTIFACT_ROOT="thesis_exp/artifacts/exp42_rubidist"
read -r -a GPUS <<< "${GPU_LIST}"
read -r -a RUN_SEEDS <<< "${SEEDS_STRING}"
[[ "${#GPUS[@]}" -gt 0 ]] || { echo "GPU_LIST is empty" >&2; exit 2; }

"${PYTHON}" - "${OUT_DIR}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
smoke = root / "private/exp42a_smoke_pass.json"
fold_hash = root / "hashes/exp42a_fold_hashes.json"
if not smoke.exists() or json.loads(smoke.read_text(encoding="utf-8")).get("status") != "PASS":
    raise SystemExit("Blocked: Exp42A smoke PASS is missing")
if not fold_hash.exists() or not json.loads(fold_hash.read_text(encoding="utf-8")).get("fold_hash_equal_exp41"):
    raise SystemExit("Blocked: Exp42A fold hash does not equal Exp41")
PY

VARIANTS=(
  v00_hard_no_rubric
  v01_soft_no_rubric
  v10_hard_raw_rubric
  v11_soft_raw_rubric
)
JOBS=()
for variant in "${VARIANTS[@]}"; do
  for seed in "${RUN_SEEDS[@]}"; do
    for fold in 0 1 2 3 4; do
      JOBS+=("${variant}:${seed}:${fold}")
    done
  done
done
echo "Exp42A formal matrix: ${#JOBS[@]} runs; GPUs=${GPU_LIST}; seeds=${SEEDS_STRING}"

run_queue() {
  local gpu="$1"
  shift
  local job variant rest seed fold summary prediction log
  for job in "$@"; do
    variant="${job%%:*}"
    rest="${job#*:}"
    seed="${rest%%:*}"
    fold="${rest##*:}"
    summary="${RUN_ROOT}/${variant}/seed_${seed}/fold_${fold}/run_summary.json"
    prediction="${RUN_ROOT}/${variant}/seed_${seed}/fold_${fold}/heldout_predictions.jsonl"
    if [[ "${SKIP_COMPLETED:-1}" == "1" && -s "${summary}" && -s "${prediction}" ]]; then
      if "${PYTHON}" - "${summary}" "${prediction}" <<'PY'
import json, pathlib, sys
summary = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
rows = sum(1 for line in pathlib.Path(sys.argv[2]).open(encoding="utf-8") if line.strip())
valid = summary.get("status") == "COMPLETED" and summary.get("fixed_final_epoch") == 10 and summary.get("formal") is True and rows == summary.get("heldout_rows") and summary.get("nan_count") == 0 and summary.get("oom_count") == 0
raise SystemExit(0 if valid else 1)
PY
      then
        echo "Skipping completed Exp42A ${variant} seed=${seed} fold=${fold}"
        continue
      fi
    fi
    log="${OUT_DIR}/logs_private/train_${variant}_seed${seed}_fold${fold}_gpu${gpu}.log"
    mkdir -p "$(dirname "${log}")"
    echo "Starting Exp42A ${variant} seed=${seed} fold=${fold} on GPU ${gpu}; log=${log}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" thesis_exp/exp42_rubidist/train_exp42a_groupcv.py \
      --variant "${variant}" --seed "${seed}" --fold "${fold}" \
      --model-name-or-path "${MODEL_NAME_OR_PATH}" --out-dir "${OUT_DIR}" \
      --run-root "${RUN_ROOT}" --artifact-root "${ARTIFACT_ROOT}" \
      --epochs 10 --learning-rate 2e-5 --weight-decay 0.01 --warmup-ratio 0.05 \
      --batch-size 4 --eval-batch-size 4 --gradient-accumulation 32 --max-length 2048 \
      2>&1 | tee "${log}"
  done
}

PIDS=()
for gpu_index in "${!GPUS[@]}"; do
  queue=()
  for job_index in "${!JOBS[@]}"; do
    if (( job_index % ${#GPUS[@]} == gpu_index )); then
      queue+=("${JOBS[$job_index]}")
    fi
  done
  run_queue "${GPUS[$gpu_index]}" "${queue[@]}" &
  PIDS+=("$!")
done
failed=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "At least one Exp42A GPU queue failed." >&2
  exit 1
fi

SEEDS="${SEEDS_STRING}" bash thesis_exp/scripts/run_exp42a_collect.sh
