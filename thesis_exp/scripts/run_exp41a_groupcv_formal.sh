#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${RUN_FORMAL:-0}" != "1" ]]; then
  echo "Blocked: set RUN_FORMAL=1 after qualification GO and smoke PASS." >&2
  exit 2
fi
OUT_DIR="thesis_exp/exp41_rubric_bridge/outputs/exp41a_rubric_bridge_groupcv_seed42"
python - "${OUT_DIR}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
decision = json.loads((root / "decision/exp41a_compiler_qualification_decision.json").read_text(encoding="utf-8"))
if not decision.get("recommend_groupcv_training"):
    raise SystemExit("Blocked by Exp41A compiler qualification NO-GO")
smoke = root / "private/smoke/exp41a_groupcv_smoke_pass.json"
if not smoke.exists() or json.loads(smoke.read_text(encoding="utf-8")).get("status") != "PASS":
    raise SystemExit("Blocked: Exp41A GroupCV smoke PASS is missing")
PY

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "GPU_LIST must contain at least one GPU." >&2
  exit 2
fi
VARIANTS=(
  v0h_human_soft
  v1_raw_rubric
  v2_deterministic_checklist
  v3_rubric_bridge
  v4_shuffled_compiled_rubric
  v5_human_soft_logit_adjustment
)
JOBS=()
for variant in "${VARIANTS[@]}"; do
  for fold in 0 1 2 3 4; do
    JOBS+=("${variant}:${fold}")
  done
done

run_queue() {
  local gpu="$1"
  shift
  local job variant fold summary prediction log
  for job in "$@"; do
    variant="${job%%:*}"
    fold="${job##*:}"
    summary="${OUT_DIR}/private/groupcv_predictions/${variant}/fold_${fold}/run_summary.json"
    prediction="${OUT_DIR}/private/groupcv_predictions/${variant}/fold_${fold}/heldout_predictions.jsonl"
    if [[ "${SKIP_COMPLETED:-1}" == "1" && -s "${summary}" && -s "${prediction}" ]]; then
      if python - "${summary}" "${prediction}" <<'PY'
import json, pathlib, sys
summary = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
prediction_rows = sum(1 for line in pathlib.Path(sys.argv[2]).open(encoding="utf-8") if line.strip())
raise SystemExit(0 if summary.get("status") == "COMPLETED" and summary.get("fixed_final_epoch") == 10 and prediction_rows == summary.get("heldout_rows") else 1)
PY
      then
        echo "Skipping completed Exp41A ${variant} fold ${fold}"
        continue
      fi
    fi
    log="${OUT_DIR}/logs_private/train_${variant}_fold_${fold}_gpu${gpu}.log"
    mkdir -p "$(dirname "${log}")"
    echo "Starting Exp41A ${variant} fold ${fold} on GPU ${gpu}; log=${log}"
    CUDA_VISIBLE_DEVICES="${gpu}" python thesis_exp/exp41_rubric_bridge/train_exp41a_groupcv.py \
      --variant "${variant}" --fold "${fold}" --model-name-or-path "${MODEL_NAME_OR_PATH}" \
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
  echo "At least one Exp41A GPU queue failed." >&2
  exit 1
fi
python thesis_exp/exp41_rubric_bridge/collect_exp41a_groupcv.py --out-dir "${OUT_DIR}"
python thesis_exp/exp41_rubric_bridge/bootstrap_exp41a_groupcv.py --out-dir "${OUT_DIR}" --resamples 5000
