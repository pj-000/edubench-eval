#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${RUN_FORMAL:-0}" != "1" ]]; then
  echo "Blocked: set RUN_FORMAL=1 after data GO and smoke PASS." >&2
  exit 2
fi

OUT_DIR="thesis_exp/exp39_educfa/outputs/exp39a_educfa_seed42"
python - "${OUT_DIR}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
decision = json.loads((root / "decision/exp39a_data_qualification_decision.json").read_text(encoding="utf-8"))
smoke = json.loads((root / "private/smoke/exp39a_groupcv_smoke_pass.json").read_text(encoding="utf-8"))
if not decision.get("recommend_groupcv_training") or smoke.get("status") != "PASS":
    raise SystemExit("Blocked: Exp39A data GO and smoke PASS are both required")
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
  v1_matched_real_low_oversampling
  v2_unverified_counterfactual
  v3_generic_corruption
  v4_educfa
  v5_shuffled_counterfactual
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
  local job variant fold
  for job in "$@"; do
    variant="${job%%:*}"
    fold="${job##*:}"
    echo "Starting Exp39A ${variant} fold ${fold} on GPU ${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" python thesis_exp/exp39_educfa/train_exp39a_groupcv.py \
      --variant "${variant}" --fold "${fold}" --model-name-or-path "${MODEL_NAME_OR_PATH}" \
      --epochs 10 --learning-rate 2e-5 --batch-size 4 --eval-batch-size 4 \
      --gradient-accumulation 32 --weight-decay 0.01 --warmup-ratio 0.05 --max-length 2048
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
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "At least one Exp39A GPU queue failed." >&2
  exit 1
fi
python thesis_exp/exp39_educfa/collect_exp39a_groupcv.py
python thesis_exp/exp39_educfa/bootstrap_exp39a_groupcv.py --resamples 5000
