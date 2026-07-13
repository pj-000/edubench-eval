#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${RUN_FORMAL:-0}" != "1" ]]; then
  echo "Blocked: set RUN_FORMAL=1 after pair qualification GO and smoke PASS." >&2
  exit 2
fi

OUT_DIR="thesis_exp/exp40_edupair_cf/outputs/exp40a_edupair_cf_seed42"
python - "${OUT_DIR}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
decision = json.loads((root / "decision/exp40a_pairwise_qualification_decision.json").read_text(encoding="utf-8"))
if not decision.get("recommend_groupcv_training"):
    raise SystemExit("Blocked by Exp40A pairwise qualification NO-GO")
smoke_path = root / "private/smoke/exp40a_groupcv_smoke_pass.json"
if not smoke_path.exists():
    raise SystemExit("Blocked: Exp40A GroupCV smoke result is missing")
smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
if smoke.get("status") != "PASS":
    raise SystemExit("Blocked: Exp40A GroupCV smoke did not pass")
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
  v1_human_real_pairs
  v2_unverified_counterfactual_pairs
  v3_edupair_cf
  v4_shuffled_pair_alignment
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
    echo "Starting Exp40A ${variant} fold ${fold} on GPU ${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" python thesis_exp/exp40_edupair_cf/train_exp40a_groupcv.py \
      --variant "${variant}" --fold "${fold}" --model-name-or-path "${MODEL_NAME_OR_PATH}" \
      --epochs 10 --learning-rate 2e-5 --batch-size 4 --eval-batch-size 4 \
      --gradient-accumulation 32 --weight-decay 0.01 --warmup-ratio 0.05 \
      --lambda-pair 0.25 --max-length 2048
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
  echo "At least one Exp40A GPU queue failed." >&2
  exit 1
fi
python thesis_exp/exp40_edupair_cf/collect_exp40a_groupcv.py --out-dir "${OUT_DIR}"
python thesis_exp/exp40_edupair_cf/bootstrap_exp40a_groupcv.py --out-dir "${OUT_DIR}" --resamples 5000
