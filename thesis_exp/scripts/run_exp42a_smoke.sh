#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${SMOKE:-0}" != "1" ]]; then
  echo "Blocked: set SMOKE=1 for the preregistered Exp42A smoke." >&2
  exit 2
fi

PYTHON="${PYTHON:-python}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
GPU_LIST="${GPU_LIST:-0}"
GPU="${GPU_LIST%% *}"
OUT_DIR="thesis_exp/exp42_rubidist/outputs/exp42a_rubidist_multiseed"
SMOKE_RUN_ROOT="thesis_exp/runs/exp42_rubidist_smoke"
SMOKE_ARTIFACT_ROOT="thesis_exp/artifacts/exp42_rubidist_smoke"

for path in \
  "${OUT_DIR}/private/data/exp42a_v11_soft_raw_rubric.jsonl" \
  "${OUT_DIR}/private/data/exp42a_groupcv_fold_assignment.csv"; do
  [[ -s "${path}" ]] || { echo "Missing prepared artifact: ${path}" >&2; exit 2; }
done

rm -rf "${SMOKE_RUN_ROOT}" "${SMOKE_ARTIFACT_ROOT}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" thesis_exp/exp42_rubidist/train_exp42a_groupcv.py \
  --variant v11_soft_raw_rubric --seed 42 --fold 0 \
  --model-name-or-path "${MODEL_NAME_OR_PATH}" \
  --out-dir "${OUT_DIR}" --run-root "${SMOKE_RUN_ROOT}" --artifact-root "${SMOKE_ARTIFACT_ROOT}" \
  --epochs 1 --learning-rate 2e-5 --weight-decay 0.01 --warmup-ratio 0.05 \
  --batch-size 1 --eval-batch-size 1 --gradient-accumulation 1 --max-length 2048 \
  --max-train-rows 32 --max-eval-rows 32 --max-updates 1 --smoke-save-reload

"${PYTHON}" - "${SMOKE_RUN_ROOT}/v11_soft_raw_rubric/seed_42/fold_0/run_summary.json" \
  "${OUT_DIR}/private/exp42a_smoke_pass.json" <<'PY'
import json, pathlib, sys
summary_path, output_path = map(pathlib.Path, sys.argv[1:])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
passed = (
    summary.get("status") == "COMPLETED"
    and summary.get("global_step") == 1
    and str(summary.get("smoke_save_reload", "")).startswith("PASS:")
    and summary.get("nan_count") == 0
    and summary.get("oom_count") == 0
    and summary.get("dev_access_count") == 0
    and summary.get("test_access_count") == 0
)
if not passed:
    raise SystemExit(f"Exp42A smoke failed: {summary}")
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps({"status": "PASS", "variant": "v11_soft_raw_rubric", "seed": 42, "fold": 0, "hard_soft_loss_code_path": "shared_standard_cross_entropy", "save_reload": summary["smoke_save_reload"], "nan_count": 0, "oom_count": 0, "dev_access_count": 0, "test_access_count": 0}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo "Exp42A smoke PASS."
