#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${SMOKE:-0}" != "1" ]]; then
  echo "Blocked: set SMOKE=1 only after compiler qualification GO." >&2
  exit 2
fi
OUT_DIR="thesis_exp/exp41_rubric_bridge/outputs/exp41a_rubric_bridge_groupcv_seed42"
python - "${OUT_DIR}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
decision = json.loads((root / "decision/exp41a_compiler_qualification_decision.json").read_text(encoding="utf-8"))
if not decision.get("recommend_groupcv_training"):
    raise SystemExit("Blocked by Exp41A compiler qualification NO-GO")
for path in (root / "private/data/exp41a_v0h_human_soft.jsonl", root / "private/data/exp41a_v3_rubric_bridge.jsonl", root / "private/data/exp41a_groupcv_fold_assignment.csv"):
    if not path.exists():
        raise SystemExit(f"Missing prepared artifact: {path}")
PY

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
GPU_LIST="${GPU_LIST:-0}"
GPU="${GPU_LIST%% *}"
for variant in v0h_human_soft v3_rubric_bridge; do
  CUDA_VISIBLE_DEVICES="${GPU}" python thesis_exp/exp41_rubric_bridge/train_exp41a_groupcv.py \
    --variant "${variant}" --fold 0 --model-name-or-path "${MODEL_NAME_OR_PATH}" \
    --epochs 1 --learning-rate 2e-5 --batch-size 4 --eval-batch-size 4 \
    --gradient-accumulation 32 --max-length 2048 --max-train-rows 32 --max-eval-rows 16
  smoke_run="${OUT_DIR}/private/smoke/groupcv_predictions/${variant}/fold_0"
  rm -rf "${smoke_run}"
  mkdir -p "$(dirname "${smoke_run}")"
  mv "${OUT_DIR}/private/groupcv_predictions/${variant}/fold_0" "${smoke_run}"
done
python - "${OUT_DIR}/private/smoke/exp41a_groupcv_smoke_pass.json" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1]); path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({"status": "PASS", "variants": ["v0h_human_soft", "v3_rubric_bridge"], "dev_access_count": 0, "test_access_count": 0}, indent=2) + "\n", encoding="utf-8")
PY
echo "Exp41A GroupCV smoke PASS."
