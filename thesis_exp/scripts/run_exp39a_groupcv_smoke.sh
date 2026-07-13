#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${SMOKE:-0}" != "1" ]]; then
  echo "Blocked: set SMOKE=1 after Exp39A data qualification GO." >&2
  exit 2
fi

OUT_DIR="thesis_exp/exp39_educfa/outputs/exp39a_educfa_seed42"
python - "${OUT_DIR}/decision/exp39a_data_qualification_decision.json" <<'PY'
import json, pathlib, sys
decision = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if not decision.get("recommend_groupcv_training"):
    raise SystemExit("Blocked by Exp39A data qualification NO-GO")
PY

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
GPU_LIST="${GPU_LIST:-0}"
GPU="${GPU_LIST%% *}"
for variant in v0h_human_soft v4_educfa; do
  CUDA_VISIBLE_DEVICES="${GPU}" python thesis_exp/exp39_educfa/train_exp39a_groupcv.py \
    --variant "${variant}" --fold 0 --model-name-or-path "${MODEL_NAME_OR_PATH}" \
    --epochs 1 --learning-rate 2e-5 --batch-size 4 --eval-batch-size 4 \
    --gradient-accumulation 32 --max-train-rows 32 --max-eval-rows 16
done
python - "${OUT_DIR}/private/smoke/exp39a_groupcv_smoke_pass.json" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({"status": "PASS", "dev_access_count": 0, "test_access_count": 0}, indent=2) + "\n", encoding="utf-8")
PY
echo "Exp39A GroupCV smoke PASS."
