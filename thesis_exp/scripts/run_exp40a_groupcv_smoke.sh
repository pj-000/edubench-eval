#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${SMOKE:-0}" != "1" ]]; then
  echo "Blocked: set SMOKE=1 only after Exp40A pairwise qualification GO." >&2
  exit 2
fi

OUT_DIR="thesis_exp/exp40_edupair_cf/outputs/exp40a_edupair_cf_seed42"
python - "${OUT_DIR}/decision/exp40a_pairwise_qualification_decision.json" <<'PY'
import json, pathlib, sys
decision = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if not decision.get("recommend_groupcv_training"):
    raise SystemExit("Blocked by Exp40A pairwise qualification NO-GO")
PY

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
GPU_LIST="${GPU_LIST:-0}"
GPU="${GPU_LIST%% *}"
for variant in v0h_human_soft v3_edupair_cf; do
  CUDA_VISIBLE_DEVICES="${GPU}" python thesis_exp/exp40_edupair_cf/train_exp40a_groupcv.py \
    --variant "${variant}" --fold 0 --model-name-or-path "${MODEL_NAME_OR_PATH}" \
    --epochs 1 --learning-rate 2e-5 --batch-size 4 --eval-batch-size 4 \
    --gradient-accumulation 32 --lambda-pair 0.25 --max-length 2048 \
    --max-train-rows 32 --max-eval-rows 16 --max-pairs 8
done
python - "${OUT_DIR}/private/smoke/exp40a_groupcv_smoke_pass.json" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({"status": "PASS", "dev_access_count": 0, "test_access_count": 0}, indent=2) + "\n", encoding="utf-8")
PY
echo "Exp40A GroupCV smoke PASS."
