#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${EDUBENCH_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"
PYTHON="${PYTHON:-/home/jpang/miniconda3/envs/llama_factory/bin/python}"
[[ -x "${PYTHON}" ]] || PYTHON="$(command -v python3)"
MODEL="${RUBIMOR_MODEL_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
GPU="${SMOKE_GPU:-${GPU_LIST%% *}}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m thesis_exp.exp45_dopr_head.train_or_restore_exp45a_e4_encoders \
  --fold 0 --mode smoke --model-name-or-path "${MODEL}" \
  --epochs 1 --max-train-rows 32 --max-updates 1

"${PYTHON}" - <<'PY'
import json
from pathlib import Path
path = Path("thesis_exp/exp45_dopr_head/outputs/exp45a_dopr_seed42/private/encoders/smoke/fold_0/encoder_summary.json")
value = json.loads(path.read_text())
assert value["status"] == "COMPLETED" and value["save_reload"] == "PASS"
assert value["oom_count"] == value["nan_count"] == 0
assert value["dev_access_count"] == value["test_access_count"] == 0
print(json.dumps({"status": "ENCODER_SMOKE_GO", "fold": 0}, sort_keys=True))
PY

# The smoke checkpoint is 2+ GiB and is never used by the formal run.
rm -f thesis_exp/exp45_dopr_head/outputs/exp45a_dopr_seed42/private/encoders/smoke/fold_0/final_encoder_head.pt
