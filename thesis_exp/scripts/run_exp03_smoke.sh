#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-llama_factory}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${HOME}/miniconda3/bin/activate" ]]; then
  source "${HOME}/miniconda3/bin/activate" "${CONDA_ENV}"
fi

export MODEL_NAME_OR_PATH
export CUDA_VISIBLE_DEVICES
export REQUIRE_CUDA
export FORMAL_RUN=0
export MAX_TRAIN_SAMPLES=8
export MAX_EVAL_SAMPLES=8
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cat <<CONFIG
Exp3 smoke test
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
REQUIRE_CUDA=${REQUIRE_CUDA}
FORMAL_RUN=${FORMAL_RUN}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES}
CONFIG

python - <<'PY'
import os
import sys

require_cuda = os.environ.get("REQUIRE_CUDA", "1") == "1"
try:
    import torch
except Exception as exc:
    print(f"torch import error: {type(exc).__name__}: {exc}")
    if require_cuda:
        raise SystemExit(1)
    raise SystemExit(0)

cuda_available = torch.cuda.is_available()
print(f"torch.cuda.is_available()={cuda_available}")
print(f"torch.cuda.device_count()={torch.cuda.device_count()}")
if cuda_available:
    print(f"torch.cuda.get_device_name(0)={torch.cuda.get_device_name(0)}")
print(f"torch.version.cuda={torch.version.cuda}")
if require_cuda and not cuda_available:
    raise SystemExit("ERROR: REQUIRE_CUDA=1 but CUDA is unavailable.")
PY

python -m thesis_exp.src.edujudge.exp03.run_exp03 \
  --config thesis_exp/configs/exp03_input_ablation/exp03_smoke_test.yaml \
  --templates A3_question_answer_metric_rubric A4_question_answer_metric_rubric_metadata \
  --mode smoke
