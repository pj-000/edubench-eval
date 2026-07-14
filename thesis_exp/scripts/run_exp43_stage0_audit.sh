#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO_ROOT="${EDUBENCH_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"; cd "${REPO_ROOT}"
PYTHON="${PYTHON:-/home/jpang/miniconda3/envs/llama_factory/bin/python}"; MODEL="${RUBIMOR_MODEL_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"; GPU_LIST="${GPU_LIST:-0 1 2 3}"
"${PYTHON}" -m thesis_exp.exp43_rubimor.audit_exp43_data_and_splits
"${PYTHON}" - <<'PY'
import json
row=json.load(open("thesis_exp/exp43_rubimor/outputs/exp43_rubimor_preregistered/decision/exp43_stage0_decision.json"))
raise SystemExit(0 if row["status"]=="GO" else 2)
PY
"${PYTHON}" -m thesis_exp.exp43_rubimor.prepare_exp43_datasets --model-name-or-path "${MODEL}"
"${PYTHON}" -m thesis_exp.exp43_rubimor.prepare_exp43_pairs
CUDA_VISIBLE_DEVICES="${GPU_LIST%% *}" "${PYTHON}" -m thesis_exp.exp43_rubimor.audit_exp43_loss_scales --model-name-or-path "${MODEL}"
