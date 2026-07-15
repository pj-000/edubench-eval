#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${EDUBENCH_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"
PYTHON="${PYTHON:-/home/jpang/miniconda3/envs/llama_factory/bin/python}"
MODEL="${EXP46_TEACHER_MODEL_PATH:-/home/jpang/models/modelscope/Qwen/Qwen3-Reranker-4B}"
GPU="${EXP46_SMOKE_GPU:-0}"
SMOKE_ROOT="thesis_exp/runs/exp46_hato_kd_smoke"
rm -rf "${SMOKE_ROOT}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m thesis_exp.exp46_hato_kd.train_exp46_groupcv \
  --variant T1_4B_teacher --fold 0 --model-name-or-path "${MODEL}" --run-root "${SMOKE_ROOT}" \
  --epochs 1 --learning-rate 1e-4 --batch-size 1 --eval-batch-size 1 --gradient-accumulation 1 --max-length 512 \
  --max-train-rows 8 --max-eval-rows 8 --max-updates 1 --smoke
"${PYTHON}" - "${SMOKE_ROOT}/groupcv/T1_4B_teacher/seed_42/fold_0/run_summary.json" <<'PY'
import json,pathlib,sys
row=json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert row["status"]=="COMPLETED" and row["global_step"]==1
assert row["question_key_overlap"]==row["dev_access_count"]==row["test_access_count"]==0
assert row["parameter_counts"]["trainable"] < row["parameter_counts"]["total"]
print("EXP46_TEACHER_SMOKE_PASS")
PY
