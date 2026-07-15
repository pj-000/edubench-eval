#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${EDUBENCH_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"
PYTHON="${PYTHON:-/home/jpang/miniconda3/envs/llama_factory/bin/python}"
MODEL="${EXP46_STUDENT_MODEL_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
GPU="${EXP46_SMOKE_GPU:-0}"
SMOKE_ROOT="thesis_exp/runs/exp46_hato_kd_smoke"
for variant in K1_standard_kd K2_hato_kd K3_shuffled_hato_control; do
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m thesis_exp.exp46_hato_kd.train_exp46_groupcv \
    --variant "${variant}" --fold 0 --model-name-or-path "${MODEL}" --run-root "${SMOKE_ROOT}" \
    --epochs 1 --learning-rate 2e-5 --batch-size 1 --eval-batch-size 1 --gradient-accumulation 1 --max-length 512 \
    --max-train-rows 8 --max-eval-rows 8 --max-updates 1 --smoke
done
"${PYTHON}" - "${SMOKE_ROOT}" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1])/"groupcv"
for variant in ("K1_standard_kd","K2_hato_kd","K3_shuffled_hato_control"):
    row=json.loads((root/variant/"seed_42/fold_0/run_summary.json").read_text(encoding="utf-8"))
    assert row["status"]=="COMPLETED" and row["global_step"]==1
    assert row["teacher_logit_train_coverage"]==row["train_rows"]==8
    assert row["question_key_overlap"]==row["dev_access_count"]==row["test_access_count"]==0
print("EXP46_STUDENT_SMOKE_PASS")
PY
