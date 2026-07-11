#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${HOME}/miniconda3/bin/activate" ]]; then
  source "${HOME}/miniconda3/bin/activate" "${CONDA_ENV:-llama_factory}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_LIST="${GPU_LIST:-0}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
OUT_DIR="thesis_exp/exp17_low_score_evidence/outputs/exp27q_safe16_multiseed_seed42_44"
RUN_DIR="thesis_exp/runs/exp27q_safe16_smoke/v3_safe16_original_low_anchor/seed_42"
ARTIFACT_DIR="thesis_exp/artifacts/exp27q_safe16_smoke/v3_safe16_original_low_anchor/seed_42"

"${PYTHON_BIN}" thesis_exp/exp17_low_score_evidence/prepare_exp27q_v3_safe16_dataset.py
"${PYTHON_BIN}" thesis_exp/exp17_low_score_evidence/validate_exp27q_safe16_dataset.py --out-dir "${OUT_DIR}"

"${PYTHON_BIN}" - <<'PY'
import torch
from thesis_exp.src.edujudge.exp27p.train_exp27p_soft_target_reranker import global_scaled_soft_ce
logits = torch.zeros((2, 5), requires_grad=True)
targets = torch.tensor([[1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.]])
weights = torch.tensor([0., 1.])
loss = global_scaled_soft_ce(logits, targets, weights, dataset_rows=2, global_weight_sum=1.)
loss.backward()
assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
PY

rm -rf "${RUN_DIR}" "${ARTIFACT_DIR}"
CUDA_VISIBLE_DEVICES="${GPU_LIST%% *}" "${PYTHON_BIN}" -m thesis_exp.src.edujudge.exp27q.train_exp27q_safe16 \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --run_dir "${RUN_DIR}" \
  --checkpoint_output_dir "${ARTIFACT_DIR}" \
  --seed 42 \
  --max_train_samples 32 \
  --max_eval_samples 32 \
  --num_train_epochs 0.05 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --log_steps 1

"${PYTHON_BIN}" - "${RUN_DIR}/run_summary.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
assert summary["status"] == "COMPLETED"
assert summary["checkpoint_reload_pass"] is True
assert summary["test_access_count"] == 0
print("Exp27Q smoke PASS")
PY
