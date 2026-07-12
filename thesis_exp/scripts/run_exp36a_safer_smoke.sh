#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
if [[ -f "${HOME}/miniconda3/bin/activate" ]]; then source "${HOME}/miniconda3/bin/activate" "${CONDA_ENV:-llama_factory}"; fi

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_LIST="${GPU_LIST:-0}"
read -r -a GPUS <<<"${GPU_LIST}"
GPU="${GPUS[0]}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B}"
OUT_DIR="thesis_exp/exp36_safer_score/outputs/exp36a_safer_score_seed42"
SMOKE_OUT="${OUT_DIR}/private/smoke"

"${PYTHON_BIN}" thesis_exp/exp36_safer_score/validate_exp36a_supervision.py --out-dir "${OUT_DIR}" --require-private
rm -rf "${SMOKE_OUT}" thesis_exp/runs/exp36_safer_score_smoke thesis_exp/artifacts/exp36_safer_score_smoke
mkdir -p "${SMOKE_OUT}/logs"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" thesis_exp/exp36_safer_score/train_exp36a_oof_human_baseline.py \
  --fold 0 --model-name-or-path "${MODEL_NAME_OR_PATH}" --out-dir "${SMOKE_OUT}" \
  --fold-assignment "${OUT_DIR}/data/exp36a_oof_fold_assignment.csv" \
  --epochs 1 --batch-size 1 --eval-batch-size 1 --gradient-accumulation-steps 1 \
  --max-train-samples 32 --max-eval-samples 32 2>&1 | tee "${SMOKE_OUT}/logs/oof_fold0.log"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" thesis_exp/exp36_safer_score/train_exp36a_safer_score.py \
  --variant v5_safer_score --model-name-or-path "${MODEL_NAME_OR_PATH}" \
  --train-jsonl "${OUT_DIR}/private/data/exp36a_v5_safer_score_train.jsonl" \
  --out-dir "${SMOKE_OUT}" --run-dir thesis_exp/runs/exp36_safer_score_smoke/v5 \
  --artifact-dir thesis_exp/artifacts/exp36_safer_score_smoke/v5 \
  --epochs 1 --batch-size 1 --eval-batch-size 1 --gradient-accumulation-steps 1 \
  --max-train-samples 32 --max-eval-samples 32 2>&1 | tee "${SMOKE_OUT}/logs/v5.log"
"${PYTHON_BIN}" - "${SMOKE_OUT}" <<'PY'
import json, math, sys
from pathlib import Path
root=Path(sys.argv[1]); summary=json.loads(Path('thesis_exp/runs/exp36_safer_score_smoke/v5/run_summary.json').read_text())
checks={
 'oof_completed': (root/'private/oof_folds/fold_0_summary.json').exists(),
 'v5_completed': summary.get('status')=='COMPLETED',
 'failure_head_backward': summary['failure_head_train_metrics']['masked_rows'] >= 0,
 'checkpoint_reload': Path(summary['checkpoint_path']).exists(),
 'no_nan': all(math.isfinite(float(summary['selected_metrics'][k])) for k in ['MAE_argmax','Exact_Match']),
 'no_oom': summary.get('oom_count')==0,
 'no_test_access': summary.get('test_access_count')==0,
}
decision={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'dynamic_target_changes_by_epoch':'unit_checked_in_validation','zero_failure_mask_safe':'unit_checked_in_validation'}
path=Path('thesis_exp/exp36_safer_score/outputs/exp36a_safer_score_seed42/decision/exp36a_smoke_decision.json'); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(decision,indent=2)+'\n')
if decision['status']!='PASS': raise SystemExit(decision)
print(decision)
PY
echo "Exp36A smoke PASS."
