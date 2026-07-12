#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES=""
export PYTHONHASHSEED=42

python -m py_compile \
  thesis_exp/exp35_edudart_cal/analyze_exp35a_blind_reviews.py \
  thesis_exp/exp35_edudart_cal/evaluate_exp35a_edudart_cal.py \
  thesis_exp/exp35_edudart_cal/validate_exp35a.py

python thesis_exp/exp35_edudart_cal/analyze_exp35a_blind_reviews.py

python - <<'PY'
import json
from pathlib import Path

path = Path("thesis_exp/exp35_edudart_cal/outputs/exp35a_model_reviewed_qualification_seed42/decision/exp35a_review_decision.json")
decision = json.loads(path.read_text(encoding="utf-8"))
if not decision.get("review_gate_passed"):
    raise SystemExit("Exp35A review gate is not complete/passed; qualification remains locked")
PY

python thesis_exp/exp35_edudart_cal/evaluate_exp35a_edudart_cal.py
python thesis_exp/exp35_edudart_cal/validate_exp35a.py
