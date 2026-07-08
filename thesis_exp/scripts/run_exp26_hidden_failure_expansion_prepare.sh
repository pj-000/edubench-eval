#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

OUT_DIR="${OUT_DIR:-thesis_exp/exp17_low_score_evidence/outputs/exp26_hidden_failure_expansion_seed42}"
TRAIN_JSONL="${TRAIN_JSONL:-thesis_exp/data/splits/question_seed42/train.jsonl}"
DEV_JSONL="${DEV_JSONL:-thesis_exp/data/splits/question_seed42/dev.jsonl}"
TEST_JSONL="${TEST_JSONL:-thesis_exp/data/splits/question_seed42/test.jsonl}"
REASON_ROOT="${REASON_ROOT:-5-grades}"
A0_CANDIDATES="${A0_CANDIDATES:-thesis_exp/exp17_low_score_evidence/outputs/exp17_a0_train_hidden_failure_signals_seed42/train_hidden_failure_candidates.csv}"
A0_HIGH_CONTROLS="${A0_HIGH_CONTROLS:-thesis_exp/exp17_low_score_evidence/outputs/exp17_a0_train_hidden_failure_signals_seed42/train_clean_high_controls.csv}"
EXP21_CANDIDATES="${EXP21_CANDIDATES:-thesis_exp/exp17_low_score_evidence/outputs/exp21_d1_like_risk_annotation_seed42/train_candidates/exp21_train_risk_annotation_candidates.csv}"
R7H_MIXED_JSON="${R7H_MIXED_JSON:-thesis_exp/exp17_low_score_evidence/outputs/exp25_structured_src_dpo_seed42/data/edubench_r7h_structured_src_dpo_train.json}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

cat <<CONFIG
Exp26 hidden-failure evidence expansion data preparation
OUT_DIR=${OUT_DIR}
TRAIN_JSONL=${TRAIN_JSONL}
DEV_JSONL=${DEV_JSONL}
TEST_JSONL=${TEST_JSONL}
REASON_ROOT=${REASON_ROOT}
A0_CANDIDATES=${A0_CANDIDATES}
A0_HIGH_CONTROLS=${A0_HIGH_CONTROLS}
EXP21_CANDIDATES=${EXP21_CANDIDATES}
R7H_MIXED_JSON=${R7H_MIXED_JSON}

This step does not train, does not require GPU, and reads dev/test only for ID leakage guards.
CONFIG

python -m py_compile thesis_exp/exp17_low_score_evidence/prepare_exp26_hidden_failure_expansion.py

python thesis_exp/exp17_low_score_evidence/prepare_exp26_hidden_failure_expansion.py \
  --train-jsonl "${TRAIN_JSONL}" \
  --dev-jsonl "${DEV_JSONL}" \
  --test-jsonl "${TEST_JSONL}" \
  --reason-root "${REASON_ROOT}" \
  --a0-candidates "${A0_CANDIDATES}" \
  --a0-high-controls "${A0_HIGH_CONTROLS}" \
  --exp21-candidates "${EXP21_CANDIDATES}" \
  --r7h-mixed-json "${R7H_MIXED_JSON}" \
  --out-dir "${OUT_DIR}"

python thesis_exp/exp17_low_score_evidence/validate_exp19_llamafactory_data.py \
  --dataset-dir "${OUT_DIR}" \
  --dataset-info "${OUT_DIR}/dataset_info.json" \
  --max-records 0

EXP26_OUT_DIR="${OUT_DIR}" python - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["EXP26_OUT_DIR"])
decision = out / "decision" / "exp26_hidden_failure_expansion_decision.json"
if decision.exists():
    data = json.loads(decision.read_text(encoding="utf-8"))
    print(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True))
else:
    print(f"WARNING: decision file not found: {decision}")
PY

echo "Exp26 hidden-failure evidence expansion data preparation completed."
