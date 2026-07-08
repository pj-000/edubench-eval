#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

OUT_DIR="${OUT_DIR:-thesis_exp/exp17_low_score_evidence/outputs/exp27e_provider_bias_conflict_analysis_seed42}"
EXP27D_DIR="${EXP27D_DIR:-thesis_exp/exp17_low_score_evidence/outputs/exp27d_teacher_audit_v4_seed42}"
TRAIN_JSONL="${TRAIN_JSONL:-thesis_exp/data/splits/question_seed42/train.jsonl}"
DEV_JSONL="${DEV_JSONL:-thesis_exp/data/splits/question_seed42/dev.jsonl}"
TEST_JSONL="${TEST_JSONL:-thesis_exp/data/splits/question_seed42/test.jsonl}"

cat <<CONFIG
Run Exp27E provider-bias and conflict-adjudication analysis
OUT_DIR=${OUT_DIR}
EXP27D_DIR=${EXP27D_DIR}
TRAIN_JSONL=${TRAIN_JSONL}
DEV_JSONL=${DEV_JSONL}
TEST_JSONL=${TEST_JSONL}
CONFIG

python thesis_exp/exp17_low_score_evidence/analyze_exp27e_provider_bias_and_conflicts.py \
  --exp27d-dir "${EXP27D_DIR}" \
  --out-dir "${OUT_DIR}" \
  --train-jsonl "${TRAIN_JSONL}" \
  --dev-jsonl "${DEV_JSONL}" \
  --test-jsonl "${TEST_JSONL}"

cp thesis_exp/exp17_low_score_evidence/prompts/exp27e_conflict_adjudication_prompt.md \
  "${OUT_DIR}/prompts/exp27e_conflict_adjudication_prompt.md"
cp thesis_exp/exp17_low_score_evidence/schemas/exp27e_conflict_adjudication_schema.json \
  "${OUT_DIR}/schema/exp27e_conflict_adjudication_schema.json"

python thesis_exp/exp17_low_score_evidence/validate_exp27e_conflict_adjudication.py \
  --out-dir "${OUT_DIR}" \
  --dev-jsonl "${DEV_JSONL}" \
  --test-jsonl "${TEST_JSONL}"
