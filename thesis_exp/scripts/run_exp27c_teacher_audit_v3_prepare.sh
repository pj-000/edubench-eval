#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

OUT_DIR="${OUT_DIR:-thesis_exp/exp17_low_score_evidence/outputs/exp27c_teacher_audit_v3_seed42}"
TRAIN_JSONL="${TRAIN_JSONL:-thesis_exp/data/splits/question_seed42/train.jsonl}"
DEV_JSONL="${DEV_JSONL:-thesis_exp/data/splits/question_seed42/dev.jsonl}"
TEST_JSONL="${TEST_JSONL:-thesis_exp/data/splits/question_seed42/test.jsonl}"
EXP27A_AGREEMENT="${EXP27A_AGREEMENT:-thesis_exp/exp17_low_score_evidence/outputs/exp27a_teacher_audit_pilot_seed42/tables/exp27a_teacher_blind_cross_provider_agreement.csv}"
SEED="${SEED:-42}"
NEW_LOW_COUNT="${NEW_LOW_COUNT:-16}"
NEW_MID_COUNT="${NEW_MID_COUNT:-12}"
NEW_HIGH_COUNT="${NEW_HIGH_COUNT:-12}"
BATCH_SIZE="${BATCH_SIZE:-20}"

cat <<CONFIG
Exp27C teacher-audit v3 re-pilot preparation
OUT_DIR=${OUT_DIR}
TRAIN_JSONL=${TRAIN_JSONL}
DEV_JSONL=${DEV_JSONL}
TEST_JSONL=${TEST_JSONL}
EXP27A_AGREEMENT=${EXP27A_AGREEMENT}
SEED=${SEED}
NEW_LOW_COUNT=${NEW_LOW_COUNT}
NEW_MID_COUNT=${NEW_MID_COUNT}
NEW_HIGH_COUNT=${NEW_HIGH_COUNT}
BATCH_SIZE=${BATCH_SIZE}

This step does not call APIs, does not train, and reads dev/test only for ID guards.
CONFIG

python thesis_exp/exp17_low_score_evidence/prepare_exp27c_teacher_audit_v3_packets.py \
  --train-jsonl "${TRAIN_JSONL}" \
  --dev-jsonl "${DEV_JSONL}" \
  --test-jsonl "${TEST_JSONL}" \
  --exp27a-agreement "${EXP27A_AGREEMENT}" \
  --out-dir "${OUT_DIR}" \
  --seed "${SEED}" \
  --new-low-count "${NEW_LOW_COUNT}" \
  --new-mid-count "${NEW_MID_COUNT}" \
  --new-high-count "${NEW_HIGH_COUNT}" \
  --batch-size "${BATCH_SIZE}"

python thesis_exp/exp17_low_score_evidence/validate_exp27c_teacher_audit.py \
  --out-dir "${OUT_DIR}" \
  --exp27a-agreement "${EXP27A_AGREEMENT}"

echo "Exp27C teacher-audit v3 preparation completed."
