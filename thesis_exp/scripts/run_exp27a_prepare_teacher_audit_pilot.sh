#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

OUT_DIR="${OUT_DIR:-thesis_exp/exp17_low_score_evidence/outputs/exp27a_teacher_audit_pilot_seed42}"
TRAIN_JSONL="${TRAIN_JSONL:-thesis_exp/data/splits/question_seed42/train.jsonl}"
DEV_JSONL="${DEV_JSONL:-thesis_exp/data/splits/question_seed42/dev.jsonl}"
TEST_JSONL="${TEST_JSONL:-thesis_exp/data/splits/question_seed42/test.jsonl}"
MID_COUNT="${MID_COUNT:-100}"
HIGH_COUNT="${HIGH_COUNT:-150}"
BATCH_SIZE="${BATCH_SIZE:-20}"
SEED="${SEED:-42}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

cat <<CONFIG
Exp27A teacher-audit pilot packet preparation
OUT_DIR=${OUT_DIR}
TRAIN_JSONL=${TRAIN_JSONL}
DEV_JSONL=${DEV_JSONL}
TEST_JSONL=${TEST_JSONL}
MID_COUNT=${MID_COUNT}
HIGH_COUNT=${HIGH_COUNT}
BATCH_SIZE=${BATCH_SIZE}
SEED=${SEED}

This step does not call teacher APIs, does not train, and reads dev/test only for ID leakage guards.
CONFIG

python -m py_compile \
  thesis_exp/exp17_low_score_evidence/prepare_exp27a_teacher_audit_packets.py \
  thesis_exp/exp17_low_score_evidence/run_exp27a_teacher_audit_api.py \
  thesis_exp/exp17_low_score_evidence/validate_exp27a_teacher_audit.py

python thesis_exp/exp17_low_score_evidence/prepare_exp27a_teacher_audit_packets.py \
  --train-jsonl "${TRAIN_JSONL}" \
  --dev-jsonl "${DEV_JSONL}" \
  --test-jsonl "${TEST_JSONL}" \
  --out-dir "${OUT_DIR}" \
  --seed "${SEED}" \
  --mid-count "${MID_COUNT}" \
  --high-count "${HIGH_COUNT}" \
  --batch-size "${BATCH_SIZE}"

python thesis_exp/exp17_low_score_evidence/validate_exp27a_teacher_audit.py \
  --out-dir "${OUT_DIR}"

echo "Exp27A teacher-audit pilot packet preparation completed."
