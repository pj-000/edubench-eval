#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

OUT_DIR="${OUT_DIR:-thesis_exp/exp17_low_score_evidence/outputs/exp27d_teacher_audit_v4_seed42}"
TRAIN_JSONL="${TRAIN_JSONL:-thesis_exp/data/splits/question_seed42/train.jsonl}"
DEV_JSONL="${DEV_JSONL:-thesis_exp/data/splits/question_seed42/dev.jsonl}"
TEST_JSONL="${TEST_JSONL:-thesis_exp/data/splits/question_seed42/test.jsonl}"
EXP27C_PACKETS="${EXP27C_PACKETS:-thesis_exp/exp17_low_score_evidence/outputs/exp27c_teacher_audit_v3_seed42/packets/exp27c_v3_repilot_blind_packets.jsonl}"
EXP27C_AUDIT_REFERENCE="${EXP27C_AUDIT_REFERENCE:-thesis_exp/exp17_low_score_evidence/outputs/exp27c_teacher_audit_v3_seed42/packets/exp27c_v3_repilot_audit_reference_private.jsonl}"
SEED="${SEED:-42}"
LOW_STRESS_COUNT="${LOW_STRESS_COUNT:-8}"
HIGH_STRESS_COUNT="${HIGH_STRESS_COUNT:-4}"
MID_STRESS_COUNT="${MID_STRESS_COUNT:-4}"
EDU_STRESS_COUNT="${EDU_STRESS_COUNT:-4}"
BATCH_SIZE="${BATCH_SIZE:-20}"

cat <<CONFIG
Exp27D teacher-audit v4 re-pilot preparation
OUT_DIR=${OUT_DIR}
TRAIN_JSONL=${TRAIN_JSONL}
DEV_JSONL=${DEV_JSONL}
TEST_JSONL=${TEST_JSONL}
EXP27C_PACKETS=${EXP27C_PACKETS}
EXP27C_AUDIT_REFERENCE=${EXP27C_AUDIT_REFERENCE}
SEED=${SEED}
LOW_STRESS_COUNT=${LOW_STRESS_COUNT}
HIGH_STRESS_COUNT=${HIGH_STRESS_COUNT}
MID_STRESS_COUNT=${MID_STRESS_COUNT}
EDU_STRESS_COUNT=${EDU_STRESS_COUNT}
BATCH_SIZE=${BATCH_SIZE}

This step does not call APIs, does not train, and reads dev/test only for ID guards.
CONFIG

python thesis_exp/exp17_low_score_evidence/prepare_exp27d_teacher_audit_v4_packets.py \
  --train-jsonl "${TRAIN_JSONL}" \
  --dev-jsonl "${DEV_JSONL}" \
  --test-jsonl "${TEST_JSONL}" \
  --exp27c-packets "${EXP27C_PACKETS}" \
  --exp27c-audit-reference "${EXP27C_AUDIT_REFERENCE}" \
  --out-dir "${OUT_DIR}" \
  --seed "${SEED}" \
  --low-stress-count "${LOW_STRESS_COUNT}" \
  --high-stress-count "${HIGH_STRESS_COUNT}" \
  --mid-stress-count "${MID_STRESS_COUNT}" \
  --edu-stress-count "${EDU_STRESS_COUNT}" \
  --batch-size "${BATCH_SIZE}"

python thesis_exp/exp17_low_score_evidence/validate_exp27d_teacher_audit.py \
  --out-dir "${OUT_DIR}"

echo "Exp27D teacher-audit v4 preparation completed."
