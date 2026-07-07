#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python -m py_compile thesis_exp/exp17_low_score_evidence/prepare_exp19_r7_reason_recovered_real_dpo.py

python thesis_exp/exp17_low_score_evidence/prepare_exp19_r7_reason_recovered_real_dpo.py \
  --train-jsonl thesis_exp/data/splits/question_seed42/train.jsonl \
  --dev-jsonl thesis_exp/data/splits/question_seed42/dev.jsonl \
  --test-jsonl thesis_exp/data/splits/question_seed42/test.jsonl \
  --reason-root 5-grades \
  --out-dir thesis_exp/exp17_low_score_evidence/outputs/exp19_r7_reason_recovered_real_dpo_seed42

python thesis_exp/exp17_low_score_evidence/validate_exp19_llamafactory_data.py \
  --dataset-dir thesis_exp/exp17_low_score_evidence/outputs/exp19_r7_reason_recovered_real_dpo_seed42 \
  --dataset-info thesis_exp/exp17_low_score_evidence/outputs/exp19_r7_reason_recovered_real_dpo_seed42/dataset_info.json
