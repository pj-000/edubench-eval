#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

OUT_DIR="${OUT_DIR:-thesis_exp/exp17_low_score_evidence/outputs/exp22_r7_matched_controls_seed42}"

python -m py_compile thesis_exp/exp17_low_score_evidence/prepare_exp22_r7_matched_controls.py

python thesis_exp/exp17_low_score_evidence/prepare_exp22_r7_matched_controls.py \
  --r7d thesis_exp/exp17_low_score_evidence/outputs/exp19_r7_reason_recovered_real_dpo_seed42/data/edubench_r7d_strict_label_consistent_reason_real_dpo_train.json \
  --r7a thesis_exp/exp17_low_score_evidence/outputs/exp19_r7_reason_recovered_real_dpo_seed42/data/edubench_r7a_score_only_reason_covered_real_dpo_train.json \
  --manifest thesis_exp/exp17_low_score_evidence/outputs/exp19_r7_reason_recovered_real_dpo_seed42/review/r7_reason_recovered_dpo_review_manifest.csv \
  --pair-counts thesis_exp/exp17_low_score_evidence/outputs/exp19_r7_reason_recovered_real_dpo_seed42/tables/r7_pair_counts.csv \
  --dev-jsonl thesis_exp/data/splits/question_seed42/dev.jsonl \
  --test-jsonl thesis_exp/data/splits/question_seed42/test.jsonl \
  --out-dir "${OUT_DIR}"

python thesis_exp/exp17_low_score_evidence/validate_exp19_llamafactory_data.py \
  --dataset-dir "${OUT_DIR}" \
  --dataset-info "${OUT_DIR}/dataset_info.json"
