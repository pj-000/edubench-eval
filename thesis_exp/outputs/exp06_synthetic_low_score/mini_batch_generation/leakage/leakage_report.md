# Exp6-3 Mini-batch Leakage Report

Status: **DRY_RUN_NO_GENERATED_ROWS**

- Filtered input path: `thesis_exp/outputs/exp06_synthetic_low_score/mini_batch_generation/filtered/filtered_synthetic_candidates.jsonl`
- Leakage summary path: `thesis_exp/outputs/exp06_synthetic_low_score/mini_batch_generation/leakage/leakage_summary.csv`
- Leakage details path: `thesis_exp/outputs/exp06_synthetic_low_score/mini_batch_generation/leakage/leakage_details.csv`
- Filtered rows checked: **0**
- Blocked rows: **0**

Checks:

- source question/triple keys must not occur in dev/test.
- synthetic question and question+answer keys must not occur in dev/test.
- synthetic answers must not duplicate human test answers.
- synthetic ids and answers must be unique within the batch.
- source split must be train.

Any dev/test leakage blocks the row and blocks Exp6 training use.
