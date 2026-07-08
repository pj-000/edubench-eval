# Exp27I Codex-Calibrated Teacher-Audited Data

This step builds the final train-only calibrated 361-row annotation set from Exp27I Qwen/DeepSeek
outputs.

## What Codex Reviewed

- The top 80 conflict cases were inspected in batches using the evaluator output, metric, Qwen reason, DeepSeek reason, original human score, and audit flags.
- The recurring conflict modes were: answer-key/rubric ambiguity, scenario-integration rubric mismatch, evaluator-output internal contradictions, and high-score-protection disagreements.
- High-conflict rows are intentionally not forced into high-weight gold labels. They are marked `review_only` unless there is clean teacher/human agreement.

## Outputs

- `data/exp27i_teacher_audited_361_calibrated_train.jsonl`: final calibrated train-only annotations.
- `data/exp27i_teacher_audited_sft_train_high_low_weight.jsonl`: SFT-ready subset using only high/low-weight calibrated rows.
- `annotation/exp27i_codex_top80_direct_review.csv`: top-conflict review decisions.
- `review/exp27i_calibrated_361_review_manifest.csv`: compact all-row review manifest.
- `tables/exp27i_calibration_use_counts.csv`: train-use counts.

## Counts

- calibrated_rows: 361
- top80_direct_review_rows: 80
- high_weight_rows: 256
- low_weight_rows: 36
- review_only_rows: 69
- train_ready_rows_high_or_low_weight: 292
- sft_ready_rows: 292
- calibrated_label_counts: `{'1': 67, '2': 66, '3': 33, '4': 81, '5': 114}`

## Training Recommendation

Use only `high_weight` and `low_weight` rows for the first teacher-audited SFT/DPO data-quality
experiment. Keep `review_only` rows for human/GPT adjudication or qualitative analysis.

This step does not fabricate DPO rejected responses. For DPO, use this calibrated dataset as the
corrected/chosen source and build rejected responses from real model mistakes in a separate
pair-construction step.

## Guardrails

- Only train split samples are included.
- Dev/test were used only as leakage guards in packet preparation.
- No test label was read.
- Raw API responses are not part of this calibrated output.
