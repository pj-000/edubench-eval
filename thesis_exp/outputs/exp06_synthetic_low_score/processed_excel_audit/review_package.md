# Exp6-1 Processed Excel Review Package

Can Exp6 training start with processed_excel_data_1*? **NO**

Are labels human-confirmed? **NO**

Is processed_excel_data_1.jsonl duplicate of en+zh? **YES**

How many train-only candidates remain? **659**

How many low-score candidates remain? **8**

Is this enough for a meaningful augmentation experiment? **NO**

Should we use it as pilot only? **YES**, only after manual approval as pseudo-label data.

Do we need to generate new train-only synthetic low-score data? **YES**, if Exp6 needs meaningful
low-score augmentation; this audit did not generate any new data.

## Final Recommendation

| source_file | usable_for_exp6_training | recommended_use | reason | label_provenance_status | train_only_rows | low_score_rows | duplicate_risk | leakage_risk | next_action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek_output/processed_excel_data_1.jsonl | NO | pilot_only_after_manual_approval | no dev/test overlap after exact checks, but labels are not human-conf... | model_label_likely | 659 | 8 | HIGH_IF_COMBINED_WITH_EN_ZH | LOW | confirm label provenance; use only as pseudo-label pilot if approved;... |
| deepseek_output/processed_excel_data_1_en.jsonl | NO | do_not_use_with_merged | en/zh split source is duplicate of processed_excel_data_1.jsonl | model_label_likely | 0 | 0 | DUPLICATE_OF_MERGED | LOW | confirm label provenance; use only as pseudo-label pilot if approved;... |
| deepseek_output/processed_excel_data_1_zh.jsonl | NO | do_not_use_with_merged | en/zh split source is duplicate of processed_excel_data_1.jsonl | model_label_likely | 0 | 0 | DUPLICATE_OF_MERGED | LOW | confirm label provenance; use only as pseudo-label pilot if approved;... |

## Required Manual Confirmations

- Trace `scores[].score` back to a human-reviewed Excel process, or keep it marked as model/pseudo label.
- Use either merged file or en+zh split files, never both.
- Keep processed Excel candidates train-only; do not add to dev/test.
- Do not run a full Exp6 mix until there are enough low-score rows and provenance is explicit.
