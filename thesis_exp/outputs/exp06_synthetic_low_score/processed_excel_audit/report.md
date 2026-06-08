# Exp6-1 Processed Excel Label Provenance / Dedup / Train-only Confirmation

## Scope

This audit only inspects `deepseek_output/processed_excel_data_1*`. It does not train models, call
APIs, generate synthetic data, modify Exp0-Exp5 results, or add synthetic rows to any train/dev/test
split.

## Main Findings

- `processed_excel_data_1.jsonl` duplicate of `_en + _zh`: **YES**
- Labels human-confirmed: **NO**
- Any exact dev/test overlap: **NO**
- Train-only candidate rows after canonical dedup: **659**
- Low-score train-only candidate rows: **8**
- Can Exp6 training start with processed Excel data now: **NO**

## Source Comparison

| source_a | source_b | records_a | records_b | score_rows_a | score_rows_b | record_overlap | question_overlap | triple_overlap | qa_overlap | candidate_overlap | merge_equals_en_plus_zh | recommended_non_duplicate_use |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek_output/processed_excel_data_1.jsonl | deepseek_output/processed_excel_data_1_en.jsonl | 129 | 64 | 659 | 331 | 64 | 64 | 331 | 64 | 331 | True | use merged only OR en+zh only, never both |
| deepseek_output/processed_excel_data_1.jsonl | deepseek_output/processed_excel_data_1_zh.jsonl | 129 | 65 | 659 | 328 | 65 | 65 | 328 | 65 | 328 | True | use merged only OR en+zh only, never both |
| deepseek_output/processed_excel_data_1_en.jsonl | deepseek_output/processed_excel_data_1_zh.jsonl | 64 | 65 | 331 | 328 | 0 | 0 | 0 | 0 | 0 | True | use merged only OR en+zh only, never both |

## Schema Profile

| source_file | num_records | score_rows | top_level_fields | nested_score_fields | score_count_distribution | language_distribution |
| --- | --- | --- | --- | --- | --- | --- |
| deepseek_output/processed_excel_data_1.jsonl | 129 | 659 | ["id", "level", "model", "question", "response", "scores"] | ["criterion", "reason", "score"] | {"5": 115, "6": 14} | {"en": 319, "zh": 340} |
| deepseek_output/processed_excel_data_1_en.jsonl | 64 | 331 | ["id", "level", "model", "question", "response", "scores"] | ["criterion", "reason", "score"] | {"5": 53, "6": 11} | {"en": 319, "zh": 12} |
| deepseek_output/processed_excel_data_1_zh.jsonl | 65 | 328 | ["id", "level", "model", "question", "response", "scores"] | ["criterion", "reason", "score"] | {"5": 62, "6": 3} | {"zh": 328} |

## Label Provenance

The files contain `scores[].score` and `scores[].reason`, plus a top-level `model` field. They do
not contain a human/reviewer/annotator marker. Therefore labels are not human-confirmed and should
be
treated as model/pseudo labels unless manually traced back to a human-reviewed Excel source.

| source_file | score_fields | evidence_for_human_label | evidence_for_model_label | evidence_for_pseudo_label | provenance_status | confidence | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek_output/processed_excel_data_1.jsonl | ["scores[].criterion", "scores[].reason", "scores[].score"] |  | model field values=['DeepSeek-V3.2-Exp'] | scores[].score with scores[].reason; no human provenance marker | model_label_likely | high | model field and score/reason structure indicate labels are model or p... |
| deepseek_output/processed_excel_data_1_en.jsonl | ["scores[].criterion", "scores[].reason", "scores[].score"] |  | model field values=['DeepSeek-V3.2-Exp'] | scores[].score with scores[].reason; no human provenance marker | model_label_likely | high | model field and score/reason structure indicate labels are model or p... |
| deepseek_output/processed_excel_data_1_zh.jsonl | ["scores[].criterion", "scores[].reason", "scores[].score"] |  | model field values=['DeepSeek-V3.2-Exp'] | scores[].score with scores[].reason; no human provenance marker | model_label_likely | high | model field and score/reason structure indicate labels are model or p... |

## Dedup

| scope | source_file | total_candidate_rows | unique_candidate_keys | duplicate_candidate_groups | duplicate_candidate_rows | dedup_recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| within_source | deepseek_output/processed_excel_data_1.jsonl | 659 | 659 | 0 | 0 | deduplicate repeated candidate_key rows within source |
| within_source | deepseek_output/processed_excel_data_1_en.jsonl | 331 | 331 | 0 | 0 | deduplicate repeated candidate_key rows within source |
| within_source | deepseek_output/processed_excel_data_1_zh.jsonl | 328 | 328 | 0 | 0 | deduplicate repeated candidate_key rows within source |
| all_sources | ALL | 1318 | 659 | 659 | 659 | if used, choose merged only as canonical; en/zh are duplicate split v... |

## Leakage

Exact train/dev/test checks were run using normalized question, question+answer, and
question+answer+metric keys. The candidate source has no exact dev/test overlap after canonical
dedup.

| source_file | usable_for_exp6_training | recommended_use | label_provenance_status | train_only_rows | low_score_rows | duplicate_risk | leakage_risk | next_action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek_output/processed_excel_data_1.jsonl | NO | pilot_only_after_manual_approval | model_label_likely | 659 | 8 | HIGH_IF_COMBINED_WITH_EN_ZH | LOW | confirm label provenance; use only as pseudo-label pilot if approved;... |
| deepseek_output/processed_excel_data_1_en.jsonl | NO | do_not_use_with_merged | model_label_likely | 0 | 0 | DUPLICATE_OF_MERGED | LOW | confirm label provenance; use only as pseudo-label pilot if approved;... |
| deepseek_output/processed_excel_data_1_zh.jsonl | NO | do_not_use_with_merged | model_label_likely | 0 | 0 | DUPLICATE_OF_MERGED | LOW | confirm label provenance; use only as pseudo-label pilot if approved;... |

## Low-Score Candidate Summary

```json
{
  "by_label": {
    "2": 8
  },
  "by_language": {
    "en": 5,
    "zh": 3
  },
  "by_metric": {
    "Motivation, Guidance & Positive Feedback": 8
  },
  "by_source_file": {
    "deepseek_output/processed_excel_data_1.jsonl": 8
  },
  "total_low_score_candidates": 8
}
```

## Recommendation

Do **not** start full Exp6 training from `processed_excel_data_1*` yet. The canonical merged source
has train-only rows after exact leakage checks, but only **8** low-score rows and
the labels are not human-confirmed. At most, use it as a tiny pseudo-label pilot after manual
approval. For a meaningful low-score augmentation experiment, obtain or generate additional
train-only low-score data with explicit provenance and no dev/test overlap.

## Files

- `thesis_exp/outputs/exp06_synthetic_low_score/processed_excel_audit/processed_excel_source_comparison.csv`
- `thesis_exp/outputs/exp06_synthetic_low_score/processed_excel_audit/processed_excel_label_provenance.csv`
- `thesis_exp/outputs/exp06_synthetic_low_score/processed_excel_audit/processed_excel_train_only_candidates.csv`
- `thesis_exp/outputs/exp06_synthetic_low_score/processed_excel_audit/processed_excel_low_score_candidates.csv`
- `thesis_exp/outputs/exp06_synthetic_low_score/processed_excel_audit/processed_excel_final_recommendation.csv`
