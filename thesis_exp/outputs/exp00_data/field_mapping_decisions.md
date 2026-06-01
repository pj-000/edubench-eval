# Field Mapping Decisions

## Primary Human-Scored Source

Dataset name: `edubench_audit_human_scored_subset`.

`results_merge.jsonl` is selected as the primary local source because it contains one row per scored
item with `question`, `answer`, `metric`, `task`, generator `model`, and an `evaluate` object that
includes `human_1`, `human_2`, and `human_3` together with automatic judge scores.

`human_1.jsonl`, `human_2.jsonl`, and `human_3.jsonl` are treated as real human annotation sources
for provenance and schema profiling, but they are not directly concatenated into the main dataset
because they use a 1-10 score scale and would duplicate or partially overlap the already merged
scored items.

`sampled_merge_50_new.json` and `sampled_merge_50_new_swift.json` are inventory-only
synthetic/augmented files and are excluded from `edubench_scoring_all.jsonl`. The 5536-row dataset
is the PDF audit human-scored subset, not the full official EduBench data.

## Standard Field Construction

| standard_field | source_logic |
| --- | --- |
| question | results_merge.question |
| answer | results_merge.answer |
| metric_raw | results_merge.metric |
| scenario_raw | results_merge.task |
| generator_model / answer_model | results_merge.model |
| human_1/2/3 | results_merge.evaluate.human_1/2/3 |
| judge_scores | non-human keys from results_merge.evaluate |
| subject / education_level | explicit profile fields when recoverable, otherwise conservative keyw... |
| language | script-based detection from question, answer, and metric |
| record_id / triple_key / question_key / answer_key | stable SHA1 hashes over normalized text fields |

## Score Scale Handling

Each record stores `human_1_raw`/`human_2_raw`/`human_3_raw` and
`human_1_5`/`human_2_5`/`human_3_5`. Records with all available human scores in 1-5 are kept as
already normalized and rounded to `label_5`. Records with a 1-10 scale would be mapped using the
repository `5-grades.py` rule: 1-2->1, 3-4->2, 5-6->3, 7-8->4, 9-10->5. Ambiguous scales are
excluded from the main processed JSONL and recorded in `invalid_or_ambiguous_scores.csv`.

## Canonicalization Status

- Standardized rows: 5536
- Canonical metrics observed: 12
- Canonical scenarios observed: 9
- Unmapped metric rows in mapping table: 11
- Unmapped scenario rows in mapping table: 0
