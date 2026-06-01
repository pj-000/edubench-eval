# Data Card: Exp 0.1 EduBench Audit Human-Scored Subset

## Dataset Identity

| field | value |
| --- | --- |
| dataset_name | edubench_audit_human_scored_subset |
| not_full_official_edubench | true |
| primary_local_source | results_merge.jsonl |
| source_status | local_derived_merged_human_scored_subset |
| official_alignment_status.scenario_metric | aligned |
| official_alignment_status.corpus_size | aligned_with_pdf_audit |
| official_alignment_status.subject | aligned_with_local_enriched_audit_metadata |
| official_alignment_status.full_official_data | not_reconstructed |

## Source Selection

Primary source: `results_merge.jsonl`.

The selected source has merged scored items with three human annotation fields and automatic judge
metadata. Synthetic/sample files are excluded from the processed dataset. This is the PDF audit
corpus / human-scored subset, not the official full EduBench benchmark.

## Distinction between official EduBench full data and PDF audit subset

Official EduBench full data is described as 9 scenarios, 4000+ educational contexts, 18821 data
points, and 500 sampled queries evaluated by human raters and LLMs. The local thesis dataset here is
the 5536-item PDF audit corpus with human and judge scores. Downstream Exp1 evaluator
training/testing should use this 5536-row audit corpus as the main human-labeled dataset. Future
synthetic augmentation or distillation may use official full data separately, but those rows must
not be mixed into the main human-labeled test set.

## Dataset Statistics

| stat | value |
| --- | --- |
| dataset_name | edubench_audit_human_scored_subset |
| total_scored_items | 5536 |
| unique_triple_key | 5423 |
| unique_question_key | 197 |
| unique_answer_key | 963 |
| duplicate_triple_groups | 77 |
| canonical_metric_count | 12 |
| canonical_scenario_count | 9 |
| canonical_subject_count | 25 |
| education_level_count | 6 |
| language_count | 2 |

## Label Distribution

| value | count | pct |
| --- | --- | --- |
| 5 | 2927 | 0.5287 |
| 4 | 1903 | 0.3438 |
| 3 | 507 | 0.0916 |
| 2 | 113 | 0.0204 |
| 1 | 86 | 0.0155 |

## Generator Model Distribution

| value | count | pct |
| --- | --- | --- |
| deepseek-r1 | 1119 | 0.2021 |
| qwen-max | 1110 | 0.2005 |
| qwen2.5-7b-instruct | 1108 | 0.2001 |
| deepseek-v3 | 1100 | 0.1987 |
| qwen2.5-14b-instruct | 1099 | 0.1985 |

## Metric Distribution

| value | count | pct |
| --- | --- | --- |
| Instruction Following & Task Completion | 990 | 0.1788 |
| Content Relevance & Scope Control | 660 | 0.1192 |
| Basic Factual Accuracy | 660 | 0.1192 |
| Scenario Element Integration | 547 | 0.0988 |
| Clarity, Simplicity & Inspiration | 432 | 0.078 |
| Higher-Order Thinking & Skill Development | 428 | 0.0773 |
| Reasoning Process Rigor | 421 | 0.076 |
| Personalization, Adaptation & Learning Support | 329 | 0.0594 |
| Domain Knowledge Accuracy | 327 | 0.0591 |
| Motivation, Guidance & Positive Feedback | 315 | 0.0569 |
| Role & Tone Consistency | 220 | 0.0397 |
| Error Identification & Correction Precision | 207 | 0.0374 |

## Scenario Distribution

| value | count | pct |
| --- | --- | --- |
| Idea Provision | 863 | 0.1559 |
| Teaching Material Generation | 770 | 0.1391 |
| Error Correction | 750 | 0.1355 |
| Question Generation | 660 | 0.1192 |
| Automatic Grading | 628 | 0.1134 |
| Emotional Support | 549 | 0.0992 |
| Personalized Learning Support | 542 | 0.0979 |
| Question Answering | 444 | 0.0802 |
| Personalized Content Creation | 330 | 0.0596 |

## Reference checks against EduBench paper/PDF

| check | observed | reference | note |
| --- | --- | --- | --- |
| total scored items | 5536 | 5536 = 3318 train pool + 2218 held-out test | matches PDF audit corpus total |
| unique triple_key | 5423 | question-answer-metric scored item | used for evaluator-vs-human split |
| unique question_key | 197 | not fixed in task | used for robustness split |
| unique answer_key | 963 | not fixed in task | used for leakage diagnostics |
| generator_model distribution | 5 | 5 generated models | see distribution table |
| canonical metrics | 12 | 12 | aligned |
| canonical scenarios | 9 | 9 | aligned |
| canonical subjects | 25 | 25 canonical subjects | recovered from local enriched audit metadata when available |
| education levels | 6 | 6 education stages | inferred from question profile |
| languages | ["en", "zh"] | English / Chinese | script-detected |
| human annotator fields | ["human_1", "human_2", "human_3"] | 3 annotators | all three present |
| 3318 train pool / 2218 held-out test | reproducible at row-count level | 3318 / 2218 | make_splits targets this only when total is close to 5536 |
