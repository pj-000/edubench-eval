# Exp6 Final Low-score Quality Review Pool

## Pool Composition

- Combined candidate count: **399**
- Final quality-review pool count: **384**
- Overflow candidate count: **15**
- Final label distribution: `{'1': 168, '2': 168, '3': 48}`
- Final language distribution: `{'en': 194, 'zh': 190}`
- Metric coverage: **12**
- Error type coverage: **7**

## Source Reuse

- Topup2 prior source question reuse max/mean: **2 / 1.1625**
- Topup2 within-batch source question reuse max/mean: **4 / 1.7937**
- Source reuse is expected because all 118 unique `question_seed42/train` question keys had already been used by Batch96 + Topup1 before Topup2.

## Review Gates

- Final low-score quality review can start: **YES**
- Final synthetic pool build can start: **YES**
- Exp6 training can start: **NO**

Synthetic labels remain `synthetic_design` / `pseudo_label`; they are not human labels. The final384 pool is for manual/GPT quality review and curation before any training use.

## Outputs

- `final384_low_score_candidates_for_quality_review.jsonl`
- `final384_quality_review_sheet.csv`
- `combined_low_score_review_candidates.jsonl`
- `final384_overflow_candidates.csv`
- `final_pool_summary.csv`

## Sanity Check

| check_name | status | count | notes |
| --- | --- | ---: | --- |
| combined_count_399 | PASS | 399 | Batch96 curated + Topup1 curated + Topup2 filtered |
| final384_count | PASS | 384 | selected final quality review pool |
| final384_label_distribution | PASS | 384 | {"1": 168, "2": 168, "3": 48} |
| no_duplicate_synthetic_id | PASS | 0 | final384 pool |
| no_duplicate_answer_hash | PASS | 0 | normalized answer hash |
| all_sources_train | PASS | 0 | source_split must be train |
| topup2_leakage_pass | PASS | 0 | Topup2 leakage summary blocked rows=0 |
| training_blocked | PASS | 0 | Exp6 training remains NO until quality review/curation complete |
