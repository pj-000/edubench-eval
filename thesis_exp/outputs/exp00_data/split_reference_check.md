# Split Reference Check

This split uses the local original held-out flag from report/results_merge_enriched.jsonl.

| field | value |
| --- | --- |
| split_name | paper_like_triple_seed42 |
| split_source | original_heldout_flag_repaired_for_triple_isolation |
| train rows | 2654 |
| dev rows | 664 |
| train_pool rows | 3318 |
| test rows | 2218 |
| test scenario count | 8 |
| test metric count | 12 |
| test subject count | 25 |
| test education level count | 6 |
| test language count | 2 |

PDF reference: train_pool = 3318, held-out test = 2218, split unit = question-answer-metric triple. The held-out test reported in the PDF spans 8 scenarios and complete coverage of subjects, education levels, and dimensions.
