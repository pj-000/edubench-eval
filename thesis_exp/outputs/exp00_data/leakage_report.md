# Leakage Report

Overall status: **WARNING**

Interpretation: triple-key overlap is a failure for both split strategies. Question-key overlap is a warning for `paper_like_triple_seed42` because that split only promises question-answer-metric isolation, but it is a failure for `question_seed42`. Synthetic sampled overlap is reported only as a future augmentation risk and does not mean synthetic rows entered the main dataset.

## paper_like_triple_seed42: WARNING

| check | key_type | overlap_count | severity |
| --- | --- | --- | --- |
| triple_key cross-split overlap | triple_key | 0 | PASS |
| question_key cross-split overlap | question_key | 490 | WARNING |
| answer_key cross-split overlap | answer_key | 1454 | WARNING |
| normalized question+answer cross-split overlap | qa_key | 1454 | WARNING |
| duplicate record_id across split files | record_id | 0 | PASS |
| duplicate full scored item within same split | full_record_key | 25 | WARNING |
| duplicate full scored item across split files | full_record_key | 0 | PASS |
| synthetic sampled question overlap with dev/test | sampled_merge_50_new.json | 157 | INFO |
| synthetic sampled question overlap with dev/test | sampled_merge_50_new_swift.json | 157 | INFO |

## question_seed42: WARNING

| check | key_type | overlap_count | severity |
| --- | --- | --- | --- |
| triple_key cross-split overlap | triple_key | 0 | PASS |
| question_key cross-split overlap | question_key | 0 | PASS |
| answer_key cross-split overlap | answer_key | 0 | PASS |
| normalized question+answer cross-split overlap | qa_key | 0 | PASS |
| duplicate record_id across split files | record_id | 0 | PASS |
| duplicate full scored item within same split | full_record_key | 25 | WARNING |
| duplicate full scored item across split files | full_record_key | 0 | PASS |
| synthetic sampled question overlap with dev/test | sampled_merge_50_new.json | 66 | INFO |
| synthetic sampled question overlap with dev/test | sampled_merge_50_new_swift.json | 66 | INFO |

## Details

Detailed examples are written to `tables/leakage_details.csv`.
