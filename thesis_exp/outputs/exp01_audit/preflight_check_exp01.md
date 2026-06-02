# Exp1 Preflight Check

Overall status: **PASS**

| check | status | observed | expected | notes |
| --- | --- | --- | --- | --- |
| Exp0.1 sanity check exists | PASS | thesis_exp/outputs/exp00_data/sanity_check_exp00_reference.md | exists |  |
| main processed dataset exists | PASS | thesis_exp/data/processed/edubench_scoring_all.jsonl | exists |  |
| main test split exists | PASS | thesis_exp/data/splits/paper_like_triple_seed42/test.jsonl | exists |  |
| main dataset name | PASS | edubench_audit_human_scored_subset | edubench_audit_human_scored_subset |  |
| paper-like test set row count | PASS | 2218 | 2218 | Do not abort on mismatch; write warning to report. |
| synthetic/sample candidates excluded | PASS | ["download_raw/deepseek-r1_pointwise_filtered_en_data_sampled.jsonl", "download_raw/deepseek-... | excluded from alignment/data source | Inventory may list them as excluded; Exp1 does not use them as prediction sources. |
