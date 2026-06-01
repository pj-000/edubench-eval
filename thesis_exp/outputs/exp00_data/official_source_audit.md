# Official Source Audit

This audit checks local files, `EduBench.zip`, and a read-only clone of the official EduBench GitHub repository when network access is available.

## Summary

| field | value |
| --- | --- |
| primary_local_source | results_merge.jsonl |
| results_merge_status | local_derived_from_edubench_not_official_filename |
| official_roots | ["/Users/sss/edubench-eval/thesis_exp/.cache/official_edubench"] |
| official_inventory_rows | 23 |
| official_exact_triple_matches_sampled | 0 |
| official_question_fuzzy_matches_sampled | 47 |

## Inventory

| source_path | source_origin | file_role | num_records | contains_question | contains_answer | contains_metric | contains_task_or_scenario | contains_human_score | contains_model_eval_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EduBench.zip | zip | unknown | 12 | False | False | False | False | False | False |
| README.md | official_github_clone | readme | 0 | False | False | False | False | False | False |
| code/readme.md | official_github_clone | readme | 0 | False | False | False | False | False | False |
| data/all_data/en_data/AG.jsonl | official_github_clone | en_data | 1042 | True | True | False | False | False | True |
| data/all_data/en_data/EC.jsonl | official_github_clone | en_data | 1301 | True | True | False | False | False | False |
| data/all_data/en_data/ES.jsonl | official_github_clone | en_data | 1061 | True | True | False | False | False | False |
| data/all_data/en_data/IP.jsonl | official_github_clone | en_data | 1301 | True | True | False | False | False | False |
| data/all_data/en_data/PCC.jsonl | official_github_clone | en_data | 252 | True | True | False | False | False | False |
| data/all_data/en_data/PLS.jsonl | official_github_clone | en_data | 448 | True | True | False | True | False | False |
| data/all_data/en_data/Q&A.jsonl | official_github_clone | en_data | 1285 | True | True | False | False | False | False |
| data/all_data/en_data/QG.jsonl | official_github_clone | en_data | 1288 | True | True | False | False | False | False |
| data/all_data/en_data/TMG.jsonl | official_github_clone | en_data | 1185 | True | True | False | False | False | False |
| data/all_data/sampled_data/en_data_sampled.jsonl | official_github_clone | en_data | 99 | True | True | True | True | False | True |
| data/all_data/sampled_data/zh_data_sampled.jsonl | official_github_clone | zh_data | 99 | True | True | True | True | False | True |
| data/all_data/zh_data/AG.jsonl | official_github_clone | zh_data | 931 | True | True | False | False | False | True |
| data/all_data/zh_data/EC.jsonl | official_github_clone | zh_data | 620 | True | True | False | False | False | False |
| data/all_data/zh_data/IP.jsonl | official_github_clone | zh_data | 1342 | True | True | False | False | False | False |
| data/all_data/zh_data/PCC.jsonl | official_github_clone | zh_data | 568 | True | True | False | False | False | False |
| data/all_data/zh_data/PLS.jsonl | official_github_clone | zh_data | 348 | True | True | False | True | False | False |
| data/all_data/zh_data/Q&A.jsonl | official_github_clone | zh_data | 1306 | True | True | False | False | False | False |
| data/all_data/zh_data/QG.jsonl | official_github_clone | zh_data | 1343 | True | True | False | False | False | False |
| data/all_data/zh_data/TMG.jsonl | official_github_clone | zh_data | 1335 | True | True | False | False | False | False |
| data/readme.md | official_github_clone | readme | 0 | False | False | False | False | False | False |

## Local Source Status

`results_merge.jsonl` is treated as `local_derived_from_edubench_not_official_filename`. It is not labeled as official full EduBench raw data unless an official repository file with the same role/name is found.
