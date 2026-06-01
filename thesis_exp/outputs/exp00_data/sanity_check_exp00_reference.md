# Exp 0.1 Reference Sanity Check

Overall status: **PASS**

| check | status | observed | expected | notes |
| --- | --- | --- | --- | --- |
| processed dataset rows | PASS | 5536 | 5536 |  |
| label_5 distribution unchanged | PASS | {1: 86, 2: 113, 3: 507, 4: 1903, 5: 2927} | {1: 86, 2: 113, 3: 507, 4: 1903, 5: 2927} |  |
| label_5 domain | PASS | [1, 2, 3, 4, 5] | [1, 2, 3, 4, 5] |  |
| generator model count | PASS | ['deepseek-r1', 'deepseek-v3', 'qwen-max', 'qwen2.5-14b-instruct', 'qwen2.5-7b-instruct'] | ['deepseek-r1', 'qwen-max', 'qwen2.5-7b-instruct', 'deepseek-v3', 'qwen2.5-14b-instruct'] |  |
| canonical metric count | PASS | 12 | 12 |  |
| canonical scenario count | PASS | 9 | 9 |  |
| all rows have metric_canonical | PASS | 0 | 0 |  |
| all rows have scenario_canonical | PASS | 0 | 0 |  |
| synthetic files excluded | PASS | 0 | 0 |  |
| paper_like_triple_seed42 split counts | PASS | {'train': 2654, 'dev': 664, 'test': 2218} | {'train': 2654, 'dev': 664, 'test': 2218} |  |
| question_seed42 split counts | PASS | {'train': 3326, 'dev': 1107, 'test': 1103} | {'train': 3326, 'dev': 1107, 'test': 1103} |  |
| paper-like train+dev rows | PASS | 3318 | 3318 |  |
| paper-like test rows | PASS | 2218 | 2218 |  |
| paper-like triple_key cross-split overlap | PASS | 0 | 0 |  |
| question split question_key cross-split overlap | PASS | 0 | 0 |  |
| score scale mapping matches 5-grades.py logic | PASS | 0 | 0 |  |
| canonical subject count | PASS | 25 | 25 | WARNING is non-blocking for global sanity. |
| duplicate full scored item across split files | PASS | 0 | 0 | If non-zero, see tables/leakage_details.csv. |
| reference contract yaml.safe_load | PASS | dict | readable YAML |  |
| reference contract required fields | PASS | [] | [] |  |
