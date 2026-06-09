# BLOCKED: Exp6-10 Topup-2 Generation Did Not Run

Generation was not executed and no synthetic rows were fabricated.

Reason: **EXP6_RUN_GENERATION is not set to 1**

Required environment:

- `EXP6_RUN_GENERATION=1`
- `GENERATION_MODEL=deepseek-v4-pro`
- an API key environment variable, unless a local endpoint is configured and needs no key

Static checks:

| check_name | status | count | notes |
| --- | --- | --- | --- |
| py_compile_exp06_generation | PASS | 0 |  |
| topup2_prompts_exists | PASS | 1 | thesis_exp/outputs/exp06_synthetic_low_score/topup_generation/topup2/... |
| prompt_count_160 | PASS | 160 | expected 160 |
| label_distribution_70_66_24 | PASS | 160 | {'3': 24, '2': 66, '1': 70} |
| language_distribution_80_80 | PASS | 160 | {'en': 80, 'zh': 80} |
| metric_coverage_12 | PASS | 12 | 12 metrics |
| error_type_coverage_7 | PASS | 7 | 7 error types |
| all_sources_from_question_seed42_train | PASS | 0 | source_split must be train |
| no_source_question_overlap_dev_test | PASS | 0 | dev/test question forbidden |
| no_source_triple_overlap_dev_test | PASS | 0 | dev/test triple forbidden |
| source_reuse_count_recorded | PASS | 160 | rows reusing Batch96/Topup1 source questions |
| api_key_not_written_to_topup2_logs | PASS | 0 | scanned secret-like key markers |
| no_checkpoint_or_weights_tracked | PASS | 0 |  |

The runner is capped at exactly 160 prompt rows and never logs API keys.
