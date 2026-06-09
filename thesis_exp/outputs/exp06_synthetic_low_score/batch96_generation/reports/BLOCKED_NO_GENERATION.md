# BLOCKED: Exp6-6 Batch96 Generation Did Not Run

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
| batch96_prompts_exists | PASS | 1 | thesis_exp/outputs/exp06_synthetic_low_score/batch96_generation/promp... |
| prompt_count_96 | PASS | 96 | expected 96 |
| label_distribution_40_40_16 | PASS | 96 | {'1': 40, '2': 40, '3': 16} |
| language_distribution_48_48 | PASS | 96 | {'en': 48, 'zh': 48} |
| metric_coverage_12 | PASS | 12 | 12 metrics if possible |
| error_type_coverage_7 | PASS | 7 | 7 error types if possible |
| all_sources_from_question_seed42_train | PASS | 0 | source_split must be train |
| no_source_question_overlap_dev_test | PASS | 0 | dev/test question forbidden |
| no_source_triple_overlap_dev_test | PASS | 0 | dev/test triple forbidden |
| api_key_not_written_to_batch96_logs | PASS | 0 | scanned secret-like key markers |
| no_checkpoint_or_weights_tracked | PASS | 0 |  |

The runner is capped at exactly 96 prompt rows and never logs API keys.
