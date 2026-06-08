# Exp6 Synthetic Low-Score Output Schema

| field | specification |
| --- | --- |
| `synthetic_id` | string; stable generated id |
| `source_record_id` | string; train split source record id |
| `source_question_key` | string; source train question key |
| `source_triple_key` | string; source train triple key |
| `source_split` | literal train |
| `question` | string; copied from train source question |
| `answer_synthetic` | string; generated plausible flawed answer |
| `metric_canonical` | string; one of the 12 EduBench metrics |
| `rubric_text` | string; source rubric |
| `language` | en or zh |
| `scenario_canonical` | string |
| `subject_canonical` | string |
| `education_level_canonical` | string |
| `target_label_5` | integer 1-5; synthetic design pseudo-label |
| `label_source` | literal synthetic_design |
| `error_type` | one of the Exp6-2 error types |
| `generation_model` | string; planned model, e.g. deepseek-v4-pro |
| `generation_prompt_version` | string |
| `generation_timestamp` | ISO timestamp from generation runner |
| `generation_status` | dry_run/planned/generated/failed |
| `raw_generation` | raw model JSON/text if generated |
| `filter_status` | pending/pass/fail |
| `filter_reasons` | array of strings |

Synthetic labels are pseudo labels from experimental design. They must not be described as human
labels.
Generated rows may only be used for train-side augmentation after leakage and filtering checks pass.
