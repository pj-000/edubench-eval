# Exp19 First-Round SFT Dev Prediction Dataset

- split: `dev`
- source jsonl: `thesis_exp/data/splits/question_seed42/dev.jsonl`
- examples: 1107
- LLaMA-Factory dataset name: `edubench_exp19_dev_score_eval`
- eval data file: `thesis_exp/exp17_low_score_evidence/outputs/exp19_sft_dpo_datasets_seed42/data/edubench_exp19_dev_score_eval.json`
- reference table: `thesis_exp/exp17_low_score_evidence/outputs/exp19_sft_dpo_datasets_seed42/tables/exp19_dev_reference.csv`

The user prompt contains only question, answer, metric, rubric, and metadata.
The assistant label is present only as the held-out reference target for prediction bookkeeping.
This step does not read test and does not train a model.
