# Exp19-SFT-D2 Diagnostic Datasets

This prepare step builds prompt-only LLaMA-Factory datasets for train-vs-dev structured failure
behavior diagnosis.
The user prompt contains only question, answer, metric, rubric, and metadata.
Gold labels, A0 weak labels, and D1 annotations are stored only in reference CSV files for
evaluation.

| subset | split | n | purpose |
|---|---|---:|---|
| train_low_reasoned_subset | train | 80 | Check whether R2/R2c learned to emit failure fields on train low-score examples with recovered reasons. |
| train_clean_high_subset | train | 2656 | Check whether high-score train controls are protected from false failure assignment. |
| dev_label2_subset | dev | 38 | Check whether dev label-2 samples are still over-scored. |
| dev_d1_hidden_subset | dev | 26 | Check whether recovered hidden-failure cases generalize on dev. |
| dev_d1_matched_controls | dev | 36 | Check whether matched high controls remain high and mostly no-failure. |
| dev_high_subset | dev | 400 | Check high-score protection on a stratified dev-high subset. |

This step does not read test, does not train, and does not use human rationale as prompt input.
