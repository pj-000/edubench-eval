# Exp19-SFT-D2 Train-vs-Dev Structured Failure Diagnostic

This diagnostic uses existing SFT adapters for prediction only. It does not train, does not read
test, and does not put human rationale or failure labels into prompts.

## Key Answers

- Does R2n/R2c output major failures on train low reasoned examples? R2c nonempty rate is `1.0000`.
- If train is learned, does it generalize to dev D1 hidden cases? R2c dev D1 nonempty rate is `0.0000`.
- Does R4b behave similarly to R2c? R4b train low nonempty rate is `0.0250`; compare this with R2c above.
- Are high controls protected? R2c train clean-high nonempty failure rate is `0.0008`.
- Is score calibration still risky? R2c dev D1 pred>=4 rate is `1.0000`.
- Decision: `failure_learned_on_train_not_dev`; recommendation: DPO or data augmentation, but do not run test/all_5536 yet..

## Diagnostic Metrics

| adapter | subset | n | MAE | QWK | mean pred | pred>=4 | pred<=2 | label2 recall | high-to-low | parse |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| r2n_reason_score_natural | train_low_reasoned_subset | 80 | 0.8250 | 0.3245 | 2.2750 | 0.1500 | 0.5000 | 0.1111 | nan | 1.0000 |
| r2n_reason_score_natural | train_clean_high_subset | 2656 | 0.0907 | 0.8081 | 4.6427 | 0.9891 | 0.0000 | nan | 0.0000 | 1.0000 |
| r2n_reason_score_natural | dev_label2_subset | 38 | 2.2632 | nan | 4.2632 | 0.7368 | 0.0000 | 0.0000 | nan | 1.0000 |
| r2n_reason_score_natural | dev_d1_hidden_subset | 26 | 2.7308 | nan | 4.7308 | 1.0000 | 0.0000 | 0.0000 | nan | 1.0000 |
| r2n_reason_score_natural | dev_d1_matched_controls | 36 | 0.3611 | 0.3433 | 4.4444 | 0.8889 | 0.0000 | nan | 0.0000 | 1.0000 |
| r2n_reason_score_natural | dev_high_subset | 400 | 0.2200 | 0.5969 | 4.5675 | 0.9650 | 0.0000 | nan | 0.0000 | 1.0000 |
| r2c_clean_reason_score_balanced | train_low_reasoned_subset | 80 | 0.0000 | 1.0000 | 1.4500 | 0.0000 | 1.0000 | 1.0000 | nan | 1.0000 |
| r2c_clean_reason_score_balanced | train_clean_high_subset | 2656 | 0.1510 | 0.6544 | 4.6532 | 0.9838 | 0.0008 | nan | 0.0008 | 1.0000 |
| r2c_clean_reason_score_balanced | dev_label2_subset | 38 | 2.3947 | nan | 4.3947 | 0.7632 | 0.0000 | 0.0000 | nan | 1.0000 |
| r2c_clean_reason_score_balanced | dev_d1_hidden_subset | 26 | 2.8846 | nan | 4.8846 | 1.0000 | 0.0000 | 0.0000 | nan | 1.0000 |
| r2c_clean_reason_score_balanced | dev_d1_matched_controls | 36 | 0.3889 | 0.4126 | 4.4167 | 0.9167 | 0.0000 | nan | 0.0000 | 1.0000 |
| r2c_clean_reason_score_balanced | dev_high_subset | 400 | 0.2425 | 0.5376 | 4.5750 | 0.9650 | 0.0000 | nan | 0.0000 | 1.0000 |
| r4b_shuffled_reason_balanced | train_low_reasoned_subset | 80 | 0.0000 | 1.0000 | 1.4500 | 0.0000 | 1.0000 | 1.0000 | nan | 1.0000 |
| r4b_shuffled_reason_balanced | train_clean_high_subset | 2656 | 0.1713 | 0.6131 | 4.6367 | 0.9782 | 0.0026 | nan | 0.0026 | 1.0000 |
| r4b_shuffled_reason_balanced | dev_label2_subset | 38 | 2.1579 | nan | 4.1053 | 0.6579 | 0.0789 | 0.0526 | nan | 1.0000 |
| r4b_shuffled_reason_balanced | dev_d1_hidden_subset | 26 | 2.5769 | nan | 4.5000 | 0.8462 | 0.0769 | 0.0385 | nan | 1.0000 |
| r4b_shuffled_reason_balanced | dev_d1_matched_controls | 36 | 0.5000 | 0.3108 | 4.3611 | 0.8333 | 0.0556 | nan | 0.0556 | 1.0000 |
| r4b_shuffled_reason_balanced | dev_high_subset | 400 | 0.2500 | 0.5001 | 4.5275 | 0.9500 | 0.0075 | nan | 0.0075 | 1.0000 |
| r1b_score_only_balanced | train_low_reasoned_subset | 80 | 0.0000 | 1.0000 | 1.4500 | 0.0000 | 1.0000 | 1.0000 | nan | 1.0000 |
| r1b_score_only_balanced | train_clean_high_subset | 2656 | 0.1344 | 0.7078 | 4.6374 | 0.9748 | 0.0011 | nan | 0.0011 | 1.0000 |
| r1b_score_only_balanced | dev_label2_subset | 38 | 2.3684 | nan | 4.3684 | 0.7368 | 0.0000 | 0.0000 | nan | 1.0000 |
| r1b_score_only_balanced | dev_d1_hidden_subset | 26 | 2.8846 | nan | 4.8846 | 1.0000 | 0.0000 | 0.0000 | nan | 1.0000 |
| r1b_score_only_balanced | dev_d1_matched_controls | 36 | 0.5000 | 0.3108 | 4.3611 | 0.8333 | 0.0556 | nan | 0.0556 | 1.0000 |
| r1b_score_only_balanced | dev_high_subset | 400 | 0.2200 | 0.5982 | 4.5575 | 0.9525 | 0.0000 | nan | 0.0000 | 1.0000 |

## Structured Behavior

| adapter | subset | n | nonempty failure | no major failure | score cap nonnull | rubric true | full schema |
|---|---|---:|---:|---:|---:|---:|---:|
| r2n_reason_score_natural | train_low_reasoned_subset | 80 | 0.4500 | 0.1500 | 0.8500 | 0.1500 | 0.6500 |
| r2n_reason_score_natural | train_clean_high_subset | 2656 | 0.0000 | 0.9891 | 0.0109 | 0.9891 | 0.9891 |
| r2n_reason_score_natural | dev_label2_subset | 38 | 0.0000 | 0.7368 | 0.2632 | 0.7368 | 0.7368 |
| r2n_reason_score_natural | dev_d1_hidden_subset | 26 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| r2n_reason_score_natural | dev_d1_matched_controls | 36 | 0.0000 | 0.8889 | 0.1111 | 0.8889 | 0.8889 |
| r2n_reason_score_natural | dev_high_subset | 400 | 0.0000 | 0.9650 | 0.0350 | 0.9650 | 0.9650 |
| r2c_clean_reason_score_balanced | train_low_reasoned_subset | 80 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| r2c_clean_reason_score_balanced | train_clean_high_subset | 2656 | 0.0008 | 0.9842 | 0.0158 | 0.9842 | 0.9849 |
| r2c_clean_reason_score_balanced | dev_label2_subset | 38 | 0.0000 | 0.7632 | 0.2368 | 0.7632 | 0.7632 |
| r2c_clean_reason_score_balanced | dev_d1_hidden_subset | 26 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| r2c_clean_reason_score_balanced | dev_d1_matched_controls | 36 | 0.0000 | 0.9167 | 0.0833 | 0.9167 | 0.9167 |
| r2c_clean_reason_score_balanced | dev_high_subset | 400 | 0.0000 | 0.9650 | 0.0350 | 0.9650 | 0.9650 |
| r4b_shuffled_reason_balanced | train_low_reasoned_subset | 80 | 0.0250 | 0.8875 | 1.0000 | 0.0000 | 0.9250 |
| r4b_shuffled_reason_balanced | train_clean_high_subset | 2656 | 0.0011 | 0.9989 | 0.0218 | 0.9782 | 0.9992 |
| r4b_shuffled_reason_balanced | dev_label2_subset | 38 | 0.0000 | 1.0000 | 0.3421 | 0.6579 | 1.0000 |
| r4b_shuffled_reason_balanced | dev_d1_hidden_subset | 26 | 0.0000 | 1.0000 | 0.1538 | 0.8462 | 1.0000 |
| r4b_shuffled_reason_balanced | dev_d1_matched_controls | 36 | 0.0000 | 1.0000 | 0.1667 | 0.8333 | 1.0000 |
| r4b_shuffled_reason_balanced | dev_high_subset | 400 | 0.0000 | 1.0000 | 0.0500 | 0.9500 | 1.0000 |
| r1b_score_only_balanced | train_low_reasoned_subset | 80 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| r1b_score_only_balanced | train_clean_high_subset | 2656 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| r1b_score_only_balanced | dev_label2_subset | 38 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| r1b_score_only_balanced | dev_d1_hidden_subset | 26 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| r1b_score_only_balanced | dev_d1_matched_controls | 36 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| r1b_score_only_balanced | dev_high_subset | 400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Failure Type Evaluation

| adapter | subset | gold failure n | micro-F1 | macro-F1 | missing recall | factual/rubric recall | insufficient recall | task recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| r2n_reason_score_natural | train_low_reasoned_subset | 80 | 0.5345 | 0.2719 | 0.6400 | nan | 0.4783 | 0.0000 |
| r2n_reason_score_natural | train_clean_high_subset | 0 | nan | nan | nan | nan | nan | nan |
| r2n_reason_score_natural | dev_label2_subset | 0 | nan | nan | nan | nan | nan | nan |
| r2n_reason_score_natural | dev_d1_hidden_subset | 26 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| r2n_reason_score_natural | dev_d1_matched_controls | 0 | nan | nan | nan | nan | nan | nan |
| r2n_reason_score_natural | dev_high_subset | 0 | nan | nan | nan | nan | nan | nan |
| r2c_clean_reason_score_balanced | train_low_reasoned_subset | 80 | 1.0000 | 1.0000 | 1.0000 | nan | 1.0000 | 1.0000 |
| r2c_clean_reason_score_balanced | train_clean_high_subset | 0 | nan | nan | nan | nan | nan | nan |
| r2c_clean_reason_score_balanced | dev_label2_subset | 0 | nan | nan | nan | nan | nan | nan |
| r2c_clean_reason_score_balanced | dev_d1_hidden_subset | 26 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| r2c_clean_reason_score_balanced | dev_d1_matched_controls | 0 | nan | nan | nan | nan | nan | nan |
| r2c_clean_reason_score_balanced | dev_high_subset | 0 | nan | nan | nan | nan | nan | nan |
| r4b_shuffled_reason_balanced | train_low_reasoned_subset | 80 | 0.0000 | 0.0000 | 0.0000 | nan | 0.0000 | 0.0000 |
| r4b_shuffled_reason_balanced | train_clean_high_subset | 0 | nan | nan | nan | nan | nan | nan |
| r4b_shuffled_reason_balanced | dev_label2_subset | 0 | nan | nan | nan | nan | nan | nan |
| r4b_shuffled_reason_balanced | dev_d1_hidden_subset | 26 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| r4b_shuffled_reason_balanced | dev_d1_matched_controls | 0 | nan | nan | nan | nan | nan | nan |
| r4b_shuffled_reason_balanced | dev_high_subset | 0 | nan | nan | nan | nan | nan | nan |
| r1b_score_only_balanced | train_low_reasoned_subset | 80 | 0.0000 | 0.0000 | 0.0000 | nan | 0.0000 | 0.0000 |
| r1b_score_only_balanced | train_clean_high_subset | 0 | nan | nan | nan | nan | nan | nan |
| r1b_score_only_balanced | dev_label2_subset | 0 | nan | nan | nan | nan | nan | nan |
| r1b_score_only_balanced | dev_d1_hidden_subset | 26 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| r1b_score_only_balanced | dev_d1_matched_controls | 0 | nan | nan | nan | nan | nan | nan |
| r1b_score_only_balanced | dev_high_subset | 0 | nan | nan | nan | nan | nan | nan |

## Guardrails

- D2 does not read the test split.
- D2 does not train any model.
- Human rationale is used only through A0/D1 evaluation references, never as prompt input.
- Raw predictions, configs, logs, and prompt datasets remain gitignored.
- This diagnostic should not recommend test/all_5536.
