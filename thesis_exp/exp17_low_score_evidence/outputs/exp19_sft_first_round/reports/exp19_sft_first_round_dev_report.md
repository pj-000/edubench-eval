# Exp19 First-Round SFT Dev Evaluation

This report summarizes LLaMA-Factory `do_predict` outputs for R1/R2/R4 on the original dev split.
Raw generated predictions remain in gitignored `dev_predictions/` directories.

| run | n | parse | MAE | QWK | bias | exact | low-to-high | label2 recall | label5 recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R1 score-only natural | 1107 | 1.0000 | 0.3948 | 0.5536 | 0.2087 | 0.6893 | 31 (0.5439) | 0.0000 | 0.8615 |
| R2 reason-score balanced | 1107 | 1.0000 | 0.4083 | 0.5336 | 0.2060 | 0.6811 | 32 (0.5614) | 0.0000 | 0.8615 |
| R4 shuffled reason control | 1107 | 1.0000 | 0.3993 | 0.5894 | 0.1391 | 0.6739 | 22 (0.3860) | 0.0526 | 0.8112 |

## Parse Summary

- R1 score-only natural: success=1107/1107 (1.0000), json=1107, regex=0, failed=0
- R2 reason-score balanced: success=1107/1107 (1.0000), json=1107, regex=0, failed=0
- R4 shuffled reason control: success=1107/1107 (1.0000), json=1107, regex=0, failed=0

## Interpretation

- R1 is score-only natural SFT.
- R2 is the non-control reason-score balanced SFT run.
- R4 is a shuffled-reason control; if R4 matches or beats R2, reason semantics are not proven.
- R2 beats R1: `False`.
- R2 beats R4: `False`.
- recommendation: R4 shuffled control is competitive with or better than R2; do second-round SFT ablation before R3/DPO.

## D1 Hidden Evaluation

| run | n | mean pred | pred>=4 | pred=5 | label2 recall | control mean | control-case gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| R1 score-only natural | 26 | 4.7692 | 1.0000 | 0.7692 | 0.0000 | 4.4444 | -0.5679 |
| R2 reason-score balanced | 26 | 4.9231 | 1.0000 | 0.9231 | 0.0000 | 4.4444 | -0.5926 |
| R4 shuffled reason control | 26 | 4.1538 | 0.7308 | 0.5385 | 0.0385 | 4.1667 | -0.2716 |

## Structured Field Quality

- R1 score-only natural: full_schema=0.0000, major_failures=0.0000, score_cap=0.0000, rubric_satisfied=0.0000.
- R2 reason-score balanced: full_schema=0.9268, major_failures=1.0000, score_cap=1.0000, rubric_satisfied=1.0000.
- R4 shuffled reason control: full_schema=1.0000, major_failures=1.0000, score_cap=1.0000, rubric_satisfied=1.0000.

## Failure Type Evaluation

- R1 score-only natural: micro-F1=0.0000, macro-F1=0.0000, D1 nonempty failure=0.0000, high-control nonempty failure=0.0000.
- R2 reason-score balanced: micro-F1=0.0000, macro-F1=0.0000, D1 nonempty failure=0.0000, high-control nonempty failure=0.0000.
- R4 shuffled reason control: micro-F1=0.0000, macro-F1=0.0000, D1 nonempty failure=0.0000, high-control nonempty failure=0.0000.

## Evaluation Guardrails

- Evaluation uses the original question-disjoint dev split, not a balanced training distribution.
- No test split is read by this collection script.
- The collector parses only generated assistant text and gold labels from the dev reference table.
- D1 annotations are used only as evaluation references, not as model prompts or training labels.
