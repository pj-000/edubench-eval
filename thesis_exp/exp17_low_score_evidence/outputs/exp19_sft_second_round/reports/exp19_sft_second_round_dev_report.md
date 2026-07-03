# Exp19 Second-Round SFT Dev Evaluation

This report summarizes LLaMA-Factory do_predict outputs for R1b/R2n/R2c/R4b on the original dev
split.
Raw generated predictions remain in gitignored `dev_predictions/` directories.

| run | n | parse | MAE | QWK | bias | exact | low-to-high | label2 recall | label5 recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R1b score-only balanced | 1107 | 1.0000 | 0.3975 | 0.5565 | 0.1716 | 0.6865 | 30 (0.5263) | 0.0000 | 0.8507 |
| R2n reason-score natural | 1107 | 1.0000 | 0.3966 | 0.5535 | 0.1942 | 0.6865 | 30 (0.5263) | 0.0000 | 0.8471 |
| R2c clean reason-score balanced | 1107 | 1.0000 | 0.4219 | 0.4982 | 0.2195 | 0.6712 | 34 (0.5965) | 0.0000 | 0.8435 |
| R4b shuffled reason balanced | 1107 | 1.0000 | 0.4074 | 0.5617 | 0.1400 | 0.6802 | 27 (0.4737) | 0.0526 | 0.8363 |

## Parse Summary

- R1b score-only balanced: success=1107/1107 (1.0000), json=1107, regex=0, failed=0
- R2n reason-score natural: success=1107/1107 (1.0000), json=1107, regex=0, failed=0
- R2c clean reason-score balanced: success=1107/1107 (1.0000), json=1107, regex=0, failed=0
- R4b shuffled reason balanced: success=1107/1107 (1.0000), json=1107, regex=0, failed=0

## Interpretation

- best overall run by MAE: `r2n_reason_score_natural`.
- best low-risk run by low-to-high: `r4b_shuffled_reason_balanced`.
- proceed to R3: `False`.
- proceed to DPO: `True`.
- recommendation: Score-only balanced or shuffled-reason control is stronger on low-risk metrics; prefer risk-balanced DPO or target-schema revision before R3.

## D1 Hidden Evaluation

| run | n | mean pred | pred>=4 | pred=5 | label2 recall | control mean | control-case gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| R1b score-only balanced | 26 | 4.8846 | 1.0000 | 0.8846 | 0.0000 | 4.3611 | -0.8025 |
| R2n reason-score natural | 26 | 4.7308 | 1.0000 | 0.7308 | 0.0000 | 4.4444 | -0.4938 |
| R2c clean reason-score balanced | 26 | 4.8846 | 1.0000 | 0.8846 | 0.0000 | 4.4167 | -0.6420 |
| R4b shuffled reason balanced | 26 | 4.5000 | 0.8462 | 0.7692 | 0.0385 | 4.3611 | -0.4198 |

## Structured Field Quality

- R1b score-only balanced: full_schema=0.0000, major_failures=0.0000, score_cap=0.0000, rubric_satisfied=0.0000.
- R2n reason-score natural: full_schema=0.9386, major_failures=1.0000, score_cap=1.0000, rubric_satisfied=1.0000.
- R2c clean reason-score balanced: full_schema=0.9413, major_failures=1.0000, score_cap=1.0000, rubric_satisfied=1.0000.
- R4b shuffled reason balanced: full_schema=1.0000, major_failures=1.0000, score_cap=1.0000, rubric_satisfied=1.0000.

## Failure Type Evaluation

- R1b score-only balanced: micro-F1=0.0000, macro-F1=0.0000, D1 nonempty failure=0.0000, high-control nonempty failure=0.0000.
- R2n reason-score natural: micro-F1=0.0000, macro-F1=0.0000, D1 nonempty failure=0.0000, high-control nonempty failure=0.0000.
- R2c clean reason-score balanced: micro-F1=0.0000, macro-F1=0.0000, D1 nonempty failure=0.0000, high-control nonempty failure=0.0000.
- R4b shuffled reason balanced: micro-F1=0.0000, macro-F1=0.0000, D1 nonempty failure=0.0000, high-control nonempty failure=0.0000.

## Evaluation Guardrails

- Evaluation uses the original question-disjoint dev split, not a balanced training distribution.
- No test split is read by this collection script.
- The collector parses only generated assistant text and gold labels from the dev reference table.
- D1 annotations are used only as evaluation references, not as model prompts or training labels.
