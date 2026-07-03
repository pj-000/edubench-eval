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

## Evaluation Guardrails

- Evaluation uses the original question-disjoint dev split, not a balanced training distribution.
- No test split is read by this collection script.
- The collector parses only generated assistant text and gold labels from the dev reference table.
