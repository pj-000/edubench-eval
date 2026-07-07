# Exp24R ORC-DPO Failure Diagnosis

This diagnosis explains why Exp24 score-channel ORC-DPO did not pass the dev success rule.
It uses existing dev predictions only and writes no raw generations.

## Main Findings

- Low dev samples analyzed: 57.
- D1 hidden cases analyzed: 26.
- DPO0 vs R2C changed predictions for 141 samples (0.1274).
- ORC-B vs DPO0 changed predictions for 20 samples (0.0181).
- ORC-B fixed 0 DPO0 low-to-high cases and worsened 1.
- R7D vs R7E fixed 16 low-to-high cases.
- ORC-B reduced D1 hidden predictions vs DPO0 for 0/26 cases.
- R7D reduced D1 hidden predictions vs R2C for 9/26 cases.

## Label-2 Distribution

| run | n | mean pred | pred>=4 | exact label2 recall | pred 5 rate |
|---|---:|---:|---:|---:|---:|
| `r2c` | 38 | 4.3947 | 0.7632 | 0.0000 | 0.6316 |
| `r4b` | 38 | 4.1053 | 0.6579 | 0.0526 | 0.5526 |
| `r7d` | 38 | 3.5789 | 0.6316 | 0.2632 | 0.4211 |
| `r7e` | 38 | 4.0526 | 0.9211 | 0.0789 | 0.2105 |
| `dpo0` | 38 | 4.4737 | 0.8947 | 0.0526 | 0.6316 |
| `orc_b` | 38 | 4.5676 | 0.8947 | 0.0000 | 0.6316 |
| `orc_b_noreason` | 38 | 4.5263 | 0.8947 | 0.0000 | 0.6316 |

## Interpretation

- Same-trainer DPO0 moved the model toward high-score outputs and increased low-to-high relative to R2C.
- ORC-B's weights/margins changed too few predictions relative to DPO0 to repair low-to-high.
- Human-reason ordinary DPO (R7D) is the only compared run that materially lowers D1 hidden predictions, but it hurts MAE/QWK.
- Therefore the bottleneck is not parse or trainer failure; it is the score-only preference signal being too weak for hidden low-score evidence.

## Recommended Next Step

Do not run test or multi-seed Exp24 yet. Use this diagnosis to design SRC-DPO or hidden-failure data
expansion:

1. SRC-DPO: contrast reason-score consistency negatives, not only score-only negatives.
2. Train-only hidden-failure expansion: add more low-to-high/D1-like cases and high-score protection controls.
3. Optionally add an explicit low-tail term penalizing P(score>=4 | gold<=2) if using token probabilities.

## Missing Runs

- none

## Guardrails

- No test split is read.
- Dev labels are used for diagnosis only.
- Raw predictions are read but not written to the output directory.
- Output files contain IDs and scalar predictions only, not answers, raw generations, or human rationales.
