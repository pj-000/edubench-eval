# Exp27L-R1 Balanced Crossfit Report

This is a train-only internal silver-reference diagnostic. It does not construct trainer data.

## Score Fusion

- Qwen weighted MAE: 0.5356885147324113
- NLL-selected global fusion weighted MAE: 0.6338252979870773
- Fusion-minus-Qwen MAE 95% cluster CI: [-0.13258566603525856, 0.09019633680326794]
- Fusion lock status: not_locked

## Severe Human-Silver Conflict Ranking

- Full Logistic tie-safe AUPRC: 0.349133663444419
- Core Logistic tie-safe AUPRC: 0.39387572860885794
- Qwen-human gap tie-safe AUPRC: 0.4479676747810074
- Learned-risk lock status: learned_risk_model_negative_use_simple_disagreement

No external review is complete, so all downstream training gates remain false.
