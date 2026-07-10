# Exp27L Group Cross-Fitted Calibration Report

This report is a train-only, question-key OOF audit. Exp27J adjudication is a silver target, not external human gold.

## Score Calibration

- Qwen weighted MAE on representative rows: 0.5356885147324113
- Soft-fusion weighted MAE on representative rows: 0.7747143716175586
- Soft-minus-Qwen MAE 95% cluster-bootstrap CI: [0.07339897775105231, 0.40710328058356926]

## Severe Human-Silver Conflict Detection

- Logistic OOF AUPRC on representative rows: 0.23007805109588958
- Qwen-human-gap heuristic AUPRC on representative rows: 0.47181852974752125
- AUPRC difference 95% cluster-bootstrap CI: [-0.44495346000496205, -0.08358475414810951]

## Policy Boundary

Policy A is fixed. Policy B selects its review budget within each outer training fold using inner OOF predictions, then applies those thresholds unchanged to the held-out fold.

## Locked Decision

No Exp27M trainer data is produced here. Both training flags remain false until the separate external blind-review lockbox is filled and independently adjudicated.
