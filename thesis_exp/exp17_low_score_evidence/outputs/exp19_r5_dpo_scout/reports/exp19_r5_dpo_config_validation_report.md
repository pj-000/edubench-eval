# Exp19-R5 DPO Config Validation

This pre-flight check validates R5C/R5D/R5E LLaMA-Factory DPO configs before scout training.
Adapter path checks are warnings because local and server environments can differ.
The scout train script validates these base configs, then overrides `MAX_STEPS=100` and
`PREF_BETA=0.05` by default at runtime.

- configs checked: 4
- failed configs: 0
- configs with warnings: 0

| config | status | dataset | stage | pref_beta | warnings | errors |
|---|---|---|---|---:|---|---|
| `llamafactory_qwen3_4b_r5c_dpo_from_r2c.yaml` | PASS | `edubench_r5c_score_risk_dpo_train` | dpo | 0.1 |  |  |
| `llamafactory_qwen3_4b_r5c_dpo_from_r1b.yaml` | PASS | `edubench_r5c_score_risk_dpo_train` | dpo | 0.1 |  |  |
| `llamafactory_qwen3_4b_r5d_dpo_from_r2c.yaml` | PASS | `edubench_r5d_evidence_consistency_dpo_train` | dpo | 0.1 |  |  |
| `llamafactory_qwen3_4b_r5e_dpo_control_from_r2c.yaml` | PASS | `edubench_r5e_hard_synthetic_dpo_control_train` | dpo | 0.1 |  |  |

## Guardrails

- This script does not read train/dev/test examples.
- This script does not train or run inference.
- Full DPO JSON files may remain gitignored and server-side.
