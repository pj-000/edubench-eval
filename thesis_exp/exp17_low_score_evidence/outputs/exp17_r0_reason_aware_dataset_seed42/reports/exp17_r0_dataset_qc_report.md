# Exp17-R0 Reason-Aware Dataset QC Report

R0 is a future causal/instruction-model experiment. This run prepares redacted samples and QC only;
no model is trained.

- train redacted sample count: 80
- dev eval redacted sample count: 27
- train candidate count: 111
- train usable count: 84
- max question group rate: 0.1000
- score phrase conflict count: 2
- rationale length mean/p50/max words: 5.04/5/11
- safe for SFT: `True`

## Guardrails

- Human rationale is not included in the input.
- Train rationales are represented only as target fields.
- Dev D1 annotations are evaluation-only and must not be used as train labels.
- Full raw SFT jsonl is intentionally not written by this script.
