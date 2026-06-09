# Exp6-7 Batch96 Full Generation Readiness

## Inputs Reviewed

- Filtered synthetic samples: **91**
- Leakage status: **PASS**
- Label distribution: **1=37, 2=39, 3=15**
- Language distribution: **en=47, zh=44**
- Metric coverage: **12**
- Error type coverage: **7**

## Manual Review Status

Manual decision fields are intentionally blank in `batch96_manual_spotcheck_decisions.csv`.
This package is a full 91-row human review sheet, not an automated acceptance decision.

- Full/top-up generation can start: **NO**
- Reason: **pending full spot-check**
- Top-up generation can start after review: **YES only if usable_after_revision >= 80% and leakage remains PASS**
- If usable_after_revision < 80%: **NO, prompt/filter revision needed**
- Exp6 training can start: **NO until full filtered pool complete**

## Heuristic Risk Prefill

- Rows needing human attention by heuristic: **28**
- Possible label too low: **6**
- Possible error type mismatch: **10**
- Possible artifact phrase: **0**

Heuristic flags are only triage aids. Final decisions must be filled manually using
`accept`, `revise_label`, `revise_error_type`, or `reject`.
