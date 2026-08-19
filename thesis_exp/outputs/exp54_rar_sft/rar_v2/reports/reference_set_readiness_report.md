# RAR-SFT V2 reference-set readiness report

## Boundary

- Status: `REFERENCE_SETS_READY`
- Train rows: `2654`
- Dev/test accessed: `false`
- API/model/GPU/training used: `false`
- Semantic filtering or rewriting: `false`

## Counts

- Reason-covered rows: `1612`
- Score-only fallback rows: `1042`
- R1 all-rater references: `4836`
- R3 label-consistent references: `3934`
- Dissenting references retained for R1: `902`
- R3 reference-count distribution: `{"2": 902, "3": 710}`
- Explicit-score redaction events: `41`
- Remaining detected explicit score reports: `0`

## Deterministic checks

- `train_rows_exact`: `PASS`
- `row_ids_match`: `PASS`
- `all_rater_active_rows_exact`: `PASS`
- `label_consistent_active_rows_exact`: `PASS`
- `no_reason_rows_exact`: `PASS`
- `all_rater_reference_count_exact`: `PASS`
- `label_consistent_reference_count_exact`: `PASS`
- `dissenting_reference_count_exact`: `PASS`
- `consistent_reference_distribution_exact`: `PASS`
- `all_active_reasons_nonempty`: `PASS`
- `all_reference_ids_unique`: `PASS`
- `explicit_score_leakage_zero`: `PASS`
- `consistent_subset_of_all`: `PASS`

## Interpretation

This stage validates source alignment, deterministic cleaning, provenance, and reference-set
membership only. It does not claim semantic correctness or human validation. R2 donor matching,
training manifests, token-budget auditing, model snapshot locking, and all training remain locked.
