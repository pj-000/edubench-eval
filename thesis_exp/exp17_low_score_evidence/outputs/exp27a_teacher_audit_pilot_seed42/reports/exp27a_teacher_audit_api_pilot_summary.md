# Exp27A Teacher Audit API Pilot Summary

## Scope

- providers: qwen3.7-max, deepseek-v4-pro
- sampled packets: 20 train examples
- stages completed: blind + audit
- test labels: not read
- training: none

## API / Validation

- total annotation rows: 80
- parse failures: 0
- semantic validation errors: 0

## Blind Cross-Provider Agreement

- paired blind rows: 20
- exact score agreement: 13/20
- adjacent score agreement: 20/20
- risk_flag exact agreement: 11/20

## Per-Provider Summary

### deepseek / audit
- rows: 20
- score_distribution: `{"1": 5, "2": 3, "3": 1, "4": 4, "5": 7}`
- risk_flag_distribution: `{"borderline": 5, "clean_high": 9, "hidden_low_failure": 5, "unclear": 1}`
- score_agreement_distribution: `{"adjacent": 6, "conflict": 4, "exact": 10}`
- label_quality_distribution: `{"plausible_adjacent": 6, "reliable": 10, "suspected_conflict": 4}`
- needs_human_review_count: 4

### deepseek / blind
- rows: 20
- score_distribution: `{"1": 5, "2": 3, "3": 1, "4": 4, "5": 7}`
- risk_flag_distribution: `{"borderline": 5, "clean_high": 9, "hidden_low_failure": 5, "unclear": 1}`
- score_agreement_distribution: `{"unclear": 20}`
- label_quality_distribution: `{"unclear": 20}`
- needs_human_review_count: 20

### qwen / audit
- rows: 20
- score_distribution: `{"1": 5, "2": 4, "4": 6, "5": 5}`
- risk_flag_distribution: `{"borderline": 3, "clean_high": 9, "hidden_low_failure": 8}`
- score_agreement_distribution: `{"adjacent": 6, "conflict": 2, "exact": 12}`
- label_quality_distribution: `{"plausible_adjacent": 6, "reliable": 12, "suspected_conflict": 2}`
- needs_human_review_count: 2

### qwen / blind
- rows: 20
- score_distribution: `{"1": 5, "2": 4, "4": 6, "5": 5}`
- risk_flag_distribution: `{"borderline": 3, "clean_high": 9, "hidden_low_failure": 8}`
- score_agreement_distribution: `{"unclear": 20}`
- label_quality_distribution: `{"unclear": 20}`
- needs_human_review_count: 20

## Interpretation

- The pilot confirms that both teacher APIs can produce parseable structured outputs under the current schema.
- The blind stage does not receive original benchmark scores or recovered human reasons.
- The audit stage uses original train labels only to assess label reliability after blind scoring.
- Raw teacher outputs remain ignored; only lightweight summaries should be reviewed/committed.