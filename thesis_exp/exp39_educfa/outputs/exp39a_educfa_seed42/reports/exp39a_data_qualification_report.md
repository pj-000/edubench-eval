# Exp39A EduCFA data qualification

- Status: **DATA_QUALIFICATION_NO_GO**
- Completed / accepted: `240 / 73`
- Accepted targets: `{"1": 12, "2": 37, "3": 24}`
- Operator success rate: `0.9667`
- Rubric failure verification rate: `0.9667`
- Mean accepted edit / length ratio: `0.1320` / `0.9268`
- Duplicate rate: `0.0042`
- Gates: `{"accepted_rows": false, "accepted_target1": false, "accepted_target2": false, "accepted_target3": false, "api_schema_success": true, "duplicate_rate": true, "mean_edit_ratio": true, "no_dev_test_access": true, "no_low_verified_high": true, "operator_success_rate": true, "rubric_failure_verification_rate": true, "source_counterfactual_qkey_match": true}`
- Failed gates: `["accepted_rows", "accepted_target1", "accepted_target2", "accepted_target3"]`
- Recommend GroupCV: `false`
- No original train labels were replaced.
- No paper-like dev/test data were accessed.

## Leading rejection reasons

- `edit_ratio_out_of_range`: `107` / `240` (`0.4458`)
- `target_outside_verifier_range`: `72` / `240` (`0.3000`)
- `generator_verifier_intersection_empty`: `68` / `240` (`0.2833`)
- `length_ratio_out_of_range`: `54` / `240` (`0.2250`)
- `verifier_center_too_far`: `28` / `240` (`0.1167`)
- `insufficient_verified_score_drop`: `26` / `240` (`0.1083`)
- `changed_span_invalid`: `13` / `240` (`0.0542`)
- `target3_center_above_3`: `13` / `240` (`0.0542`)

## Pre-registered stop

The accepted pool does not satisfy the frozen sample-count gates. GroupCV smoke/formal training is therefore not permitted.
The acceptance thresholds were not relaxed after observing results, and the same source/prompt campaign must not be replayed as a new formal attempt.
