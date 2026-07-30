# P3 versus P2 rationale blind audit

This is an exploratory Codex-agent model preference audit, not human or expert correctness. Both evaluators are from the Codex GPT-5.6 family, so evaluator-family independence is not satisfied.

| Evaluator | Stage | Dimension | P3 win / tie / loss | Tie-adjusted preference (95% CI) |
|---|---|---|---:|---:|
| codex_sol | score_blind | rubric_alignment | 23 / 78 / 19 | 0.517 [0.471, 0.562] |
| codex_sol | score_blind | answer_grounding | 17 / 96 / 7 | 0.542 [0.504, 0.579] |
| codex_sol | score_blind | specificity | 28 / 68 / 24 | 0.517 [0.463, 0.571] |
| codex_sol | score_blind | unsupported_claims_control | 18 / 90 / 12 | 0.525 [0.475, 0.571] |
| codex_sol | score_blind | completeness | 24 / 76 / 20 | 0.517 [0.463, 0.571] |
| codex_sol | score_blind | repetition_control | 14 / 93 / 13 | 0.504 [0.463, 0.546] |
| codex_sol | score_blind | overall_preference | 34 / 56 / 30 | 0.517 [0.454, 0.579] |
| codex_sol | score_visible | score_rationale_consistency | 32 / 57 / 31 | 0.504 [0.450, 0.558] |
| codex_sol | score_visible | overall_scoring_justification_usefulness | 32 / 57 / 31 | 0.504 [0.450, 0.558] |
| codex_sol | score_visible | overall_preference | 32 / 57 / 31 | 0.504 [0.450, 0.558] |
| codex_terra | score_blind | rubric_alignment | 19 / 81 / 20 | 0.496 [0.442, 0.550] |
| codex_terra | score_blind | answer_grounding | 20 / 80 / 20 | 0.500 [0.438, 0.562] |
| codex_terra | score_blind | specificity | 26 / 73 / 21 | 0.521 [0.458, 0.583] |
| codex_terra | score_blind | unsupported_claims_control | 24 / 78 / 18 | 0.525 [0.467, 0.588] |
| codex_terra | score_blind | completeness | 22 / 78 / 20 | 0.508 [0.446, 0.571] |
| codex_terra | score_blind | repetition_control | 5 / 111 / 4 | 0.504 [0.483, 0.525] |
| codex_terra | score_blind | overall_preference | 23 / 77 / 20 | 0.512 [0.450, 0.575] |
| codex_terra | score_visible | score_rationale_consistency | 23 / 78 / 19 | 0.517 [0.454, 0.579] |
| codex_terra | score_visible | overall_scoring_justification_usefulness | 32 / 64 / 24 | 0.533 [0.463, 0.600] |
| codex_terra | score_visible | overall_preference | 36 / 59 / 25 | 0.546 [0.475, 0.617] |

All forced-completed outputs remain in the primary aggregate. Forced-completion strata are diagnostic only.
