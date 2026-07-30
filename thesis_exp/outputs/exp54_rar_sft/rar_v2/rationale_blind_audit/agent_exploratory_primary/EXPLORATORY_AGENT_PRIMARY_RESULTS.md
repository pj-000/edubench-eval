# Exp54 exploratory agent primary rationale preference

**Claim scope:** Codex-agent exploratory primary score-visible preference.

Evaluator-family independence is not satisfied. This artifact does not complete the formal preregistered two-family blind audit.

Both orientations were judged within each agent's persistent context. Therefore the zero orientation-conflict result is not an independent position-bias diagnostic and must not be interpreted as one.

| Agent evaluator | Comparison | Dimension | Win / tie / loss | Tie-adjusted preference (95% CI) | Net preference (95% CI) |
|---|---|---|---:|---:|---:|
| codex_sol | R3_vs_R2 | score_rationale_consistency | 97 / 8 / 15 | 0.842 [0.767, 0.912] | 0.683 [0.533, 0.825] |
| codex_sol | R3_vs_R2 | overall_scoring_justification_usefulness | 97 / 9 / 14 | 0.846 [0.771, 0.912] | 0.692 [0.542, 0.825] |
| codex_sol | R3_vs_R2 | overall_preference | 97 / 9 / 14 | 0.846 [0.771, 0.912] | 0.692 [0.542, 0.825] |
| codex_terra | R3_vs_R2 | score_rationale_consistency | 60 / 46 / 14 | 0.692 [0.629, 0.754] | 0.383 [0.258, 0.508] |
| codex_terra | R3_vs_R2 | overall_scoring_justification_usefulness | 60 / 46 / 14 | 0.692 [0.629, 0.754] | 0.383 [0.258, 0.508] |
| codex_terra | R3_vs_R2 | overall_preference | 60 / 46 / 14 | 0.692 [0.629, 0.754] | 0.383 [0.258, 0.508] |

Intervals use 10,000 record-cluster bootstrap replicates with seed 20260728, carrying all three training-seed pairs whenever a record cluster is sampled.

Forced-completed outputs remain in the primary aggregate; preplanned forced-completion strata are diagnostic.
