# SORC-DPO P3-versus-P2 rationale blind-audit results

## Question

P3 adds a rationale-alignment preference block to the P2 score-only
risk-conditioned objective. This audit asks whether that block improves the
visible scoring rationale, rather than merely fitting the frozen train
rationale pairs.

The audit reused the frozen 40-row low-score-enriched dev sample:

- all 20 Label-1/2 rows;
- 20 fixed Label-3/4/5 rows;
- seeds 42, 43, and 44;
- 120 paired P3-versus-P2 instances;
- both A/B and B/A orientations;
- all forced-completed outputs retained in the primary result.

Two isolated Codex agents judged score-blind and score-visible presentations.
Neither agent saw the answer key, arm identity, gold score, checkpoint, dev
metric, or the other agent's judgments.

## Overall result

| Agent | Stage | Primary dimension | P3 win / tie / loss | Tie-adjusted preference | 95% record-cluster CI |
|---|---|---|---:|---:|---:|
| Codex Sol | Score blind | Overall rationale preference | 34 / 56 / 30 | 0.517 | [0.454, 0.579] |
| Codex Terra | Score blind | Overall rationale preference | 23 / 77 / 20 | 0.512 | [0.450, 0.575] |
| Codex Sol | Score visible | Overall scoring-justification usefulness | 32 / 57 / 31 | 0.504 | [0.450, 0.558] |
| Codex Terra | Score visible | Overall scoring-justification usefulness | 32 / 64 / 24 | 0.533 | [0.463, 0.600] |

None of the four overall intervals excludes `0.5`. The two agents agreed on
only `56.7%` of score-visible overall judgments (`Cohen's κ = 0.305`).

Sol's score-blind answer-grounding endpoint was mildly positive (`0.542`,
95% CI `[0.504, 0.579]`), but Terra did not replicate it (`0.500`, 95% CI
`[0.438, 0.562]`). Terra preferred P3 on the low-score score-visible subset,
whereas Sol did not. These evaluator-specific findings are diagnostic, not a
robust rationale-alignment effect.

## Interpretation

The train-only mechanism diagnostic showed that P3 moved the aligned-versus-
shuffled rationale contrast in the intended direction for `90.15%` of its
rationale pairs. This confirms train-pair learning.

The blind audit does not show a stable corresponding improvement in visible
dev rationale quality. The correct conclusion is:

> The P3 rationale-preference block learned its frozen training preferences,
> but did not establish a robust cross-agent improvement in visible
> score-rationale alignment or overall scoring-justification usefulness on
> the fixed dev audit sample.

Therefore:

- P3 must not be claimed to improve internal reasoning;
- P3 must not be claimed to improve visible rationale alignment over P2;
- the small P3-versus-P2 score changes cannot be attributed to demonstrably
  better rationales;
- the rationale block remains an informative near-zero ablation.

The result remains model-based preference, not human correctness. Both
evaluators are variants of the Codex GPT-5.6 family, so evaluator-family
independence is not satisfied.

Test remains sealed.
