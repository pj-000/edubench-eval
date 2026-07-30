# Exp54 Codex-agent exploratory rationale blind audit

## Purpose and scope

This audit evaluates the already frozen RAR-SFT rationale control:

- target: R3 answer-aligned, label-consistent human rationale supervision;
- comparator: R2 token-frequency-matched shuffled-rationale control;
- sample: the preregistered 40-row, low-score-enriched dev sample;
- seeds: 42, 43, and 44;
- endpoint: score-visible rationale consistency and overall scoring-
  justification usefulness.

It does **not** directly compare the post-DPO P3 and P2 outputs. It tests the
scientific premise used by the later preference experiment: whether semantic
alignment contributes value beyond rationale-token exposure.

## Agent execution

Two isolated Codex agents judged the same 240 anonymous presentations. The
tasks contained question, answer, metric, rubric, and candidates A/B, but no
arm, seed, gold label, forced-completion flag, or answer mapping.

The two agents are variants of the same GPT-5.6 family. Consequently:

- evaluator-family independence is not satisfied;
- the preregistered two-family formal audit is not complete;
- the result is an exploratory Codex-agent model preference, not human or
  expert correctness.

Both A/B orientations were judged inside each agent's persistent context.
Although no orientation conflicts occurred, that zero-conflict observation
cannot be treated as an independent position-bias check.

## Primary result: R3 versus R2

| Agent | Dimension | R3 win / tie / loss | Tie-adjusted preference | 95% cluster-bootstrap CI |
|---|---|---:|---:|---:|
| Codex Sol | Score–rationale consistency | 97 / 8 / 15 | 0.842 | [0.767, 0.912] |
| Codex Sol | Overall usefulness | 97 / 9 / 14 | 0.846 | [0.771, 0.912] |
| Codex Terra | Score–rationale consistency | 60 / 46 / 14 | 0.692 | [0.629, 0.754] |
| Codex Terra | Overall usefulness | 60 / 46 / 14 | 0.692 | [0.629, 0.754] |

Intervals use 10,000 record-cluster bootstrap replicates with seed 20260728
and carry all three training-seed pairs whenever a record is sampled.
Forced-completed outputs remain included.

## Cross-agent agreement

| Dimension | Exact agreement | Cohen's κ |
|---|---:|---:|
| Score–rationale consistency | 52.5% | 0.145 |
| Overall usefulness | 53.3% | 0.157 |
| Overall preference | 53.3% | 0.157 |

Both agents favor R3, but they disagree substantially about how often the
difference is decisive rather than a tie. This limits claims about effect
magnitude and demonstrates why the result cannot be presented as a substitute
for independent evaluator families.

## Permitted conclusion

The appropriate conclusion is:

> On the fixed low-score-enriched audit sample, two independent Codex-agent
> runs both preferred R3's semantically aligned rationales over R2's shuffled
> rationales. This provides exploratory model-based evidence that semantic
> answer–rationale alignment adds visible justification value beyond matched
> rationale-token exposure. Low cross-agent agreement and shared model-family
> provenance prevent a formal correctness or evaluator-robustness claim.

This result complements the preference-training dev result:

- RAR-SFT establishes a directional semantic-alignment rationale effect;
- SORC-DPO at learning rate `5e-6` establishes a directional low-score-risk
  effect;
- neither result alone is statistically or evaluator-family decisive.

Test remains sealed.
