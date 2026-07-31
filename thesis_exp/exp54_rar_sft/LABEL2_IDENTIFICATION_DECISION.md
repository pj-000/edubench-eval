# Label-2 mechanism identification decision

## Decision

The frozen RAR-SFT and Field-DPO experiments remain valid master-thesis
results. The mechanism audit does **not** identify a sufficiently stable,
dominant failure mechanism to justify inventing or training another preference
loss at this checkpoint.

The strongest hypothesis for further investigation is:

> The observed-consensus score-2 target collapses a partly unstable 2/3
> boundary, while some supplied rubrics do not make that boundary operationally
> reproducible.

This is a hypothesis, not an established causal finding. It is supported by
the original rater patterns and an independent two-model-agent blind review;
the latter is not human validity evidence.

## What has been established

- P1 Field-DPO makes a small favorable movement relative to the frozen R3
  comparator: mean Exact changes from 0.7038 to 0.7103, mean MAE from 0.3409
  to 0.3353, and mean low-to-high count from 10.33 to 9.00.
- P1 does not solve the low-score tail: mean Label-2 recall is 0.0476.
- Five-way frozen score probabilities show that P1 generally moves some mass
  away from scores 4/5 and toward scores 2/3, but score 4 often remains the
  dominant option.
- Structured decoding is not the main mechanism. Ordinary cross-fitted vector
  scaling also does not recover the Label-2 failures.
- Train-prior neutralization and missing direct 2-over-3 pair coverage explain
  subsets of failures, but neither passes the preregistered dominance gate.
- Five of 14 Label-2 dev records meet the automatic three-rater ambiguity rule.
- Both model agents flag a non-unique 2/3 boundary for 10 of 14 records and a
  missing or vague rubric criterion for 8 of 14 records. Only 2 of 14 are
  judged uniquely score 2 by both agents. Agreement is moderate for the score
  questions and weak for rubric vagueness.

## Claims that are not allowed

- The audit does not prove that the human aggregate label is wrong.
- The model-agent review cannot be described as a human audit.
- The audit does not establish that rubric ambiguity causes model error.
- It does not show that a new loss, DPO variant, or uncertainty head will solve
  the problem.
- No CCF-A-level method contribution has yet been established by this audit.

## Research continuation gate

Further method development is permitted only after a no-training feasibility
test asks whether frozen model uncertainty tracks the original three-rater
boundary uncertainty.

For every locked dev row, and separately for R3 and P1 across all seeds, the
test will compare:

- human disagreement: rater range, rater variance, and whether the rounded
  target lies on a mixed adjacent-rating boundary;
- model uncertainty: five-way entropy, top-two probability margin, ordinal
  predictive variance, and score-2-versus-score-3 log-odds;
- error severity: absolute error and low-to-high failure.

All associations must use question-clustered uncertainty and report each seed
separately. This is descriptive identification, not a new trained result.

The continuation decision is:

1. If ordinary frozen probability uncertainty already tracks rater
   disagreement reliably, use it as a strong baseline and do not claim a new
   representation principle.
2. If disagreement is systematic but absent from frozen uncertainty, a
   disagreement-aware ordinal/rubric-boundary estimator becomes a testable
   research direction. Existing soft-label, ordinal, calibration, and
   multi-rater baselines must be falsified before proposing a new method.
3. If neither human disagreement nor model uncertainty has a stable
   relationship with the failures, stop algorithm expansion and write the
   completed work with bounded claims.

No GPU training, new preference pairs, test access, or method naming is
authorized by this decision. The immediate next action is the frozen,
no-training uncertainty-versus-disagreement feasibility analysis.
