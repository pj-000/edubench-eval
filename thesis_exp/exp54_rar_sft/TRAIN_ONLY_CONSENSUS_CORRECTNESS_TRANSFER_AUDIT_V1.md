# Train-only Consensus Correctness and Transfer Audit v1

## Status

This protocol is frozen before running the row-level audit. It reuses the 16
already-rated train-only pilot items and makes no model call, performs no
training, uses no GPU, and accesses neither dev nor test.

## Definitions

For five panel scores, `mode_share` is the largest score count divided by five.
If multiple scores tie for the largest count, `mode` is null and the item is
classified as `Pweak`.

- `H0`: all three human scores are identical.
- `H1`: exactly two adjacent human scores occur, with a 2--1 split.
- `H2`: the human-score range is at least two.
- `P5`: mode share is 1.0.
- `P4`: mode share is at least 0.8 but below 1.0.
- `Pweak`: mode share is below 0.8 or the mode is tied.
- consensus error: mode share is at least 0.8 and panel mode differs from
  `label_5`.
- severe consensus error: consensus error with absolute mode error at least 2.
- severe L2H/H2L: mode error is respectively at least +2 / at most -2.
- clarification fix: original mode is wrong and clarified mode equals `label_5`.
- clarification harm: original mode equals `label_5` and clarified mode is wrong.
- per-seed R3 direction transfer: an erroneous R3 seed and the GPT original
  mode have the same nonzero signed error direction.
- boundary transfer: the GPT original mode crosses the selected adjacent
  boundary from `label_5` in the same direction as that R3 seed.

The historical aggregate label is an operational target, not latent truth.
All counts are stratified by `H0/H1/H2`; an `H2` disagreement cannot establish
a human-stable judge error.

## Frozen direction rules

Authorize a common-mode-error verification pilot only if, among the 12
recurrent failures, all hold:

1. at least four original-condition high-consensus errors;
2. at least three remain high-consensus errors after clarification;
3. they span at least two adjacent boundaries;
4. at least three are `H0`, or none is `H2`;
5. at least two are severe L2H/H2L errors.

Stop general LLM-as-a-Judge method development if GPT original fixes at least
10/12 recurrent R3 failures, has at most one high-consensus error, and all four
clear anchors remain correct.

Classify the main issue as random instability if at least 4/12 recurrent
failures are `Pweak`. Classify human non-uniqueness as dominant if at least
6/12 recurrent failures are `H2`, or if GPT modes usually fall within the set
of three human scores. These are research-decision rules, not population-level
hypothesis tests.

If no branch fires, the result is `NO_DIRECTION_SELECTED_MORE_EVIDENCE_NEEDED`;
it does not authorize inventing or training a new method.
