# LLM-as-a-Judge Specification-Intervention Pilot v1

## Status and purpose

This document freezes a train-only, model-only feasibility pilot before any
rubric clarification is written or any formal judge call is made. It does not
authorize model training, access to dev/test, or a CCF-A method claim.

The research question is:

> When independent LLM-judge instances disagree under a natural-language
> scoring specification, can an audited, policy-preserving clarification
> reduce a distinct component of that disagreement, and does the value of
> clarification differ from the value of another judge call?

The population in this pilot is explicitly **LLM judge instances**, not human
raters. Results cannot be described as human-rating validity or as recovery of
the rubric author's latent intent.

## Why this is not another scoring loss

Passive observations identify only a judge distribution under the observed
specification. They do not generally identify what the same judge population
would do after a controlled clarification. The candidate object therefore
depends on a specification intervention rather than a new token mask, loss
weight, DPO offset, or uncertainty head.

For an ordered score in `1..5`, define

```text
d_ord(y, y') = abs(y - y') / 4
```

and, for specification condition `z` in `{original, clarified}`,

```text
D_z(x, r) = E[d_ord(Y_z, Y'_z) | x, r]
U_clarifiable(x, r) = D_original(x, r) - D_clarified(x, r)
U_residual(x, r) = D_clarified(x, r)
```

These are properties of a frozen model/prompt/runtime population. A positive
`U_clarifiable` does not establish that a human-authored policy has been
recovered.

## Role isolation

The following roles use fresh contexts and cannot exchange row-level output.

1. **Clarification proposer**: sees a rubric, its target adjacent boundary,
   and clarification-development examples only. It never sees evaluation
   answers, labels, failure-bank predictions, or judge outcomes.
2. **Fidelity auditor A/B**: independently compare the original and proposed
   clarification. They never score evaluation answers and never see outcomes.
3. **Original-condition judge panel**: sees only original rubrics.
4. **Clarified-condition judge panel**: sees only clarified rubrics and is not
   told that another condition exists.
5. **Analysis code**: unblinds conditions only after all panel files and hashes
   are frozen.

The target runtime for model roles is `gpt-5.6-sol` with reasoning effort
`max`. Independent calls are called *judge instances*, not independent people
or posterior samples.

## Clarification contract

A clarification may only operationalize text already present in the original
rubric. It must not:

- add or remove a criterion;
- change criterion weights or ordering;
- add a score cap not entailed by the original text;
- reveal any evaluation answer or desired score;
- specialize the rule to a single answer;
- replace the original construct with a new policy.

Each edit includes a clause-level provenance map from clarification text to
the original text. Two fidelity auditors independently answer the fixed
checklist. A clarification is eligible only when both auditors accept policy
preservation and neither flags answer leakage or a new scoring rule.

Because no rubric owner or trained human expert participates, passing this
checklist establishes only **model-audited textual policy preservation**.

## Train-only sampling frame

The private R3 train failure bank is the sampling frame. No dev/test row may be
read. The builder first creates an aggregate inventory, then deterministically
selects four rubric-boundary groups, targeting all adjacent boundaries:

```text
1 <-> 2
2 <-> 3
3 <-> 4
4 <-> 5
```

Each selected group must contain:

- two clarification-development answers;
- four evaluation answers not shown to the proposer, consisting of exactly
  three recurrent boundary-crossing failures and one clear anchor;
- at least five recurrent frozen-model boundary-crossing records so that the
  two development and three evaluation failures are disjoint;
- a clear anchor whose three original human scores agree and whose frozen R3
  outputs are correct in all three seeds.

One global clarification is written per rubric-boundary group. It cannot be
adapted to an evaluation answer. Selection uses only train identities and
frozen train failure evidence. Human aggregate labels and historical model
predictions are excluded from judge prompts.

The anchor is used only to detect a material scoring-policy shift. Its hidden
historical labels and predictions are never shown to a judge.

## Judge execution

For each of 16 evaluation answers:

- original condition: five fresh judge contexts;
- clarified condition: five different fresh judge contexts;
- identical question, answer, metric, serialization, output schema, and model
  settings across conditions;
- randomized but locked item order per judge instance;
- score first, followed by a concise evidence-grounded explanation;
- no other judge result is visible.

The formal pilot therefore contains `16 * 2 * 5 = 160` judge decisions. Calls
used for clarification or fidelity auditing are not counted as rating-panel
decisions.

## Outcomes

The primary feasibility effect is the paired change in mean pairwise ordinal
judge disagreement:

```text
tau_D = mean(D_original - D_clarified)
```

Secondary outcomes are:

- condition-specific score distributions and RPS-like pairwise dispersion;
- mean-score shift, reported separately from dispersion;
- exact agreement and adjacent-boundary crossing frequency;
- clear-anchor invariance;
- evidence-span consistency;
- replication value estimated by locked subsampling of judge-panel sizes;
- clarification-versus-replication value by rubric-boundary group.

No historical human label is the primary truth in this pilot. Natural scoring
metrics may be reported descriptively but cannot define clarification success.

## Feasibility gate

The direction proceeds to a larger same-data baseline study only if all of the
following hold:

1. at least 80% of proposed clarifications pass both fidelity auditors;
2. pooled mean ordinal disagreement falls by at least 25% relative;
3. the preregistered one-sided 90% randomization/cluster interval for
   `tau_D` is above zero;
4. clear-anchor score distributions do not show a material policy shift;
5. at least two rubric-boundary groups favor clarification over one additional
   judge call;
6. at least one group favors replication or direct scoring over clarification;
7. the effect is not confined to the `2 <-> 3` boundary.

This is a feasibility gate, not a confirmatory hypothesis test. Four rubric
clusters cannot support a population-level generalization claim.

## Immediate stop conditions

Stop this direction before training if any of the following occurs:

- fidelity auditors cannot reliably distinguish clarification from policy
  modification;
- fewer than 80% of edits pass the frozen checklist;
- clarification mainly changes the mean score rather than reproducibility;
- disagreement reduction is unstable or limited to one group;
- every group prefers the same acquisition action;
- repeated judge calls are effectively deterministic, making replication
  value unidentifiable;
- a generic same-data treatment or mixed-effects baseline explains the full
  useful effect;
- outputs reveal that the model inferred hidden target labels.

## Claim boundary

Passing the pilot would establish only that controlled specification edits can
produce heterogeneous effects on a frozen LLM-judge population. It would
authorize a same-data baseline comparison and theory development. It would not
establish a new method, human-rater validity, author-intent recovery, or
cross-model generality.

Any later CCF-A claim requires at minimum a second model family, an external
norm-governed ordinal task, same-data generic baselines, a source-aware
estimator, a non-equivalence argument, and a new confirmatory holdout.
