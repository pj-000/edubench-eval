# RAR-SFT preregistration draft

> **Superseded on 2026-07-23.** This document preserves the historical five-field RAR-SFT design.
> All future Exp54 implementation and interpretation follow
> [`PREREGISTRATION_V2.md`](PREREGISTRATION_V2.md). Do not use this file to authorize training.

## Research claim

The study tests whether rubric-aligned, evidence-grounded, field-gated rationale supervision can
reduce low-quality-answer overestimation and produce checkable score-consistent justifications
without materially degrading overall scoring or high-score recall.

The claim is not that JSON, rubrics, Qwen3-4B, or rationale generation are novel. The intended
contributions are supervision qualification, field-selective length-normalized loss, and a matched
semantic control.

## Frozen boundaries

- Dataset family: `paper_like_triple_seed42`.
- RAR-0 reads train only. Dev is reserved for checkpoint selection and gates. Test remains sealed
  until the frozen final campaign.
- Base model ID: `Qwen/Qwen3-4B-Instruct-2507`.
- Exact model revision/snapshot hash: **unresolved until recorded from the training server**.
- Visible rationale is an evaluation justification, not hidden chain-of-thought.
- Score is parsed only from the first JSON field and is never revised from rationale text.

## Required matched methods

| ID | Input | Output supervision | Purpose |
|---|---|---|---|
| H0 | question + answer + metric | score | historical Exp53 baseline only |
| S0 | question + answer + metric + full rubric | score | matched scientific baseline |
| R1 | same as S0 | score + unfiltered structured rationale | test harm from no gating |
| R2 | same as S0 | score + verified auxiliary fields shuffled within score x metric x language | format/length/regularization control |
| R3 | same as S0 | score + answer-matched verified auxiliary fields | RAR-SFT |

R2 must preserve the auxiliary schema, field coverage pattern, language, metric, score, and as far
as possible length distribution. It must only break answer-to-reason/evidence semantics.

## RAR-0 stages

1. **RAR-0A deterministic alignment**: exact normalized question + answer + canonical metric +
   generator-model match; no fuzzy fallback. Audit unmatched and ambiguous rows separately.
2. **RAR-0B criterion registry**: freeze stable atomic criterion IDs derived only from the fixed
   human rubrics. The included registry is a draft and is not gold until human-reviewed.
3. **RAR-0C structured conversion**: convert each aligned human reason to rubric checks, quote or
   missing evidence, and a concise rationale. Converter output is candidate data, never automatic
   gold.
4. **RAR-0D deterministic field checks**: legal criterion ID, exact quote span, valid missing claim
   form, score/verdict compatibility, schema validity, and provenance.
5. **RAR-0E independent verification and human audit**: a different verifier plus two blind human
   reviewers on 120 train-only samples, with third-reviewer adjudication.

## Field gates and loss

The score field always has gate 1. Auxiliary fields receive gradient only when their individual
gate is 1. For sample i and field f:

```text
L_i,f = mean token NLL over field f
A_i   = {f: q_i,f = 1}
L_i,aux = mean(L_i,f for f in A_i)
L_i,RAR = L_i,score + 1.0 * L_i,aux
```

If no auxiliary field qualifies, the example is exactly score-only. Field boundaries must be
derived from serialized JSON token spans and tested against the actual tokenizer; punctuation
owned by a masked field must not leak auxiliary loss.

## Seed-42 gate

- `Delta MAE <= +0.005`
- `Delta Exact >= -0.005`
- `Delta Kendall >= -0.005`
- `Delta Recall_5 >= -0.010`
- at least one low-tail improvement:
  - `L2H <= 0.329`, or
  - `Recall_2 >= 0.497`
- schema validity at least 99%; grounded evidence and score-reason consistency must exceed R1;
  human review must favor real verified fields over R2 on at least two reason-quality dimensions.

Checkpoint selection remains score-first: maximum dev Exact, then lower MAE, then earlier epoch.
Reason metrics do not select checkpoints.

## Unlock rule

Only R3 proceeds to seeds 42/43/44 after passing the seed-42 gate. Model snapshot, data hashes,
criterion registry, structured data, gates, loss, decoding, checkpoint rule, and evaluator are then
frozen before the one-shot test campaign.
