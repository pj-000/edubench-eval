# Exp33A Reviewer Training and Statistical Guide

## Reference claim

Exp33A is a provider-agnostic, multi-stage blind-first, conflict-aware, direction-constrained
model-assisted data annotation and correction method. Its scientific contributions are
blind-first source comparison, conflict adjudication, direction-aware correction, and uncertainty
fallback—not use of any particular stronger model. It independently calibrates existing
Qwen/DeepSeek silver annotations and does not package a model as a human expert. Current outputs
remain an independent model-reviewed silver reference, and `expert_reference_complete` stays
false unless a future run imports genuine human review with explicit confirmation.

The code supports `reviewer_type=human` and `reviewer_type=model`; the current lock defaults to
`model`. Model reviews require the exact `reviewer_model_id`, a unique `reviewer_run_id`, and
provenance `independent_model_reviewer` or `independent_model_adjudicator`. The current experiment
plans GPT-5.6, but actual provider/model/version are implementation provenance and must not be
hidden. Human reviews require null `reviewer_model_id` and human provenance. Human reviewer
identity must remain private.

## Locked three-stage workflow

1. Blind review: launch Model Reviewer A and Model Reviewer B in separate fresh contexts. Give each
   the reviewer prompt, blind schema, and their private packet copy. Neither sees any human,
   Qwen, DeepSeek, student, variant, sampling-risk, or model-metric field.
2. Private source comparison: freeze both result files and hashes. Only then compare human_1/2/3,
   rounded human, Qwen, DeepSeek, and A/B. Audit 4-to-5 bias, low-to-high and high-to-low shifts,
   reason-score inconsistency, evidence validity, score caps, and hard-relabel drift.
3. Adjudication/correction: route every trigger to an independently launched Model Adjudicator.
   It sees frozen A/B structured outputs and controlled source provenance. It returns a soft
   model-reviewed silver posterior/score. If correction remains unsafe, fall back to the human
   empirical distribution; never force a hard change.

Adjudication is triggered by any score difference, disjoint ranges, failure-bucket difference,
student-input-sufficiency difference, low confidence, explicit adjudication request, or domain
escalation.

## Calibration before scale-up

The current packet is limited to 240 train calibration rows plus a 180-row clean-dev lockbox.
Expansion to all 2,654 train rows is allowed only after:

- paired A/B coverage = 100%;
- schema and evidence validity = 100%;
- blind leakage count = 0;
- within-one agreement >= 0.90;
- quadratic weighted kappa >= 0.60;
- Krippendorff ordinal alpha >= 0.60;
- all triggered cases are adjudicated or explicitly fall back/unresolved.

These gates use review quality, not student/dev model performance.

## Sampling estimands

Only the 120-row representative train view and its label-stratum design weights estimate source
reliability or prevalence. The 120-row risk-enriched view reports unweighted stress metrics only.
Clean-dev is frozen and may be used once after the correction method is fully locked. The paper
split is triple-key-disjoint, not question-key-disjoint; qkey overlap is descriptive and removes no
train row.

## Post-review statistics

Report A/B exact agreement, within-one agreement, quadratic weighted kappa, Krippendorff ordinal
alpha, score-range overlap, and adjudication rate overall and by view, language, metric family,
metric, subject, and original-label region.

Compare human_1/2/3, rounded human, Qwen, DeepSeek, teacher mean/median, Dawid-Skene, and MACE
against the frozen final reference using MAE, QWK, exact, within-one, signed bias, severe error,
low-to-high, high-to-low, label-1/2/5 recall, evidence-valid/invalid error, and
student-input-sufficient/insufficient error.

Use representative design weights for original-label conflict, teacher conflict, evidence-failure,
and input-insufficiency prevalence. Use no weights for the risk-enriched stress view.

## Domain and safety handling

Domain uncertainty always routes to adjudication. A residual unresolved domain case is not forced
into teacher reliability fitting. Never access the sealed test split, call an annotation API from
these scripts, run student inference/training, or use GPU resources.
