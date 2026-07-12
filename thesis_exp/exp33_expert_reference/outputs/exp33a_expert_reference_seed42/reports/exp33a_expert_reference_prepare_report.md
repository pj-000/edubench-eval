# Exp33A Independent Model-Reviewed Silver Reference Preparation

## Outcome

Exp33A prepared 420 blind review rows for two independent reviewers. The current lock is
`reviewer_type=model` and defaults to model review. No reviewer output has been
created or implied. `model_silver_reference_complete=false` and
`expert_reference_complete=false`.

## Paper protocol boundary

- Paper split unit: `(question, answer, metric)` triple key.
- Train/dev triple-key overlap: 0.
- Train/dev question-key overlap: 184 of 184 dev qkeys; this is expected and allowed.
- Train rows on shared train/dev qkeys: 2562 / 2654.
- Future train rows removed for qkey overlap: 0; all 2654 train rows remain.
- The protocol is not replaced by question-key GroupCV.
- Locked rows: train=2654, dev=664, sealed test=2218. The test count is inherited from the locked Exp32 statement; the test file was not opened.

Clean-dev maximizes question-key diversity only inside its 180 selected rows. Its qkeys may
overlap train, and they create no future-training exclusion.

## Resolved teacher annotation inputs

- primary: `thesis_exp/exp17_low_score_evidence/outputs/exp28c_teacher_protocol_evaluation_seed42/private/qwen/p0_holistic_zero_shot/all_train.jsonl`; rows=2654; valid=2654; SHA-256=`42d4ca48d4ef7ef9bf4bddc91c5652c337bcfba108359527e21354a3247150eb`
- secondary: `thesis_exp/exp17_low_score_evidence/outputs/exp28c_teacher_protocol_evaluation_seed42/private/deepseek/p0_holistic_zero_shot/secondary_route.jsonl`; rows=1552; valid=1552; SHA-256=`07ae301026a6330de682e02fa6f0195eaf71a6f3275f589551e334b48e6c462a`

Resolution uses Exp28 machine-readable provider/model/subset summaries and cross-checks the
Exp28E locked valid-row counts. Raw API logs are not read.

## Sampling

| view | rows | labels 1..5 | unique qkeys | permitted use |
|---|---:|---|---:|---|
| representative_train | 120 | 1=12, 2=18, 3=25, 4=30, 5=35 | 120 | design-weighted population reliability/prevalence |
| risk_enriched_train | 120 | 1=12, 2=34, 3=24, 4=25, 5=25 | 87 | unweighted stress/routing analysis only |
| clean_dev | 180 | 1=6, 2=14, 3=40, 4=60, 5=60 | 175 | one-time frozen-method evaluation only |

Representative inclusion probabilities are `n_h/N_h` within original-label strata and design
weights are `N_h/n_h`. Language, metric family, and subject are auxiliary balance variables.
Risk enrichment never estimates population prevalence. Clean-dev used no teacher conflict,
student prediction, Exp29-31 result, or dev model metric.

## Blind packet and leakage

- Packet rows per independent reviewer: 420.
- Packet manifest SHA-256: `0ff65510c9172989a1f03b3c0f9774258bf2e6fb73f589478baa56639e510d17`.
- Exact sample overlap between all three views: 0.
- Human label/reason, Qwen/DeepSeek score/reason, campaign flag, student prediction, variant,
  metric-result, and sampling-risk leakage: 0 by allowlist plus source-projection audit.
- Reviewer A and Reviewer B receive identical blind content in separate local packet files.
    - Reviewer model IDs are intentionally unset until launch; the exact implementation model (currently planned GPT-5.6), provider, and a distinct run/context ID for each role must be recorded as provenance.

## Completion and escalation

The provider-agnostic method is multi-stage blind-first, conflict-aware, and direction-constrained
model-assisted annotation/correction. Its innovations are blind-first source comparison, conflict
adjudication, direction-aware correction, and uncertainty fallback—not the choice of a particular
provider or stronger model.

The locked follow-up has three stages. First, Model Reviewer A/B independently see only the
blind packet and emit score/range/evidence. Second, after both outputs are frozen, a private source
comparison audits human_1/2/3, Qwen, DeepSeek, and A/B for 4-to-5 bias, low-to-high shifts,
reason-score inconsistency, and hard-relabel drift. Third, triggered cases go to an independently
launched Model Adjudicator with frozen A/B structured results and controlled source provenance.
The adjudicator emits a model-reviewed silver posterior/score. If uncertainty remains, the system
falls back to the human empirical score distribution instead of forcing a hard relabel.

Expansion from the 240-row train calibration sample to all 2,654 train rows is forbidden until the
pre-registered completion, leakage, schema/evidence, within-one, QWK, ordinal-alpha, and
trigger-processing gates pass. No model performance metric selects this gate.

Current state: reviews not started; teacher reliability not ready; no new teacher/student training
is recommended; test access is not recommended.

## Resource audit

No API was called, no GPU was used, no model was trained, no student inference ran, and the
sealed test split was not read. Private sample IDs, blind packets, source reasons, teacher reasons,
and future filled reviews are gitignored.
