# Exp54 thesis scientific contract v1

## 1. Status and purpose

Status: `DESIGN_FROZEN_IMPLEMENTATION_NOT_STARTED`

This contract freezes the smallest set of additional experiments needed to
separate the scientific contributions already bundled inside the completed
RAR-SFT and preference-optimization studies. It does not invalidate or redesign
the completed experiments.

The frozen result anchor is Git commit:

```text
a71470d  exp54: record one-time test results
```

The completed result anchor includes:

- S0/R1/R2/R3 RAR-SFT with seeds 42/43/44;
- P0/P1/P2/P3 preference results with seeds 42/43/44;
- the frozen 2,218-row test results;
- the conclusion that P1 is the main supported preference result;
- only directional evidence for the independent P2 ordinal-offset increment;
- no supported claim that P3 improves visible rationale quality.

No new H2 dataset is required by this contract. The existing frozen test may
be used once for each new checkpoint after all new methods, configurations,
checkpoints, contrasts, and metrics are frozen. These new comparisons are
reported as post-hoc mechanism analyses, not as untouched-test confirmatory
experiments.

## 2. Scientific questions

### RQ1: Why supervise score and rationale?

The minimal output decomposition is selected from the deployed task contract:

- `score` is the required educational decision;
- `rationale` is the human-provided explanation that connects the rubric,
  answer evidence, and score;
- both fields are visible outputs required from the grader;
- `evidence`, `failure`, and similar finer fields do not have complete,
  independent human ground truth in the current dataset and are not introduced
  as model-generated pseudo-labels.

This is a theoretically and empirically motivated minimal decomposition. It is
not claimed to be globally optimal among every possible output schema.

The existing matched contrasts answer two parts of this question:

- S0 versus R3: incremental value of adding aligned rationale supervision to
  score supervision;
- R2 versus R3: incremental value consistent with semantic alignment rather
  than merely adding rationale-shaped tokens.

### RQ2: Why use block-balanced SFT instead of ordinary token-average SFT?

The completed R3 method separately normalizes the short score field and the
long rationale field before adding them. The new R3-TOKENAVG control tests
whether this field-balanced credit allocation matters when the targets,
activity masks, training events, and total active-sample loss scale are held
fixed.

### RQ3: Does field-local preference optimization add value beyond standard
full-sequence DPO?

The completed P1 result bundles a DPO update with score-field-local credit
assignment. The new P1-FULLSEQ control keeps the exact actual-error preference
pairs and changes only the log-probability support from score-value tokens to
the standard full assistant completion.

### RQ4: Does mining the cold-start policy's actual score errors add value
beyond matched synthetic negatives?

The completed P1 result also bundles field-local DPO with actual-error-driven
negative construction. The new P1-SYN control keeps records, chosen outputs,
pair blocks, masks, optimization, and seeds fixed and changes only the rejected
score source and its provenance.

## 3. Frozen method claims

### 3.1 RAR-SFT

The thesis may describe RAR-SFT as:

> Rater-aligned multi-reference rationale SFT with missing-rationale masking
> and score-rationale block-balanced supervision.

Its proposed contribution is the complete supervision protocol:

1. retain score supervision for every train event;
2. apply rationale supervision only where an eligible human rationale exists;
3. select label-consistent human rationales for R3;
4. use the strict shuffled R2 control to isolate semantic alignment;
5. normalize score and rationale losses within field before equal block
   aggregation.

The thesis must not claim that `score+rationale` is the globally optimal field
schema. It may claim that it is the smallest schema supported by the task
output contract and available independent human annotations.

### 3.2 Preference optimization

The main preference contribution should be described as:

> Policy-Error-Conditioned Field-Local DPO (PECF-DPO): construct train-only
> preference pairs from the frozen cold-start policy's observed score errors
> and assign preference credit locally to the score-value field.

P1 is the principal implementation of this contribution. P2's ordinal-risk
offset is an extension with directional, not decisive, independent evidence.
P3 is a joint rationale-preference extension that is not FLOP-matched and did
not establish visible rationale-quality improvement.

## 4. Existing R3 block-balanced loss

For causal token negative log likelihood

\[
n_{it}=-\log p_\theta(y_{it}\mid x_i,y_{i,<t}),
\]

let \(m^s_{it}\) mark score-value tokens, \(m^r_{it}\) mark rationale-content
tokens, and \(q_i\in\{0,1\}\) mark whether rationale supervision is active.

\[
L_{i,s}
=
\frac{\sum_t m^s_{it}n_{it}}
{\max(1,\sum_t m^s_{it})},
\qquad
L_{i,r}
=
\frac{\sum_t m^r_{it}n_{it}}
{\max(1,\sum_t m^r_{it})}.
\]

The existing R3 per-event objective is:

\[
L_i^{\mathrm{BLOCK}}=L_{i,s}+q_iL_{i,r}.
\]

The batch objective is the arithmetic mean over original events. Prompt
tokens, JSON keys and punctuation, assistant suffixes, padding, and inactive
rationale positions have zero direct loss weight.

## 5. New control C1: R3-TOKENAVG

### 5.1 Estimand

Does separate score/rationale field normalization improve scoring relative to
ordinary unified target-token averaging, with the same R3 aligned targets and
matched active-sample loss scale?

### 5.2 Loss

For rationale-active events:

\[
L_i^{\mathrm{TOKENAVG}}
=
2
\frac{
\sum_t(m^s_{it}+m^r_{it})n_{it}
}{
\sum_t(m^s_{it}+m^r_{it})
}.
\]

The factor 2 matches the sum of the two unit-weight task blocks in the existing
R3 objective. It prevents the control from changing the active event's total
coefficient merely because two blocks have been replaced by one.

For rationale-inactive events:

\[
L_i^{\mathrm{TOKENAVG}}=L_{i,s}.
\]

The batch objective remains the arithmetic mean over original events.

### 5.3 Only allowed difference

Only the loss reduction changes from `BLOCK` to `TOKENAVG`.

The following remain byte-identical or configuration-identical to R3:

- seed-specific materialized R3 manifests and row order;
- selected references, rationale text, active vector, score targets, masks,
  serialization, and token IDs;
- base model and tokenizer/chat template;
- LoRA rank/alpha/dropout and target modules;
- logical epochs, optimizer steps, micro-batch, accumulation, padding, and
  cutoff;
- optimizer, learning rate, schedule, warmup, precision, and randomness;
- checkpoint rule and inference protocol.

### 5.4 Required runs

```text
R3-TOKENAVG seed 42
R3-TOKENAVG seed 43
R3-TOKENAVG seed 44
```

The matched comparator is the existing seed-matched R3-BLOCK checkpoint.

## 6. Existing P1 Field-DPO loss

For active score-value token set \(T_s\), define mean field log probability:

\[
\ell_\pi^s(y\mid x)
=
\frac{1}{|T_s|}
\sum_{t\in T_s}
\log\pi(y_t\mid x,y_{<t}).
\]

For chosen \(y^+\), rejected \(y^-\), trainable policy \(\pi_\theta\), and
frozen reference \(\pi_{\mathrm{ref}}\):

\[
\Delta_\theta^s
=
\ell_{\pi_\theta}^s(y^+\mid x)
-
\ell_{\pi_\theta}^s(y^-\mid x),
\]

\[
\Delta_{\mathrm{ref}}^s
=
\ell_{\pi_{\mathrm{ref}}}^s(y^+\mid x)
-
\ell_{\pi_{\mathrm{ref}}}^s(y^-\mid x).
\]

P1 uses:

\[
L_i^{\mathrm{FIELD}}
=
-\log\sigma
\left(
\beta[
\Delta_\theta^s-\Delta_{\mathrm{ref}}^s
]
\right),
\qquad \beta=0.1.
\]

The existing dataset objective gives the adjacent-score, severe-L2H, and
high-to-low guard blocks equal total weight.

## 7. New control C2: P1-FULLSEQ-DPO

### 7.1 Estimand

Does score-field-local credit assignment improve scoring and low-score risk
relative to standard full-completion DPO on the exact same actual-error pairs?

### 7.2 Loss

Let \(T_{\mathrm{completion}}\) contain all non-padding assistant-completion
tokens selected by the standard DPO completion mask, including the serialized
score, rationale, JSON syntax, and assistant suffix where present in the
frozen serialization:

\[
\ell_\pi^{\mathrm{seq}}(y\mid x)
=
\sum_{t\in T_{\mathrm{completion}}}
\log\pi(y_t\mid x,y_{<t}).
\]

\[
\Delta_\theta^{\mathrm{seq}}
=
\ell_{\pi_\theta}^{\mathrm{seq}}(y^+\mid x)
-
\ell_{\pi_\theta}^{\mathrm{seq}}(y^-\mid x),
\]

\[
\Delta_{\mathrm{ref}}^{\mathrm{seq}}
=
\ell_{\pi_{\mathrm{ref}}}^{\mathrm{seq}}(y^+\mid x)
-
\ell_{\pi_{\mathrm{ref}}}^{\mathrm{seq}}(y^-\mid x).
\]

The control uses standard sequence DPO:

\[
L_i^{\mathrm{FULLSEQ}}
=
-\log\sigma
\left(
\beta[
\Delta_\theta^{\mathrm{seq}}
-
\Delta_{\mathrm{ref}}^{\mathrm{seq}}
]
\right),
\qquad \beta=0.1.
\]

No ODPO offset is applied.

### 7.3 Only allowed difference

Only the DPO log-probability support/reduction changes:

```text
P1-FIELD:   mean over score-value tokens only
P1-FULLSEQ: sum over the standard full assistant completion
```

The following remain identical to P1 at learning rate `5e-6`:

- all 838 actual-error pairs and their order before seeded shuffling;
- chosen and rejected serialized sequences;
- record, block, label, metric, language, and provenance vectors;
- equal three-block dataset weighting;
- seed-matched frozen R3 epoch-3 policy and reference adapters;
- one preference epoch, 27 optimizer steps, micro-batch one, accumulation 32;
- AdamW, cosine schedule, warmup, precision, LoRA, cutoff, and fixed padding;
- checkpoint and inference protocol.

### 7.4 Required runs

```text
P1-FULLSEQ seed 42
P1-FULLSEQ seed 43
P1-FULLSEQ seed 44
```

The matched comparator is the existing seed-matched P1-FIELD checkpoint at
learning rate `5e-6`.

## 8. New control C3: P1-SYN at the final learning rate

### 8.1 Estimand

Does using the frozen R3 policy's actual observed train-only score errors
improve scoring and low-score risk relative to same-record, same-block
synthetic rejected scores?

### 8.2 Only allowed difference

P1-ACTUAL and P1-SYN must share:

- the same 838 record IDs and pair-type vector in the same canonical order;
- chosen sequence, chosen score, rationale anchor, and chosen field mask;
- score block, label, metric, language, and stratum vectors;
- rejected sequence schema and rejected score field mask;
- equal three-block weights;
- seed-matched R3 epoch-3 policy/reference adapters;
- Field-DPO loss and `beta=0.1`;
- learning rate `5e-6`, one preference epoch, 27 optimizer steps;
- every other optimizer, LoRA, batch, schedule, padding, and inference setting.

They differ only in:

- rejected score value;
- rejected score source (`actual_policy_error` versus matched synthetic);
- provenance fields required to verify that source.

### 8.3 Required runs and correction to the old diagnostic

The old P1-SYN seed-42 model belongs to the original `5e-7` candidate and is
not a matched control for the final `5e-6` P1 result. Therefore all three
P1-SYN runs must be trained at `5e-6`:

```text
P1-SYN seed 42
P1-SYN seed 43
P1-SYN seed 44
```

The matched comparator is the existing seed-matched P1-ACTUAL/P1-FIELD
checkpoint at `5e-6`.

## 9. Frozen nine-run matrix

| Control | Seed 42 | Seed 43 | Seed 44 | Existing comparator |
|---|---:|---:|---:|---|
| R3-TOKENAVG | new | new | new | R3-BLOCK |
| P1-FULLSEQ | new | new | new | P1-FIELD |
| P1-SYN `5e-6` | new | new | new | P1-ACTUAL `5e-6` |

Total new full training runs: **9**.

No P2 offset rescaling, new beta search, GRPO experiment, P3 compute redesign,
or actual-rationale-negative training is part of this contract.

## 10. Checkpoint and dev rules

- R3-TOKENAVG saves logical epochs 1/2/3 exactly as R3 did.
- Its checkpoint rule is fixed before execution to the existing RAR-SFT rule:
  highest dev Exact, then lower dev MAE, then earlier epoch.
- Rationale metrics and forced-close statistics may not select checkpoints.
- P1-FULLSEQ and P1-SYN use the same final-checkpoint rule as the completed
  one-preference-epoch P1 runs; no new learning-rate, beta, or step search is
  allowed.
- Dev may diagnose execution and apply only these already frozen checkpoint
  rules. It may not change a control's definition.
- Every run is reported. An unfavorable seed may not be dropped or retrained
  under the same run identity.

## 11. Existing-test reuse boundary

The old P0/P1/P2/P3 and SFT test outputs remain frozen and must not be
regenerated.

After all nine new checkpoints are frozen:

1. freeze adapter hashes, config hashes, prompt, schema, tokenizer, decoder,
   parser, metrics, and all three contrasts;
2. run each new checkpoint on the same 2,218-row test exactly once;
3. do not tune, select, or retrain from those test results;
4. compare against the already frozen, seed-matched comparator predictions;
5. report all outcomes, including null or adverse results.

Because the three controls were designed after aggregate results from this
test had been observed, their test comparisons are explicitly:

```text
post-hoc mechanism analyses on a frozen benchmark test
```

They are not described as:

```text
untouched-test preregistered confirmatory evidence
```

This distinction does not make the controls invalid; it limits the strength of
the statistical claim.

## 12. Frozen contrasts and endpoints

### C1: field-balanced SFT

```text
R3-BLOCK minus R3-TOKENAVG
```

### C2: field-local DPO

```text
P1-FIELD minus P1-FULLSEQ
```

### C3: actual-error mining

```text
P1-ACTUAL minus P1-SYN
```

For each contrast, report seed 42/43/44 separately and the equal-seed mean.

Primary scoring endpoints:

- MAE, lower is better;
- Exact Match, higher is better.

Primary low-score-risk endpoints:

- Label-1/2 low-to-high count and rate, lower is better;
- Label-2 count and recall, higher recall is better.

Guardrails:

- Kendall's tau;
- signed bias;
- QWK;
- Label-5 recall;
- high-to-low errors;
- strict parse and forced-close rates.

Use paired question/record-cluster bootstrap with 10,000 replicates. Apply Holm
correction across the predeclared primary mechanism tests. Effect sizes and
confidence intervals remain mandatory even when corrected significance is not
reached.

## 13. Interpretation rules

### If C1 favors R3-BLOCK

The thesis may attribute incremental value to score-rationale block balancing,
in addition to the existing evidence about rationale supervision and semantic
alignment.

If C1 is null or adverse, the thesis retains the aligned-rationale supervision
result but must not claim that the block-balanced loss independently improves
generalization.

### If C2 favors P1-FIELD

The thesis may attribute incremental value to field-local preference credit
assignment over standard sequence DPO on the same actual-error pairs.

If C2 is null or adverse, P1 remains evidence that the complete preference
pipeline works, but field localization is not established as the causal
mechanism.

### If C3 favors P1-ACTUAL

The thesis may attribute incremental value to policy-observed error mining
over matched synthetic rejected scores.

If C3 is null or adverse, P1 remains a working preference method, but the
actual-error source is not established as independently superior.

No outcome from these score experiments supports a claim of improved visible
rationale quality or internal reasoning.

## 14. Next implementation gate

Before any GPU execution, the implementation stage must produce:

1. an R3-TOKENAVG loss implementation and unit tests;
2. a standard P1-FULLSEQ DPO path and field-support tamper tests;
3. P1-SYN `5e-6` manifests/configs for all three seeds with exact
   record/block/chosen-vector matching against P1;
4. deterministic nine-run configs and output identities;
5. a CPU audit showing that each control changes only its declared variable.

Passing that gate authorizes a small train-only smoke test. It does not by
itself authorize the nine full GPU runs.
