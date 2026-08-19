# Exp54 thesis evidence closure

Status: `WRITE_READY_WITH_BOUNDED_RATIONALE_CLAIMS`

## Final research story

The completed experiments support a two-stage thesis contribution:

1. **RAR-SFT** separates the score decision and visible rationale into two
   supervised task blocks. Each block is length-normalized before the two
   losses are weighted, so the longer rationale does not dominate the short
   score field.
2. **Actual-error-driven Field-DPO** starts from the frozen RAR-SFT model,
   constructs rejected responses from errors that the model actually made,
   and applies the preference objective only to the intended output field.

The thesis should not present ordinary SFT or ordinary DPO as the contribution.
The supported contributions are block-balanced structured supervision,
field-local preference optimization, and actual-error-driven negative
construction.

## RQ1: Does semantic rationale alignment add value beyond rationale exposure?

The fixed 40-row, low-score-enriched dev blind audit compares:

- R3: answer-aligned, label-consistent human rationale supervision;
- R2: token-frequency-matched shuffled-rationale supervision.

The sample includes all 20 available Label-1/2 rows, covers all 12 metrics and
both languages, and carries all three training seeds. Arm identity, seed, gold
label, human rationale, forced-completion status and provenance are hidden.
Forced-completed outputs remain in the primary analysis.

| Evaluator | R3 win / tie / loss | Tie-adjusted preference | 95% record-cluster CI |
|---|---:|---:|---:|
| Codex Sol | 97 / 9 / 14 | 0.846 | [0.771, 0.912] |
| Codex Terra | 60 / 46 / 14 | 0.692 | [0.629, 0.754] |

Both agent runs prefer R3. This supports the bounded conclusion:

> On the fixed low-score-enriched dev audit sample, semantically aligned
> rationale supervision produces visible scoring justifications that
> same-family model judges prefer over a strict shuffled-rationale control.

The two judges are variants of the GPT-5.6 family. Exact overall agreement is
53.3% and Cohen's kappa is 0.157. Therefore this result is **exploratory
model-based preference**, not human correctness, expert correctness,
rationale accuracy or evidence of faithful hidden reasoning.

## RQ2: Why use score and rationale as separate loss blocks?

The frozen post-hoc test mechanism contrast compares:

- R3-TOKENAVG: one global mean over supervised score and rationale tokens;
- R3-BLOCK: separate score/rationale normalization followed by task-block
  weighting.

R3-BLOCK improves:

- MAE by 0.0305, 95% CI [0.0176, 0.0434],
  Holm-adjusted p = 0.0012;
- Exact Match by 0.0161, 95% CI [0.0056, 0.0266];
- Kendall's tau-b by 0.0455, 95% CI [0.0278, 0.0635];
- low-to-high error by 4.53 percentage points, with the primary-family
  adjusted interval still including zero.

This control supplies empirical, rather than purely intuitive, justification
for treating score and rationale as two task fields. It does not prove that no
other useful field decomposition exists.

## RQ3: Does field-local preference learning matter?

The matched comparison between ordinary full-sequence DPO and Field-DPO shows
that Field-DPO improves:

- MAE by 0.0183, 95% CI [0.0116, 0.0255],
  Holm-adjusted p = 0.0012;
- low-to-high error by 11.65 percentage points,
  95% CI [8.58, 14.96], Holm-adjusted p = 0.0012;
- Exact Match by 0.0048, 95% CI [0.0015, 0.0080];
- Kendall's tau-b by 0.0214, 95% CI [0.0143, 0.0288].

The supported claim is that localizing the preference objective to the target
field avoids unnecessary full-sequence preference pressure and improves
educational-scoring behavior under this frozen protocol.

## RQ4: Do actual model errors provide better negatives?

The record-matched comparison between actual-error negatives and synthetic
score negatives shows that actual-error-driven Field-DPO improves:

- MAE by 0.0179, 95% CI [0.0101, 0.0261],
  Holm-adjusted p = 0.0012;
- low-to-high error by 11.00 percentage points,
  95% CI [7.88, 14.48], Holm-adjusted p = 0.0012;
- Kendall's tau-b by 0.0207, 95% CI [0.0113, 0.0304].

The Exact Match difference is positive but uncertain: +0.0033,
95% CI [-0.0018, 0.0084].

This supports using errors actually produced by the frozen SFT model as
behaviorally relevant preference negatives rather than relying only on
synthetic score replacement.

## Rationale result after preference optimization

The separate P3-versus-P2 dev blind audit evaluates whether adding a rationale
preference block improves visible rationale quality beyond score-only
risk-conditioned preference training.

Across the two agent judges, score-visible overall tie-adjusted preference is:

- Codex Sol: 0.504, 95% CI [0.450, 0.558];
- Codex Terra: 0.533, 95% CI [0.463, 0.600].

Neither interval excludes 0.5. Thus:

- the RAR-SFT semantic rationale effect is positive but model-judged;
- the preference-training stage is supported for scoring and low-score risk;
- the DPO rationale block did not establish an additional visible-rationale
  improvement and should be reported as an informative near-zero ablation.

The final preference method in the thesis should therefore be
**actual-error-driven Field-DPO**, not a claim that rationale preference
optimization improves reasoning.

## Final result and remaining limitation

The final P1 Field-DPO model has three-seed test means:

| Metric | Result |
|---|---:|
| MAE | 0.3324 |
| Exact Match | 0.7286 |
| Kendall's tau-b | 0.6117 |
| Low-to-high error | 46.93% |
| Label-1 Recall | 58.33% |
| Label-2 Recall | 1.42% |

The method significantly mitigates low-score overestimation but does not solve
the low-score long tail. Label-2 recognition remains the clearest limitation.

## Allowed thesis claims

- Block-balanced score/rationale supervision outperforms global token
  averaging.
- R3 receives positive exploratory model-based rationale preference over a
  strict shuffled-rationale control.
- Field-local DPO outperforms full-sequence DPO.
- Actual-error-driven negatives outperform matched synthetic score negatives.
- The final method reduces, but does not eliminate, low-score overestimation.

## Forbidden thesis claims

- The generated rationales are proven human-correct or expert-correct.
- The visible rationale faithfully exposes the model's hidden reasoning.
- DPO improves rationale quality or internal reasoning.
- The low-score overestimation problem is solved.
- Label-2 performance is satisfactory.

## Frozen evidence

- RAR-SFT agent blind-audit aggregate SHA-256:
  `60e5a8280d19d5cba567989784038a5d49275d136eeaeb61b96d2562909543cd`
- P3-versus-P2 rationale-audit aggregate SHA-256:
  `4f5654f632d6b4ebaf1a54ba5688d909f521e59a0a2c21d9a8bec8046f9ee425`
- Mechanism-control public report SHA-256:
  `8ebf1bb51e7f2f9df5af2cc4914fc60be351f540cefac0c13b44e457cef0f30a`
- Mechanism-control result execution commit:
  `f52eec8023fbef33b7505ed0137544bd3c7da705`

No further test-guided training, checkpoint selection or hyperparameter tuning
is permitted. The next step is thesis writing and presentation, not another
model-training round.
