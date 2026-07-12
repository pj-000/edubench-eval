# Exp35A EduDART-Cal Failure Diagnosis

## Frozen decision

Exp35A qualification failed. No Exp35B train supervision may be generated,
and no student training, dev evaluation, or test access is allowed from this
version. The qualification constants are not changed after observing these
results.

All references are independent model-reviewed silver, not human expert gold.

## Reference behavior

- Fresh general qualification (originally label3/4/5): silver point labels are
  1:1, 2:6, 3:14, 4:14, 5:85.
- Reassessed original train-low rows: silver point labels are 1:16, 2:14,
  3:4, 4:12, 5:30.
- Thus 42/76 original low rows are judged 4/5 by the independent model-review
  workflow. This is a model-reference disagreement, not proof that either side
  is objectively correct.

## Method comparison

On the 120-row fresh general view:

| Method | MAE | QWK | low-to-high | high-to-low |
|---|---:|---:|---:|---:|
| Rounded human | 0.6167 | 0.3931 | 0.2857 | 0.0000 |
| Qwen hard | 0.4667 | 0.5970 | 0.1429 | 0.0505 |
| Qwen calibrated | 0.5921 | 0.2093 | 1.0000 | 0.0303 |
| EduDART pre-projection diagnostic | 0.6925 | 0.1834 | 0.5714 | 0.0000 |
| Frozen EduDART-Cal | 0.6734 | 0.1242 | 0.5714 | 0.0202 |

On the repeated-sample low-tail stress view, Qwen hard remains strongest
(MAE 0.6974, QWK 0.6751). Frozen EduDART-Cal has MAE 1.9866, QWK -0.0276,
and label2 recall 0.

## Attribution

1. The Qwen confusion relationship estimated on Exp33 representative silver
   does not transfer to the fresh qualification silver. The calibrated Qwen
   posterior is materially worse than Qwen hard.
2. Human-disagreement preservation and model-reviewed silver disagree sharply
   on the original low tail. Mixing the two cannot satisfy both references
   without an external source of truth.
3. The pre-projection diagnostic is already poor; global marginal projection
   is not the sole or primary cause, although it further changes low-tail
   behavior.
4. The model-reviewed qualification is useful for feasibility and consistency
   auditing, but it cannot establish that model relabels are objectively higher
   quality than the original human labels.

## Consequence

Do not tune EduDART-Cal v1 on these 196 qualification rows. A successor method
would require either an external reference source or a different scientific
claim focused on model-consistent supervision rather than label correctness.
