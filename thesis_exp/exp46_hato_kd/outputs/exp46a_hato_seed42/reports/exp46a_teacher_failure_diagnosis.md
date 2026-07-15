# Exp46A Teacher Failure Diagnosis

Decision: **TEACHER_CAPACITY_NO_GO**

## What failed

- The 4B teacher identified **0 label-2 examples** correctly and label-2 recall remained 0. It predicted label 2 for 7 examples, but every such prediction was a false positive.
- Low-to-high increased from 58/76 (0.7632) to 60/76 (0.7895).
- MAE worsened from 0.3873 to 0.4043; QWK fell from 0.4649 to 0.4420.
- Exact Match fell from 0.6733 to 0.6590; Kendall tau fell from 0.5046 to 0.4805.
- Absolute signed bias increased from 0.0957 to 0.1081; the mean prediction also increased from 4.4552 to 4.4676.

## Statistical reading

- Question-key bootstrap intervals cross zero for the overall metric deltas, so the experiment does not establish statistically significant overall harm.
- It also provides no positive overall gain: every preregistered point-estimate improvement condition failed.
- Label-2 recall is exactly unchanged at zero, with bootstrap delta 0 and interval [0, 0]. This is the decisive mechanism failure for distillation.
- Human cross-entropy improved slightly, but Brier score and ranked probability score worsened. Better likelihood on the observed distributions did not translate into safer hard-score decisions.

## Causal scope

This result rejects the locked Exp46A premise that a 4B LoRA teacher trained with the same human-distribution and ordinal objective supplies transferable label-2 structure. It does **not** prove that Qwen3-Reranker-4B is inherently incapable, nor does it test full fine-tuning, additional tail supervision, or a different data distribution.

## Protocol consequence

- K1/K2/K3 student training was correctly skipped. Distilling this teacher would transfer no verified label-2 signal.
- Do not rerun Exp46A unchanged and do not tune the Gate after seeing these results.
- A new positive experiment would require a fresh preregistration and a changed source of tail evidence or optimization, not merely a larger teacher.

## Integrity

- Five of five Teacher folds completed at the locked final epoch.
- Question-key overlap: 0.
- Dev access count: 0.
- Test access count: 0.
- Failed Gate checks: bias_protection, exact_protection, label2_correct, label2_precision, label2_recall, low_to_high, overall_gain.
