# Exp17-D0 Low-Score Evidence Diagnosis

This diagnostic does not train a model. It analyzes Exp16A qmr boundary-cache dev predictions to
test whether label-2 high predictions are caused by quality-score separation failure.

## Key Findings

- Rows analyzed: `1107`.
- Label-2 samples: `38`; label-2 predicted as 4/5: `27` (0.7105)；label-2 recall remains `0.0000`.
- Mean `s` for label2: `1.6080`; mean `s` for labels 4/5: `2.6459`; delta high-minus-label2: `1.0379`.
- AUC using `s` to separate labels 4/5 from label2: `0.6849`.
- AUC using negative `s` to separate low labels <=2 from the rest: `0.7458`.
- Matched control-case mean `s` gap: `-0.0365`.
- RQ2 support level: **Strong support** (label2_l2h_rate >= 0.5, label2 mean g_i3 > 0, matched control-case mean s gap <= 0.25, label2 L2H case count >= 20).

## Score Distribution by Gold Label

| gold | n | mean s | median s | mean tau3 | mean margin tau3 | mean g_i3 | pred>=4 | recall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 19 | -1.1340 | -1.5547 | 0.4946 | -1.6286 | -0.8375 | 2/19 (0.1053) | 0.4737 |
| 2 | 38 | 1.6080 | 1.6602 | 0.0248 | 1.5832 | 2.2269 | 27/38 (0.7105) | 0.0000 |
| 3 | 105 | 2.0532 | 2.0156 | 0.3806 | 1.6725 | 2.5742 | 86/105 (0.8190) | 0.1810 |
| 4 | 389 | 2.1364 | 1.9375 | -0.1265 | 2.2629 | 3.5775 | 373/389 (0.9589) | 0.6838 |
| 5 | 556 | 3.0024 | 2.7891 | -0.9656 | 3.9679 | 6.1937 | 555/556 (0.9982) | 0.8165 |

## Label-2 Error Concentration by Metric

| metric | label2 n | label2 -> 4/5 | R2->H | label2 mean s | label2 mean g_i3 |
|---|---:|---:|---:|---:|---:|
| Instruction Following & Task Completion | 7 | 7 | 1.0000 | 2.1094 | 4.3984 |
| Clarity, Simplicity & Inspiration | 4 | 4 | 1.0000 | 2.0098 | 3.4063 |
| Error Identification & Correction Precision | 4 | 4 | 1.0000 | 1.8438 | 4.4282 |
| Scenario Element Integration | 4 | 4 | 1.0000 | 1.8887 | 1.8885 |
| Reasoning Process Rigor | 3 | 3 | 1.0000 | 1.3750 | 2.5991 |
| Basic Factual Accuracy | 2 | 2 | 1.0000 | 2.1543 | 4.7292 |
| Content Relevance & Scope Control | 2 | 2 | 1.0000 | 2.2969 | 5.7396 |
| Motivation, Guidance & Positive Feedback | 4 | 1 | 0.2500 | 1.5156 | -0.7267 |
| Higher-Order Thinking & Skill Development | 7 | 0 | 0.0000 | 0.5652 | -1.5160 |
| Personalization, Adaptation & Learning Support | 1 | 0 | 0.0000 | 0.3242 | -0.2737 |

## Manual Audit Template

`manual_audit_template.csv` contains `27` label2 high-prediction cases for human defect annotation.
Use the provided defect type options to mark whether each error is linked to missing key points,
rubric violation, insufficient evidence, surface fluency, or ambiguity.

## Interpretation

The diagnosis strongly supports the RQ2 hypothesis: Exp16A's label-2 failure is primarily a
quality-score evidence problem. The next step should be Exp17-A/B/C rather than another boundary or
threshold experiment.
If the manual audit finds clear rubric-linked defects in most sampled cases, Exp17 can define `D` as
weakly supervised low-score defect evidence and inject it into `s = s_base - lambda * D`.
