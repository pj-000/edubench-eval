# Exp19-S0 SFT/DPO Dataset QC Report

Exp19-S0 prepares reason-aware SFT and DPO datasets for Qwen3-4B. No model is trained, and the test
split is not read.

## Counts

- train samples: 3326
- dev samples read for guardrail/evaluation prep only: 1107
- low-label samples (y<=2): 111
- high-label samples (y>=4): 2918
- low example fraction: 0.0334
- matched A0 hidden-failure candidates: 111
- matched A0 clean high controls: 2656
- usable low failure targets: 84
- high-score protection count: 2918

## Dataset Counts

- edubench_r1_score_only_train: 3326 (sft)
- edubench_r2_reason_score_train: 3326 (sft)
- edubench_r3_reason_rationale_train: 3326 (sft)
- edubench_r4_shuffled_reason_control_train: 3326 (sft)
- edubench_r5_risk_balanced_dpo_train: 6466 (dpo)

## DPO Risk Counts

- high_to_low_protection: original=2918, expanded=5836
- low_to_high: original=111, expanded=333
- mid_score_calibration: original=297, expanded=297

## Distribution Summary

- label distribution: {'1': 58, '2': 53, '3': 297, '4': 1163, '5': 1755}
- major failures distribution: {'no_major_failure': 2918, 'insufficient_evidence': 320, 'unclear': 27, 'surface_fluent_but_hidden_defect': 25, 'missing_key_point': 25, 'task_constraint_violation': 4, 'answer_key_or_reference_mismatch': 4, 'format_violation': 3}
- score cap distribution: {'None': 2918, '3': 297, '1': 58, '2': 53}
- rubric_satisfied distribution: {'True': 2918, 'False': 408}
- max question_group/question_key rate: 0.0150
- language distribution: {'en': 1694, 'zh': 1632}
- top metrics: {'Instruction Following & Task Completion': 595, 'Content Relevance & Scope Control': 405, 'Basic Factual Accuracy': 405, 'Scenario Element Integration': 322, 'Reasoning Process Rigor': 258, 'Clarity, Simplicity & Inspiration': 257, 'Higher-Order Thinking & Skill Development': 256, 'Domain Knowledge Accuracy': 197, 'Personalization, Adaptation & Learning Support': 190, 'Motivation, Guidance & Positive Feedback': 187, 'Error Identification & Correction Precision': 134, 'Role & Tone Consistency': 120}

## Leakage Check

- recovered human rationales checked against user prompts: 80
- exact human-rationale leakage count in user prompts: 0
- user prompts are built only from question, answer, metric, rubric, and metadata.
- human rationale-derived fields appear only in assistant targets.

## Recommendation

- safe for SFT: `True`
- safe for DPO: `True`

Full raw JSON datasets are intentionally written under the gitignored output/data directory.
Redacted samples replace raw answers and rationales with previews/hashes.
