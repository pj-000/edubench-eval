# Exp19-R5G Transition Diagnostic Report

This report compares dev predictions across R2c, R4b, R5F2 real-only, R5G A3, and R5G B1.

## Low-Score Fixes

- low samples analyzed: 57
- fixed_by_r5f2_real: 19
- fixed_by_a3: 11
- fixed_by_b1: 10
- fixed_by_a3 among D1 hidden: 5

## High-Score Harm

- high samples analyzed: 945
- high_to_low_r5f2_real: 80
- high_to_low_a3: 20

## Metric/Language Clusters

- high_to_low_r5f2_real / Scenario Element Integration / en: 23
- high_to_low_r5f2_real / Reasoning Process Rigor / en: 14
- high_to_low_r5f2_real / Error Identification & Correction Precision / en: 7
- high_to_low_r5f2_real / Basic Factual Accuracy / en: 6
- high_to_low_r5f2_real / Domain Knowledge Accuracy / en: 6
- high_to_low_r5f2_real / Reasoning Process Rigor / zh: 5
- high_to_low_r5f2_real / Higher-Order Thinking & Skill Development / en: 4
- high_to_low_r5f2_real / Motivation, Guidance & Positive Feedback / en: 3
- high_to_low_r5f2_real / Instruction Following & Task Completion / en: 2
- high_to_low_r5f2_real / Content Relevance & Scope Control / en: 2
- high_to_low_r5f2_real / Instruction Following & Task Completion / zh: 2
- high_to_low_r5f2_real / Basic Factual Accuracy / zh: 2
- high_to_low_r5f2_real / Personalization, Adaptation & Learning Support / en: 1
- high_to_low_r5f2_real / Clarity, Simplicity & Inspiration / en: 1
- high_to_low_r5f2_real / Higher-Order Thinking & Skill Development / zh: 1
- high_to_low_r5f2_real / Motivation, Guidance & Positive Feedback / zh: 1
- high_to_low_a3 / Scenario Element Integration / en: 5
- high_to_low_a3 / Basic Factual Accuracy / en: 4
- high_to_low_a3 / Reasoning Process Rigor / en: 4
- high_to_low_a3 / Error Identification & Correction Precision / en: 3

## Interpretation

- R5F2 real-only shows how many R2c low-to-high cases can be fixed by strong low-risk DPO.
- R5G A3 shows which fixes survive a less over-conservative risk-calibrated setting.
- High-to-low rows show where strong low-risk DPO harms true high-score samples.

## Missing Prediction Runs

- none

## Guardrails

- This script reads dev predictions only.
- It does not read test.
- It does not train or write raw predictions.
