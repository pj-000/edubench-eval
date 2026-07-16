# Exp48A EduQ-TAIL qualification report

- Final status: **EDUQ_TAIL_QUALIFICATION_NO_GO**
- Generated / valid / accepted families: 60 / 60 / 7
- Accepted score 2/3/5 rows: 7 / 7 / 7
- Cross-family verification: False
- Verifier A: `{"model_families": ["deepseek-v4-pro"], "session_ids": ["exp48a_deepseek_verifier_a_pilot_20260716"], "rows": 60}`
- Verifier B: `{"model_families": ["deepseek-v4-pro"], "session_ids": ["exp48a_deepseek_verifier_b_pilot_20260716"], "rows": 60}`
- Criterion agreement: 0.9375
- Exact / within-one / QWK: 0.5778 / 0.7361 / 0.5994
- Score2-to-high failures: 7
- Style-only macro-F1: 0.4274
- Question novelty pass: True

## Gate results

- generated_60: **PASS**
- valid_at_least_54: **PASS**
- accepted_at_least_45: **FAIL**
- accepted_each_score_at_least_45: **FAIL**
- criterion_agreement_at_least_0p80: **PASS**
- exact_score_agreement_at_least_0p85: **FAIL**
- within_one_at_least_0p98: **FAIL**
- qwk_at_least_0p75: **FAIL**
- score2_to_high_zero: **FAIL**
- question_novelty_pass: **PASS**
- style_macro_f1_at_most_0p45: **PASS**
- no_eval_access: **PASS**

- recommend_scale_generation: `false`
- stop_synthetic_low_tail_route: `true`
- No training; no GPU; no dev/test access; no heavy/private artifacts are public outputs.
