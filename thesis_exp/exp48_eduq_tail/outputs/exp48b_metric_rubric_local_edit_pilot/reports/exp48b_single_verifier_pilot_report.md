# Exp48B single-verifier protocol pilot

- Status: **EXP48B_SINGLE_VERIFIER_PILOT_SIGNAL**
- Formal qualification complete: **False**
- Missing preregistered requirement: second independent cross-model-family verifier.
- Generator and verifier used separate Codex contexts, but shared-platform/model-family bias remains possible.
- Language distribution: `{"en": 12}`; cross-language reliability was not tested.
- Intended exact: 36/36 (1.0000)
- QWK: 1.0000
- Fully confirmed / ordered families: 12/12 / 12/12
- Score-2 confirmed / score2-to-4: 12/12 / 0
- Accepted metrics: 12/12
- Exact evidence substring validity: 1.0000
- Style-only macro-F1: 0.2596

## Pilot gates

- generation_12_valid: **PASS**
- outside_span_identity_12: **PASS**
- single_verifier_complete: **PASS**
- evidence_substring_validity_100pct: **PASS**
- intended_exact_ge_33: **PASS**
- qwk_ge_0p85: **PASS**
- fully_confirmed_families_ge_9: **PASS**
- score2_confirmed_ge_10: **PASS**
- score2_to_4_zero: **PASS**
- ordered_families_ge_10: **PASS**
- accepted_metrics_ge_9: **PASS**
- style_macro_f1_le_0p45: **PASS**
- question_novelty_12: **PASS**
- no_eval_access: **PASS**

- `recommend_scale_generation=false` regardless of these single-verifier results.
- No training, no GPU, no dev/test access.
