# Exp27K Representative Teacher Protocol Validation

Exp27K completes teacher coverage for the 120-row representative Exp27J probability sample and
compares
single-teacher, simple fusion, and Exp27I-v1 rule fusion against the model-adjudicated Exp27J silver
reference.

## Coverage

- representative rows with Qwen and DeepSeek scores: 120/120
- risk-enriched rows retained as a separate stress view: 60/60
- no Exp27J silver field was used in teacher prompts.

## Representative Score Comparison

- original_human: MAE=0.738174, QWK=0.378735, within-one=0.819954, severe-error=0.180046
- qwen_blind: MAE=0.535689, QWK=0.592864, within-one=0.918711, severe-error=0.081289
- deepseek_blind: MAE=0.617238, QWK=0.579820, within-one=0.883514, severe-error=0.116486
- dual_teacher_rounded_mean: MAE=0.555151, QWK=0.625296, within-one=0.921688, severe-error=0.078312
- human_qwen_deepseek_median: MAE=0.539778, QWK=0.575809, within-one=0.891331, severe-error=0.108669
- exp27i_v1_rule_fusion: MAE=0.539778, QWK=0.575809, within-one=0.891331, severe-error=0.108669

## Risk-Signal Comparison

- best simple signal: qwen_human_gap, AUPRC=0.447968
- Exp27I-v1 tier proxy AUPRC: 0.349480
- Brier/ECE values are diagnostic proxies because the v1 scores were not statistically fitted probabilities.
- evidence tables check structural grounding and score-field consistency only; they do not certify semantic correctness.

## Decision

- formal Qwen3-Reranker training remains blocked.
- next: fit revised confidence tiers with question-key-aware cross-fitting, then obtain external expert review
  for high-impact ambiguous cases before constructing the formal 3326-row in-place training
variants.
- Exp27J remains a model-adjudicated silver reference, not human-expert gold.
- representative metrics use inverse-probability design weights; risk-enriched and all-unweighted views do not.
