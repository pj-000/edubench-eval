# QD-PR2 Anchored Pairwise Ordinal Fine-tuning Plan

Run ID: `QD-PR2_AnchoredPairwiseOrdinal_human_only`
Formal training status: `not_started`.
Training executed by scaffold stage: `no`.
API called: `no`.
Synthetic generated: `no`.

## Motivation

QD-PR1 is a negative result: it learned dev pairwise separation but damaged pointwise ordinal
calibration and raised monotonic violation. QD-PR2 therefore uses QD-B1 as an anchor and performs
small-weight pairwise fine-tuning instead of from-scratch training.

## Loss

`L_total = L_point + lambda_pair * L_pair + lambda_anchor * L_anchor + lambda_mono * L_mono`

- `L_point`: QD-B1-style weighted ordinal BCE.
- `L_pair`: `w_ij * softplus(m_ij - (r(win)-r(lose)))`, with `r(x)=1+sum_t sigmoid(z_t)`.
- `L_anchor`: BCE with QD-B1 reference probabilities from the train pair examples.
- `L_mono`: `mean_t max(0, p_{t+1}-p_t)`.

Defaults: `lambda_pair=0.05`, `lambda_anchor=0.5`, `lambda_mono=0.1`, `epochs=3`,
`learning_rate=1e-5`, `effective_batch_size=128`, `max_length=2048`.

## Pair Dataset

- Train pairs: `10000`
- Dev diagnostic pairs: `3000`
- Max low-record reuse: `95`
- Mean low-record reuse: `54.090090090090094`

Pairs are selected from question_seed42 human-only rows. The priority order is `same_question >
same_metric_language`; `same_metric` and `any_valid` are fallback only and are reported when used.

| split | pair_type | pairs | same_question | same_metric_language | fallback |
| --- | --- | ---: | ---: | ---: | ---: |
| train | low_high | 4000 | 0.5208 | 0.5100 | 0.0000 |
| train | low_mid | 2000 | 0.1790 | 0.7940 | 0.0525 |
| train | adjacent | 3000 | 1.0000 | 0.1237 | 0.0000 |
| train | random_ordinal | 1000 | 1.0000 | 0.0710 | 0.0000 |
| dev | low_high | 1200 | 0.4750 | 0.5592 | 0.0000 |
| dev | low_mid | 600 | 0.4633 | 0.5917 | 0.0000 |
| dev | adjacent | 900 | 1.0000 | 0.1222 | 0.0000 |
| dev | random_ordinal | 300 | 1.0000 | 0.0800 | 0.0000 |

## Training Gate

QD-PR2 must load the QD-B1 checkpoint. If the checkpoint is missing, scripts must report
`BLOCKED_MISSING_QDB1_CHECKPOINT` and stop before smoke/formal training.

## QD-PR2 Controlled Changes

- Initialize from QD-B1 checkpoint.
- Keep `lambda_pair` small: first run `0.05`; optional sweep `{0.05, 0.1}`.
- Keep anchor loss on train examples only.
- Add monotonic regularization to protect cumulative probability order.
- Use high-comparability pairs only unless fallback shortage is reported.
- Fine-tune 2-3 epochs, never from scratch.
