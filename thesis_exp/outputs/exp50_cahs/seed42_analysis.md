# Exp50-CAHS-0.5 seed42 analysis

## Decision

Status: **SEED42_NO_GO**.

The locked CAHS target was `0.5 * onehot(label_5) + 0.5 * human_empirical_distribution`. The selected checkpoint was epoch 6 (update step 126), chosen by strictly highest dev Exact with ties retaining the earlier epoch. It matched B0 Exact and improved several secondary metrics, but its argmax MAE improvement was only 0.002008, below the preregistered 0.005 requirement. Seeds 43/44 and test therefore remain locked.

## Contract and runtime

- Fixed split: train 2654, dev 664; test not loaded.
- Input: question + answer + metric, identical to Exp49 by aggregate text hash.
- Train hash: `67a03a285ba0bde1458318d0d8ffc86409156c574bebffe6760b1fd200b5d961`.
- Dev hash: `888db0fef20586e339ee2b2a152a2ccd3ecfd53589ca10912793ebad00e8f679`.
- Model: Qwen3-Reranker-0.6B sequence classifier, five logits.
- Optimizer contract: learning rate 2e-5, weight decay 0.01, cosine with warmup, micro-batch 4, accumulation 32, effective batch 128, BF16.
- Initial classifier hash: `d7c922a1956118af437c52189ffc993465277a72698cef38290e47404247f9e2`.
- Runtime: 2601.4 seconds; peak allocated GPU memory 6,069,753,856 bytes.
- `test_access_count`: 0.

## Selected-checkpoint comparison

| Metric | B0 hard CE (epoch 8) | CAHS-0.5 (epoch 6) | Change |
|---|---:|---:|---:|
| Exact | 477/664 (0.718373) | 477/664 (0.718373) | 0 rows |
| MAE human mean | 0.386044 | 0.384036 | -0.002008 |
| Kendall | 0.577077 | 0.589939 | +0.012862 |
| Bias | +0.113956 | +0.101908 | abs bias -0.012048 |
| QWK | 0.623193 | 0.634637 | +0.011444 |
| Expected-score MAE | 0.355564 | 0.334638 | -0.020926 |
| ECE | 0.147713 | 0.076319 | -0.071393 |
| NLL | 0.905518 | 0.723173 | -0.182346 |
| Brier | 0.425590 | 0.400163 | -0.025427 |
| label 4 correct | 164/237 | 164/237 | 0 rows |
| label 5 correct | 289/345 | 291/345 | +2 rows |
| L2H | 11/20 | 8/20 | -3 rows |

## Gate

Seven checks passed: Exact, Kendall, label 4, label 5, L2H non-increase, Bias-or-L2H improvement, and finiteness. The only failed check was `B0 MAE - CAHS MAE >= 0.005`: observed improvement was 0.002008.

The paired row bootstrap mean delta `CAHS MAE - B0 MAE` was -0.002008 with a 95% interval of [-0.023594, 0.019076]. Exact-status transitions were symmetric: B0-only correct 39, CAHS-only correct 39, McNemar two-sided exact p=1.0. These diagnostics do not include training-seed uncertainty.

## Epoch trajectory

| Epoch | Exact correct | Exact | MAE | Kendall | Selected at that point |
|---:|---:|---:|---:|---:|---|
| 1 | 410 | 0.617470 | 0.516064 | 0.343978 | yes |
| 2 | 449 | 0.676205 | 0.427209 | 0.526263 | yes |
| 3 | 472 | 0.710843 | 0.406627 | 0.572751 | yes |
| 4 | 475 | 0.715361 | 0.393574 | 0.561478 | yes |
| 5 | 471 | 0.709337 | 0.389056 | 0.581497 | no |
| 6 | 477 | 0.718373 | 0.384036 | 0.589939 | yes, final |
| 7 | 467 | 0.703313 | 0.396084 | 0.577922 | no |
| 8 | 465 | 0.700301 | 0.383032 | 0.593571 | no |
| 9 | 472 | 0.710843 | 0.386546 | 0.585681 | no |
| 10 | 467 | 0.703313 | 0.386546 | 0.582749 | no |

Epoch 8 had a slightly better MAE than epoch 6 but 12 fewer Exact-correct rows, so it was correctly rejected by the locked checkpoint rule.

## Mechanistic interpretation

CAHS reduced the full-soft damage seen in Exp49: label 4 returned from 152/237 under full soft to the B0 value of 164/237, Exact returned from 474 to 477, and label 5 increased by two. Thus the original hypothesis that the 2/3:1/3 full distribution perturbed the middle/high boundary too strongly was partly supported.

However, shrinking that perturbation did not produce the required primary-metric effect. On the 373 disagreement rows CAHS improved MAE from 0.512958 to 0.501340 and Exact from 234 to 237; on the 291 unanimous rows it worsened MAE from 0.223368 to 0.233677 and Exact from 243 to 240. Unanimous rows have the same one-hot target under B0 and CAHS, so their regression cannot be direct label smoothing. It is consistent with shared-representation/boundary interference plus seed-level training variation.

The much larger gain in expected-score MAE and probability metrics than in argmax MAE is the clearest result: the human distribution signal improves distributional fit and calibration, but a single shared classification head does not reliably translate those gains into the paper's hard decision metrics.

## Protocol consequence and next method family

Do not search alpha 0.25/0.75, change the Gate, run seeds 43/44, or access test. The static single-head hard/soft target continuum has now been tested at both endpoints and its fixed midpoint without sufficient evidence for the required Pareto improvement.

If a new experiment is authorized, it should be separately preregistered as a hard-main/soft-auxiliary architecture: retain an ordinary hard-label classification head as the sole inference and checkpoint-selection head, and place the three-human distribution in a distinct auxiliary head/loss. This tests structural decoupling rather than another smoothing strength. Its loss weight, checkpoint rule, seed42 Gate, and no-test boundary must be frozen before GPU use.

## Reproducibility limitation

The two deterministic smoke runs matched the initial classifier hash, batch IDs, RNG traces, scheduler, and all losses before the first optimizer update. After the CUDA/BF16 update they were not bitwise identical (maximum observed loss drift 0.0405953). Consequently, row-bootstrap intervals must not be presented as training-randomness intervals, and the single-seed 0.002008 MAE gain is not strong enough to override the locked threshold.
