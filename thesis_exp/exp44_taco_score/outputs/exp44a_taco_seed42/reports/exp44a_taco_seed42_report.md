# Exp44A TACO-Score Seed42 Report

Final status: **TACO_SEED42_STOP**

## Gate checks

- core_label2: FAIL
- low_to_high_guard: PASS
- overall_gain: FAIL
- protection: FAIL
- beats_C1_tail: FAIL
- protects_vs_C1: FAIL
- beats_C3: PASS
- bootstrap_no_significant_harm: PASS
- bootstrap_gate: PASS

## Metrics

| Variant | MAE | QWK | Exact | Kendall | Bias | Bin agreement | L2H | H2L | Label1 | Label2 | Label5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0_E4_baseline | 0.387340 | 0.464876 | 0.673323 | 0.504579 | 0.095705 | 0.879050 | 0.763158 | 0.001289 | 0.250000 | 0.000000 | 0.763215 |
| C1_balanced_plain_contrastive | 0.376036 | 0.480049 | 0.682366 | 0.520414 | 0.100980 | 0.877543 | 0.736842 | 0.000859 | 0.250000 | 0.000000 | 0.776973 |
| C2_TACO | 0.388470 | 0.449428 | 0.675207 | 0.505400 | 0.108139 | 0.870761 | 0.763158 | 0.001719 | 0.208333 | 0.000000 | 0.775525 |
| C3_shuffled_margin_control | 0.397890 | 0.440159 | 0.664657 | 0.492989 | 0.101733 | 0.869631 | 0.789474 | 0.000430 | 0.166667 | 0.000000 | 0.762491 |

C2 label2 correct: 0/52; Wilson 95% CI=[0.000000, 0.068792]; class-2 predictions=2.

## Representation diagnostics

- C0_E4_baseline: nearest-centroid balanced accuracy=0.643341; label2 nearest class2=33/52
- C1_balanced_plain_contrastive: nearest-centroid balanced accuracy=0.589856; label2 nearest class2=24/52
- C2_TACO: nearest-centroid balanced accuracy=0.554262; label2 nearest class2=21/52
- C3_shuffled_margin_control: nearest-centroid balanced accuracy=0.557757; label2 nearest class2=26/52

## Boundaries

No API or teacher labels were used. Dev and test access counts are both zero. Raw predictions, triplets, checkpoints, embeddings, and logs remain private and ignored.
