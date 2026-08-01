# Exp54 SORC-DPO one-time test results

| Arm | MAE↓ | L2H↓ | Exact↑ | Kendall↑ | Bias | Recall-2↑ | Recall-5↑ | H2L↓ | QWK↑ | Forced close |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P0_R3_SFT | 0.3569 | 63.11% | 0.7218 | 0.5854 | +0.1673 | 0.00% | 85.18% | 0.05% | 0.5336 | 15.16% |
| P1_FIELD_DPO | 0.3324 | 46.93% | 0.7286 | 0.6117 | +0.1274 | 1.42% | 84.82% | 0.14% | 0.6290 | 15.90% |
| P2_SORC_SCORE | 0.3293 | 44.98% | 0.7299 | 0.6135 | +0.1204 | 1.42% | 84.65% | 0.16% | 0.6394 | 15.55% |
| P3_JOINT_SORC | 0.3230 | 41.75% | 0.7311 | 0.6202 | +0.1189 | 1.42% | 84.79% | 0.12% | 0.6586 | 14.58% |

| Contrast | MAE benefit [95% CI] | Holm p | L2H benefit [95% CI] | Holm p | Classification |
|---|---:|---:|---:|---:|---|
| H1_FIELD_DPO | +0.0245 [+0.0170, +0.0326] | 0.0012 | +0.1618 [+0.1255, +0.2013] | 0.0012 | STRONG_SUPPORT |
| H2_ORDINAL_OFFSET | +0.0032 [-0.0003, +0.0068] | 0.0806 | +0.0194 [+0.0031, +0.0382] | 0.0740 | DIRECTIONAL_SUPPORT |
| H3_RATIONALE_BLOCK | +0.0063 [+0.0018, +0.0111] | 0.0168 | +0.0324 [+0.0125, +0.0543] | 0.0128 | STRONG_SUPPORT |

Positive contrast values mean benefit under the preregistered endpoint-specific sign convention.

P3−P2 remains a bundled, non-FLOP-matched effect and cannot establish improved rationale quality.

This was the one-time test. Result-driven reruns are forbidden.
