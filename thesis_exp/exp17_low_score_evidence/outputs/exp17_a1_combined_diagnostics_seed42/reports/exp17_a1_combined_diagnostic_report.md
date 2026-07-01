# Exp17-A1 Combined Diagnostic Report

This report combines the primary A1 scout and diagnostic-control runs. It is the formal A1 close-out
before Exp17-C0.

## Guardrails

- Test split is not read by these diagnostics.
- Dev D1 annotations are used only for evaluation.
- Human rationale text is not used as ranker input.
- A1 evidence head does not suppress or alter the ordinal score.

## Key Results

| config | MAE | QWK | low-to-high | label2 recall | h AUC | hidden-control delta |
|---|---:|---:|---:|---:|---:|---:|
| `A1_0_baseline` | 0.3966 | 0.5507 | 0.4912 | 0.0000 | 0.4615 | 0.0054 |
| `A1_1` | 0.3930 | 0.5310 | 0.4912 | 0.0000 | 0.3985 | -0.0118 |
| `A1_2` | 0.3966 | 0.5591 | 0.5263 | 0.0000 | 0.4738 | -0.0061 |
| `A1_3` | 0.3939 | 0.5743 | 0.5263 | 0.0000 | 0.5529 | 0.0152 |
| `A1_4` | 0.3930 | 0.5530 | 0.4912 | 0.0000 | 0.3894 | -0.0176 |
| `A1_5_all_low_aux_baseline` | 0.3866 | 0.5810 | 0.4912 | 0.0000 | 0.5732 | 0.0165 |
| `A1_6_random_positive_control` | 0.4011 | 0.5694 | 0.4912 | 0.0000 | 0.3734 | -0.0278 |
| `A1F_1_frozen_base_beta_0p10` | 0.3993 | 0.5532 | 0.5088 | 0.0000 | 0.4733 | 0.0127 |
| `A1F_2_frozen_probe_lr1em3_gradaccum1_epochs20` | 0.3993 | 0.5532 | 0.5088 | 0.0000 | 0.2874 | -0.0040 |
| `A1F_3_frozen_probe_lr3em4_gradaccum1_epochs20` | 0.3993 | 0.5532 | 0.5088 | 0.0000 | 0.3798 | -0.0220 |
| `A1F_4_frozen_probe_lr1em4_gradaccum1_epochs30` | 0.3993 | 0.5532 | 0.5088 | 0.0000 | 0.4402 | -0.0169 |
| `A1_5a_all_low_downsample76_same_neg_pool` | 0.3957 | 0.5664 | 0.5088 | 0.0000 | 0.4092 | -0.0458 |
| `A1_5b_all_low111_same_clean_high_controls` | 0.3921 | 0.5709 | 0.5088 | 0.0000 | 0.5123 | 0.0027 |
| `A1_1b_a0_weak_random_high_negatives` | 0.4020 | 0.5437 | 0.5439 | 0.0000 | 0.5593 | 0.0083 |

## Questions

1. Did high-learning-rate A1F probes solve undertraining? `False`. Best A1F is `A1F_1_frozen_base_beta_0p10` with AUC 0.4733.
2. Did frozen probes transfer to dev D1 hidden cases? No, because the best frozen AUC remains below the success gate.
3. Does all-low still dominate after fair controls? `False`. Best fair all-low control is `A1_5b_all_low111_same_clean_high_controls` with AUC 0.5123. The original all-low control was `A1_5_all_low_aux_baseline` with AUC 0.5732.
4. Is A1_1b only a weak random-high-negative signal? Yes. Its AUC is 0.5593, below 0.65.
5. Did A1 succeed? `False`.
6. Enter B1 suppression? `False`.
7. Move to C0 pairwise-low quality separation? `True`.

## Final Decision

- final_decision: `A1_failed_move_to_C0`
- best_a0_filtered_config: `A1_1b_a0_weak_random_high_negatives`
- best_a0_filtered_auc: 0.5593

A1 is closed unless a later manual review changes the success gate. The next ranker-side experiment
is C0 pairwise quality separation.
