# Exp11 Checkpoint Selection Sensitivity

Status: `NO_COMPLETED_RUNS`

No completed Exp11 seed run with per-epoch metrics was found under the local ignored runs directory.
Run `./thesis_exp/scripts/run_exp11_checkpoint_selection_sensitivity.sh` to generate formal local
run artifacts, then re-run the collector.

Test metrics are post-hoc diagnostic only and are not used for selection. The core risk metric is
low-to-high.


## Review Checklist

- Verify `uses_test_for_selection` is false for every selection rule.
- Verify test diagnostic tables are not used to tune selection rules.
- Verify no checkpoint, raw prediction, `.npy`, or `.npz` artifact is written under tracked Exp11 outputs.
