from __future__ import annotations

from thesis_exp.exp51_hmsa.formal_gate import checks_from_rows


def test_formal_gate_uses_locked_multiseed_thresholds() -> None:
    base = {
        "b0_selected_epoch": 8,
        "exp51_selected_epoch": 5,
        "b0_exact_correct": 477,
        "exp51_exact_correct": 476,
        "mae_improvement": 0.006,
        "delta_exact": -1 / 664,
        "delta_kendall": 0.0,
        "delta_label4_recall": -0.01,
        "delta_label5_recall": -0.005,
        "delta_l2h_count": -1,
        "abs_bias_improvement": 0.006,
    }
    rows = [{"seed": seed, **base} for seed in (42, 43, 44)]
    _, _, checks = checks_from_rows(rows, ci_high=-0.001)
    assert all(checks.values())
    rows[2] = {**rows[2], "mae_improvement": -0.01}
    _, _, checks = checks_from_rows(rows, ci_high=0.001)
    assert not checks["paired_mae_ci_upper_below_zero"]
