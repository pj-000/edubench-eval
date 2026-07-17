from __future__ import annotations

from thesis_exp.exp49_cphce.build_targets import aggregate_text_hash, load_split as load_exp49
from thesis_exp.exp50_cahs.build_targets import load_split as load_exp50
from thesis_exp.exp50_cahs.gate import gate_checks
from thesis_exp.exp50_cahs.train import checkpoint_wins


def test_input_text_is_identical_to_exp49() -> None:
    for split in ("train", "dev"):
        assert aggregate_text_hash(load_exp49(split)) == aggregate_text_hash(load_exp50(split))


def test_checkpoint_ties_keep_earlier_epoch() -> None:
    assert checkpoint_wins(0.72, 0.71)
    assert not checkpoint_wins(0.71, 0.71)


def test_discrete_seed42_gate_thresholds() -> None:
    baseline = {"n": 664, "Exact_rounded": 477 / 664, "MAE_human_mean": 0.386, "Kendall_human_mean": 0.577, "Bias_human_mean": 0.114, "Recall_4_correct": 164, "Recall_5_correct": 289, "L2H_count": 11}
    passing = {"n": 664, "Exact_rounded": 475 / 664, "MAE_human_mean": 0.380, "Kendall_human_mean": 0.575, "Bias_human_mean": 0.108, "Recall_4_correct": 161, "Recall_5_correct": 286, "L2H_count": 11}
    assert all(gate_checks(baseline, passing).values())
    failing = {**passing, "Exact_rounded": 474 / 664}
    assert not gate_checks(baseline, failing)["exact_loses_at_most_two_rows"]
