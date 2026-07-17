from __future__ import annotations

from pathlib import Path

import pytest

from thesis_exp.exp49_cphce.build_targets import aggregate_text_hash, load_split as load_exp49
from thesis_exp.exp51_hmsa.build_targets import load_split as load_exp51
from thesis_exp.exp51_hmsa.gate import gate_checks


def test_input_text_and_targets_match_exp49() -> None:
    for split in ("train", "dev"):
        old = load_exp49(split)
        new = load_exp51(split)
        assert aggregate_text_hash(old) == aggregate_text_hash(new)
        assert [row["soft_target_5"] for row in old] == [row["soft_target_5"] for row in new]
        assert [row["label_5"] for row in old] == [row["label_5"] for row in new]


def test_test_is_refused() -> None:
    with pytest.raises(PermissionError):
        load_exp51("test")


def test_config_locks_cosine_and_lambda_one() -> None:
    text = Path("thesis_exp/configs/exp51_hmsa/hmsa_lambda1.yaml").read_text(encoding="utf-8")
    assert "aux_weight: 1.0" in text
    assert "lr_scheduler_type: cosine" in text
    assert "inference: hard_head_raw_logit_argmax" in text


def test_seed42_gate_thresholds() -> None:
    baseline = {"n": 664, "Exact_rounded": 477 / 664, "MAE_human_mean": 0.3860441767, "Kendall_human_mean": 0.5770767969, "Bias_human_mean": 0.1139558233, "Recall_4_correct": 164, "Recall_5_correct": 289, "L2H_count": 11}
    passing = {"n": 664, "Exact_rounded": 475 / 664, "MAE_human_mean": 0.3810, "Kendall_human_mean": 0.5741, "Bias_human_mean": 0.1089, "Recall_4_correct": 161, "Recall_5_correct": 286, "L2H_count": 11}
    assert all(gate_checks(baseline, passing).values())
    failing = {**passing, "MAE_human_mean": 0.382}
    assert not gate_checks(baseline, failing)["mae_improvement_at_least_0p005"]
