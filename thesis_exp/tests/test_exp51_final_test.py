from __future__ import annotations

import pytest

from thesis_exp.exp51_hmsa.final_test import checkpoint_path, result_key


def test_final_test_registry_is_exactly_two_arms_by_three_seeds() -> None:
    keys = {result_key(arm, seed) for arm in ("b0", "exp51") for seed in (42, 43, 44)}
    assert keys == {
        "b0/seed_42",
        "b0/seed_43",
        "b0/seed_44",
        "exp51/seed_42",
        "exp51/seed_43",
        "exp51/seed_44",
    }


def test_checkpoint_paths_are_state_dicts() -> None:
    assert checkpoint_path("b0", 42).as_posix().endswith("exp49_cphce/b0_hard_ce/seed_42/best/state_dict.pt")
    assert checkpoint_path("exp51", 44).as_posix().endswith("exp51_hmsa/hmsa_lambda1/seed_44/best/state_dict.pt")
    with pytest.raises(ValueError):
        checkpoint_path("unknown", 42)


def test_partial_test_is_forbidden_before_any_data_read(monkeypatch: pytest.MonkeyPatch) -> None:
    from thesis_exp.exp51_hmsa import final_test

    monkeypatch.setattr(final_test, "require_in_progress", lambda: {"test_anchor": {"rows": 2218}})
    with pytest.raises(PermissionError, match="Partial final-test"):
        final_test.evaluate_checkpoint("b0", 42, split="test", max_eval_samples=1)
