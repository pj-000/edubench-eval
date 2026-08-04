from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np
import pytest

from thesis_exp.exp61_soft_sts15_external_confirmation.geometry import select_component
from thesis_exp.exp61_soft_sts15_external_confirmation.mapping import (
    build_maximum_mismatch_mapping,
    mapping_target_lookup,
    theoretical_maximum_changes,
)
from thesis_exp.exp61_soft_sts15_external_confirmation.method import (
    empirical_target,
    hard_soft_ce_identity,
    human_mean,
    quantized_mean_label,
    residual_target,
    target_fifths,
)
from thesis_exp.exp61_soft_sts15_external_confirmation.metrics import (
    component_cluster_bootstrap,
    confirmation_gates,
    evaluate_hard_head,
)
from thesis_exp.exp61_soft_sts15_external_confirmation.analysis import (
    validate_prediction_rows,
)
from thesis_exp.exp61_soft_sts15_external_confirmation.contract import (
    ROOT,
    repository_local_import_closure,
    sha256_file,
    verify_source_lock,
)
from thesis_exp.exp61_soft_sts15_external_confirmation.evaluate_test_once import (
    _write_json_exclusive,
    validate_checkpoint_grid,
)
from thesis_exp.exp61_soft_sts15_external_confirmation.train import (
    verify_seed_preflight_binding,
)


def test_six_class_five_rating_target_contract() -> None:
    scores = [0, 2, 3, 5, 5]
    assert target_fifths(scores) == (1, 0, 1, 1, 0, 2)
    assert empirical_target(scores) == (0.2, 0.0, 0.2, 0.2, 0.0, 0.4)
    assert human_mean(scores) == 3.0
    assert quantized_mean_label(scores) == 3
    assert np.allclose(residual_target(3, empirical_target(scores)), (-0.2, 0, -0.2, 0.8, 0, -0.4))


def test_half_up_quantized_mean_not_bankers_rounding() -> None:
    assert quantized_mean_label([0, 0, 0, 0, 5]) == 1
    assert quantized_mean_label([0, 0, 0, 5, 5]) == 2


def test_soft_ce_identity() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[0.2, -0.3, 1.1, 0.7, -1.2, 0.4], [1, 2, 3, 4, 5, 6.0]])
    labels = torch.tensor([3, 4])
    targets = torch.tensor([[0.2, 0, 0.2, 0.2, 0, 0.4], [0, 0, 0, 0.2, 0.6, 0.2]])
    assert hard_soft_ce_identity(logits, labels, targets)["absolute_identity_error"] < 1e-6


def _row(index: int, label: int, fifths: tuple[int, ...]) -> dict[str, object]:
    return {
        "record_id": f"r{index}",
        "label": label,
        "target_fifths": list(fifths),
        "component_sha256": f"c{index}",
        "split": "train",
    }


def test_maximum_mismatch_mapping_is_bijective_and_train_only() -> None:
    a, b, c = (5, 0, 0, 0, 0, 0), (4, 1, 0, 0, 0, 0), (3, 2, 0, 0, 0, 0)
    rows = [_row(i, 0, state) for i, state in enumerate([a, a, a, b, c])]
    mapping, audit = build_maximum_mismatch_mapping(rows)
    assert audit["effective_target_changes"] == theoretical_maximum_changes(Counter([a, a, a, b, c])) == 4
    assert len({row["donor_record_id"] for row in mapping}) == len(rows)
    assert all(len(value) == 6 and np.isclose(sum(value), 1) for value in mapping_target_lookup(mapping).values())
    rows[0]["split"] = "dev"
    with pytest.raises(PermissionError):
        build_maximum_mismatch_mapping(rows)


def test_geometry_surface_has_only_three_registered_arms() -> None:
    common = {"backbone.x": np.array([1.0])}
    aligned = {"backbone.x": np.array([2.0])}
    shuffled = {"backbone.x": np.array([3.0])}
    assert float(select_component("quantized_mean_only", common, aligned, shuffled)["backbone.x"][0]) == 0
    assert select_component("aligned_orthogonal_only", common, aligned, shuffled) is aligned
    assert select_component("matched_shuffled_orthogonal_only", common, aligned, shuffled) is shuffled
    with pytest.raises(ValueError):
        select_component("posthoc_new_arm", common, aligned, shuffled)


def test_metrics_use_hard_head_expectation_against_five_rating_mean() -> None:
    probs = np.eye(6)[[0, 2, 5]].astype(float)
    metrics = evaluate_hard_head(probs, np.array([0.0, 2.4, 4.8]), np.array([0, 2, 5]))
    assert np.isclose(metrics["mae"], 0.2)
    assert np.isclose(metrics["nmae"], 0.04)
    assert metrics["exact_match"] == 1.0


def test_component_bootstrap_is_deterministic_and_gate_code_is_fail_closed() -> None:
    human = np.array([0.0, 1.0, 4.0, 5.0])
    aligned = {seed: human + 0.01 for seed in (61, 62, 63)}
    shuffled = {seed: human + 0.2 for seed in (61, 62, 63)}
    first = component_cluster_bootstrap(aligned, shuffled, human, ["a", "a", "b", "c"], n_resamples=100, rng_seed=7)
    second = component_cluster_bootstrap(aligned, shuffled, human, ["a", "a", "b", "c"], n_resamples=100, rng_seed=7)
    assert first == second
    assert first["ci95_upper"] < 0
    aligned_metrics = {seed: {"nmae": 0.10, "spearman": 0.80} for seed in (61, 62, 63)}
    shuffled_metrics = {seed: {"nmae": 0.12, "spearman": 0.80} for seed in (61, 62, 63)}
    main_metrics = {seed: {"nmae": 0.11, "spearman": 0.79} for seed in (61, 62, 63)}
    assert confirmation_gates(aligned_metrics, shuffled_metrics, main_metrics, first)["decision"] == "PASS"


def test_source_lock_uses_real_repository_local_transitive_closure() -> None:
    closure = set(repository_local_import_closure())
    assert "thesis_exp/exp61_soft_sts15_external_confirmation/audit_dataset.py" in closure
    assert "thesis_exp/exp60_geometry_matched_shuffle/contract.py" in closure
    assert "thesis_exp/exp60_geometry_matched_shuffle/__init__.py" in closure
    assert "thesis_exp/exp61_soft_sts15_external_confirmation/evaluate_test_once.py" in closure
    assert "thesis_exp/exp61_soft_sts15_external_confirmation/sealed_test_data.py" in closure


def test_source_lock_verifier_rejects_an_omitted_mandatory_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import thesis_exp.exp61_soft_sts15_external_confirmation.contract as contract

    mandatory = repository_local_import_closure()
    lock = {
        "status": "EXP61_SOURCE_LOCK_CANDIDATE_AWAITING_FINAL_PREFLIGHT_REVIEW",
        "files": {relative: sha256_file(ROOT / relative) for relative in mandatory},
        "mandatory_file_count": len(mandatory),
        "protocol_sha256": sha256_file(contract.FROZEN_PROTOCOL),
        "test_access_count": 0,
        "formal_training_authorized": False,
    }
    path = tmp_path / "source_lock.json"
    path.write_text(json.dumps(lock))
    monkeypatch.setattr(contract, "SOURCE_LOCK", path)
    verify_source_lock(require_formal_authorization=False)
    lock["files"].pop(
        "thesis_exp/exp61_soft_sts15_external_confirmation/audit_dataset.py"
    )
    path.write_text(json.dumps(lock))
    with pytest.raises(RuntimeError, match="dependency closure mismatch"):
        verify_source_lock(require_formal_authorization=False)


def test_seed_preflight_binding_fails_closed() -> None:
    lock = {"model": {"manifest_sha256": "model"}}
    preflight = {
        "status": "EXP61_REAL_MODEL_NO_UPDATE_PREFLIGHT_PASS",
        "seed": 61,
        "initial_parameter_sha256": "initial",
        "model_manifest_sha256": "model",
        "checks": {
            "no_parameter_update": True,
            "optimizer_step_count_zero": True,
            "test_access_count_zero": True,
        },
    }
    verify_seed_preflight_binding(
        seed=61,
        initial_parameter_sha256="initial",
        preflight=preflight,
        source_lock=lock,
    )
    preflight["initial_parameter_sha256"] = "different"
    with pytest.raises(RuntimeError):
        verify_seed_preflight_binding(
            seed=61,
            initial_parameter_sha256="initial",
            preflight=preflight,
            source_lock=lock,
        )


def _synthetic_test_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    expected: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    for index in range(1578):
        frozen = {
            "record_id": f"softsts15:{index}",
            "row_id": index,
            "split": "test",
            "component_sha256": f"component-{index // 2}",
            "human_mean": 2.0,
            "hard_label": 2,
        }
        expected.append(frozen)
        predictions.append(
            {
                **frozen,
                "variant": "quantized_mean_only",
                "seed": 61,
                "checkpoint_state_dict_sha256": "state",
                "hard_head_probabilities": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                "hard_head_expectation": 2.0,
                "hard_head_argmax": 2,
            }
        )
    return expected, predictions


def test_strict_test_prediction_metadata_rejects_any_manifest_drift() -> None:
    expected, predictions = _synthetic_test_rows()
    validate_prediction_rows(
        predictions,
        expected,
        variant="quantized_mean_only",
        seed=61,
        checkpoint_state_sha256="state",
    )
    predictions[100]["component_sha256"] = "wrong"
    with pytest.raises(RuntimeError):
        validate_prediction_rows(
            predictions,
            expected,
            variant="quantized_mean_only",
            seed=61,
            checkpoint_state_sha256="state",
        )


def test_test_access_marker_is_exclusive(tmp_path: Path) -> None:
    marker = tmp_path / "test_access.json"
    _write_json_exclusive(marker, {"test_access_count": 1})
    with pytest.raises(FileExistsError):
        _write_json_exclusive(marker, {"test_access_count": 2})


def test_checkpoint_grid_requires_nine_epoch10_artifacts_and_paired_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import thesis_exp.exp61_soft_sts15_external_confirmation.evaluate_test_once as evaluator

    monkeypatch.setattr(evaluator, "ARTIFACT_ROOT", tmp_path)
    for variant in evaluator.VARIANTS:
        for seed in evaluator.SEEDS:
            root = tmp_path / variant / f"seed_{seed}" / "epoch10"
            root.mkdir(parents=True)
            state = root / "dual_head_state_dict.pt"
            state.write_bytes(f"{variant}-{seed}".encode())
            metadata = {
                "status": "EXP61_FIXED_EPOCH10_CHECKPOINT",
                "variant": variant,
                "seed": seed,
                "epoch": 10,
                "state_dict_sha256": sha256_file(state),
                "provenance": {
                    "protocol_sha256": "protocol",
                    "source_lock_sha256": "lock",
                    "initial_parameter_sha256": f"initial-{seed}",
                    "train_batch_order_sha256": f"order-{seed}",
                },
                "test_access_count": 0,
            }
            (root / "exp61_checkpoint.json").write_text(json.dumps(metadata))
    grid = validate_checkpoint_grid(
        protocol_sha256="protocol", source_lock_sha256="lock"
    )
    assert set(grid) == set(evaluator.VARIANTS)
    bad = tmp_path / "aligned_orthogonal_only/seed_61/epoch10/exp61_checkpoint.json"
    metadata = json.loads(bad.read_text())
    metadata["provenance"]["train_batch_order_sha256"] = "different"
    bad.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError):
        validate_checkpoint_grid(protocol_sha256="protocol", source_lock_sha256="lock")
