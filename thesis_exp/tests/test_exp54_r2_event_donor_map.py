from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pytest

from thesis_exp.exp54_rar_sft.audit_r2_event_solver_oracle import (
    brute_force_objective,
    mapping_objective,
    run_audit,
    synthetic_event,
)
from thesis_exp.exp54_rar_sft.build_r2_event_donor_map import (
    build_event_donor_map,
    match_event_stratum,
    validate_event_map,
)
from thesis_exp.exp54_rar_sft.build_r2_r3_event_mask import (
    build_event_mask_rows,
    summarize_mask,
)
from thesis_exp.exp54_rar_sft.reference_schedule import build_schedule_events


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _row(
    record_id: str,
    *,
    reason: str | None,
    label: int = 2,
    qa_key: str | None = None,
) -> dict:
    references = (
        [
            {
                "reference_id": f"{record_id}:human_1",
                "rater_id": "human_1",
                "reason": reason,
                "clean_reason_sha256": _sha256(reason),
            }
        ]
        if reason is not None
        else []
    )
    return {
        "record_id": record_id,
        "normalized_qa_key": qa_key or _sha256(f"qa:{record_id}"),
        "label_5": label,
        "metric_id": "IFTC",
        "language": "zh",
        "reference_count": len(references),
        "references": references,
    }


def _feature(text: str) -> tuple[int, str]:
    token_ids = [ord(character) for character in text]
    return (
        len(token_ids),
        hashlib.sha256(
            json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    )


def _inventory() -> list[dict]:
    return [
        _row("a", reason="a"),
        _row("b", reason="bb"),
        _row("c", reason="ccc"),
        _row("d", reason="dddd"),
        _row("score-only", reason=None),
    ]


def test_event_matcher_preserves_each_epoch_and_prefix_counter() -> None:
    rows = _inventory()
    events = build_schedule_events(rows, seed=42, epochs=3)
    mapping = build_event_donor_map(
        rows,
        events,
        seed=42,
        feature_fn=_feature,
    )
    report = validate_event_map(events, mapping, seed=42)
    assert all(report["checks"].values())
    assert report["counts"]["base_row_events"] == 15
    assert report["counts"]["rationale_eligible_events"] == 12
    assert report["counts"]["active_rationale_events"] == 12
    assert report["counts"]["score_only_events"] == 3
    for prefix in report["checkpoint_prefix_frequency"]:
        assert prefix["rationale_bytes_frequency_delta_l1"] == 0
        assert prefix["rationale_token_ids_frequency_delta_l1"] == 0
        assert (
            prefix["recipient_supervised_tokens"]
            == prefix["donor_supervised_tokens"]
        )


def test_same_reference_across_epochs_uses_distinct_donor_events() -> None:
    rows = _inventory()
    events = build_schedule_events(rows, seed=42, epochs=3)
    mapping = build_event_donor_map(
        rows,
        events,
        seed=42,
        feature_fn=_feature,
    )
    a_rows = [
        row for row in mapping if row["recipient_reference_id"] == "a:human_1"
    ]
    assert len(a_rows) == 3
    assert len({row["recipient_event_id"] for row in a_rows}) == 3
    assert len({row["donor_event_id"] for row in mapping if row["active"]}) == 12


def test_score_only_event_is_inactive_without_score_mask_fields() -> None:
    rows = _inventory()
    events = build_schedule_events(rows, seed=42, epochs=3)
    mapping = build_event_donor_map(
        rows,
        events,
        seed=42,
        feature_fn=_feature,
    )
    score_only = [
        row for row in mapping if row["recipient_record_id"] == "score-only"
    ]
    assert len(score_only) == 3
    assert all(not row["active"] for row in score_only)
    assert all(
        row["inactive_reason"] == "no_label_consistent_reference"
        for row in score_only
    )
    masks = build_event_mask_rows(events, mapping)
    assert all("score_mask" not in row for row in masks)
    assert summarize_mask(masks)["active_rationale_events"] == 12


def test_same_normalized_qa_events_cannot_donate_to_each_other() -> None:
    duplicate_qa = _sha256("duplicate")
    rows = [
        _row("a", reason="a", qa_key=duplicate_qa),
        _row("b", reason="b", qa_key=duplicate_qa),
    ]
    events = build_schedule_events(rows, seed=42, epochs=3)
    mapping = build_event_donor_map(
        rows,
        events,
        seed=42,
        feature_fn=_feature,
    )
    assert all(not row["active"] for row in mapping)
    report = validate_event_map(events, mapping, seed=42)
    assert report["counts"]["inactive_eligible_rationale_events"] == 6


def test_event_stratum_rejects_mixed_epoch_input() -> None:
    items = [
        synthetic_event("a", "event-a", 1),
        {**synthetic_event("b", "event-b", 2), "epoch_index": 1, "epoch_number": 2},
    ]
    with pytest.raises(ValueError, match="multiple structured strata"):
        match_event_stratum(items)


def test_event_mapping_is_invariant_to_item_order() -> None:
    items = [
        synthetic_event("a", "event-a", 1),
        synthetic_event("b", "event-b", 2),
        synthetic_event("c", "event-c", 20),
        synthetic_event("d", "event-d", 21),
    ]
    expected = match_event_stratum(items)
    randomizer = random.Random(20260723)
    for _ in range(20):
        shuffled = list(items)
        randomizer.shuffle(shuffled)
        assert match_event_stratum(shuffled) == expected


def test_event_matcher_matches_independent_small_oracle() -> None:
    items = [
        synthetic_event("a", "event-a", 1),
        synthetic_event("b", "event-b", 4),
        synthetic_event("c", "event-c", 30),
    ]
    assert mapping_objective(match_event_stratum(items)) == brute_force_objective(items)


def test_formal_oracle_suite_passes() -> None:
    report = run_audit(seed=20260723, random_cases=16)
    assert report["status"] == "R2_EVENT_MATCHER_ORACLE_PASS"
    assert report["objective_failures"] == []
    assert report["input_order_failures"] == []


def test_validation_rejects_duplicate_missing_and_extra_map_rows() -> None:
    rows = _inventory()
    events = build_schedule_events(rows, seed=42, epochs=3)
    mapping = build_event_donor_map(
        rows,
        events,
        seed=42,
        feature_fn=_feature,
    )
    with pytest.raises(ValueError, match="exactly cover"):
        validate_event_map(events, mapping[:-1], seed=42)
    with pytest.raises(ValueError, match="duplicate recipient"):
        validate_event_map(events, [*mapping, mapping[-1]], seed=42)


def test_formal_public_reports_lock_exact_event_control_counts() -> None:
    base = Path("thesis_exp/outputs/exp54_rar_sft/rar_v2")
    oracle_path = base / "audit/r2_event_solver_oracle_report.json"
    mask_path = base / "audit/r2_r3_event_mask_report.json"
    donor_lock_path = base / "protocol/r2_event_donor_map_candidate_lock.json"
    if not all(path.exists() for path in (oracle_path, mask_path, donor_lock_path)):
        pytest.skip("formal public event-control reports are unavailable")

    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    assert oracle["status"] == "R2_EVENT_MATCHER_ORACLE_PASS"
    assert oracle["objective_failures"] == []
    assert oracle["input_order_failures"] == []

    donor_lock = json.loads(donor_lock_path.read_text(encoding="utf-8"))
    assert donor_lock["status"] == "R2_EVENT_DONOR_MAP_CANDIDATE_READY"
    assert donor_lock["manifest_freeze_allowed"] is False
    assert donor_lock["formal_training_allowed"] is False

    for seed in (42, 43, 44):
        report_path = (
            base / "audit" / f"r2_event_donor_match_report_seed{seed}.json"
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["status"] == "R2_EVENT_DONOR_MAP_CANDIDATE_READY"
        assert report["counts"]["base_row_events"] == 7962
        assert report["counts"]["rationale_eligible_events"] == 4836
        assert report["counts"]["active_rationale_events"] == 4803
        assert report["counts"]["inactive_eligible_rationale_events"] == 33
        assert all(report["checks"].values())
        assert all(
            row["rationale_bytes_frequency_delta_l1"] == 0
            and row["rationale_token_ids_frequency_delta_l1"] == 0
            and row["recipient_supervised_tokens"]
            == row["donor_supervised_tokens"]
            for row in report["individual_epoch_frequency"]
            + report["checkpoint_prefix_frequency"]
        )

    mask_report = json.loads(mask_path.read_text(encoding="utf-8"))
    assert mask_report["status"] == "R2_R3_EVENT_MASK_CANDIDATE_READY"
    assert mask_report["candidate_manifest_build_allowed"] is False
    assert all(
        all(seed_report["checks"].values())
        and seed_report["r2_event_mask_sha256"]
        == seed_report["r3_event_mask_sha256"]
        for seed_report in mask_report["seed_reports"]
    )
