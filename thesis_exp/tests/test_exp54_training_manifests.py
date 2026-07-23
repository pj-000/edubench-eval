import json
from pathlib import Path

import pytest

from thesis_exp.exp54_rar_sft.build_training_manifests import (
    select_arm_rationale,
)
from thesis_exp.exp54_rar_sft.reference_schedule import schedule_index


def _event() -> dict:
    return {
        "event_id": "recipient-event",
        "seed": 42,
        "epoch_index": 1,
        "record_id": "recipient-row",
        "selected_reference_id": "recipient-ref",
        "selected_reason_bytes_sha256": "recipient-hash",
    }


def _inputs() -> dict:
    return {
        "all_references_by_record": {
            "recipient-row": {
                "references": [
                    {
                        "reference_id": "all-ref-1",
                        "reason": "all reason one",
                    },
                    {
                        "reference_id": "all-ref-2",
                        "reason": "all reason two",
                    },
                    {
                        "reference_id": "all-ref-3",
                        "reason": "all reason three",
                    },
                ]
            }
        },
        "consistent_references_by_id": {
            "recipient-ref": {
                "reference_id": "recipient-ref",
                "reason": "aligned reason",
                "clean_reason_sha256": "recipient-hash",
            },
            "donor-ref": {
                "reference_id": "donor-ref",
                "reason": "shuffled reason",
                "clean_reason_sha256": "donor-hash",
            },
        },
        "donor_by_event": {
            "recipient-event": {
                "active": True,
                "donor_event_id": "donor-event",
                "donor_reference_id": "donor-ref",
                "donor_reason_bytes_sha256": "donor-hash",
            },
            "donor-event": {
                "active": True,
                "recipient_reference_id": "donor-ref",
            },
        },
        "mask_by_event": {
            "recipient-event": {
                "rationale_active": True,
                "inactive_reason": "",
            }
        },
    }


def test_s0_is_score_only_and_r1_uses_shared_all_rater_schedule() -> None:
    event = _event()
    inputs = _inputs()

    s0 = select_arm_rationale(arm="S0", event=event, **inputs)
    r1 = select_arm_rationale(arm="R1", event=event, **inputs)
    expected_index = schedule_index(42, "recipient-row", 1, 3)

    assert s0["rationale"] == ""
    assert s0["rationale_active"] is False
    assert r1["rationale"] == inputs["all_references_by_record"][
        "recipient-row"
    ]["references"][expected_index]["reason"]
    assert r1["rationale_active"] is True
    assert r1["arm_rationale_source_event_id"] == "recipient-event"


def test_r2_uses_donor_event_while_r3_uses_base_selected_reference() -> None:
    event = _event()
    inputs = _inputs()

    r2 = select_arm_rationale(arm="R2", event=event, **inputs)
    r3 = select_arm_rationale(arm="R3", event=event, **inputs)

    assert r2["rationale"] == "shuffled reason"
    assert r2["arm_selected_reference_id"] == "donor-ref"
    assert r2["arm_rationale_source_event_id"] == "donor-event"
    assert r3["rationale"] == "aligned reason"
    assert r3["arm_selected_reference_id"] == "recipient-ref"
    assert r3["arm_rationale_source_event_id"] == "recipient-event"


def test_r2_r3_inactive_event_materializes_empty_rationale_symmetrically() -> None:
    event = _event()
    inputs = _inputs()
    inputs["donor_by_event"]["recipient-event"]["active"] = False
    inputs["mask_by_event"]["recipient-event"] = {
        "rationale_active": False,
        "inactive_reason": "no_legal_strict_event_permutation",
    }

    r2 = select_arm_rationale(arm="R2", event=event, **inputs)
    r3 = select_arm_rationale(arm="R3", event=event, **inputs)

    assert r2 == r3
    assert r2["rationale"] == ""
    assert r2["rationale_active"] is False


def test_donor_mask_activity_mismatch_is_a_hard_failure() -> None:
    inputs = _inputs()
    inputs["mask_by_event"]["recipient-event"]["rationale_active"] = False

    with pytest.raises(ValueError, match="activity differ"):
        select_arm_rationale(arm="R2", event=_event(), **inputs)


def test_formal_materialized_candidate_report_is_aggregate_and_not_ready() -> None:
    base = Path("thesis_exp/outputs/exp54_rar_sft/rar_v2")
    report_path = base / "audit/materialized_manifest_candidate_report.json"
    lock_path = base / "protocol/materialized_manifest_candidate_lock.json"
    if not report_path.exists() or not lock_path.exists():
        pytest.skip("formal materialized-manifest report is unavailable")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert report["status"] == (
        "MATERIALIZED_MANIFEST_CANDIDATE_AUDITED_NOT_FROZEN"
    )
    assert lock["status"] == report["status"]
    for artifact in (report, lock):
        assert artifact["manifest_freeze_allowed"] is False
        assert artifact["smoke_training_allowed"] is False
        assert artifact["formal_training_allowed"] is False
        assert artifact["dev_accessed"] is False
        assert artifact["test_accessed"] is False
        assert artifact["training_used"] is False

    assert [seed["seed"] for seed in report["seed_reports"]] == [42, 43, 44]
    for seed in report["seed_reports"]:
        assert all(seed["checks"].values())
        budget = seed["training_budget"]
        assert all(budget["checks"].values())
        assert budget["optimizer_steps_total"] == 996
        padded = {
            arm: values["fixed_padded_input_tokens"]
            for arm, values in budget["per_arm"].items()
        }
        assert set(padded.values()) == {16_306_176}
        for values in budget["per_arm"].values():
            assert values["score_supervised_events"] == 7_962
            assert values["maximum_sequence_tokens"] <= 2_048
        assert (
            budget["per_arm"]["R2"]["unpadded_sequence_tokens"]
            == budget["per_arm"]["R3"]["unpadded_sequence_tokens"]
        )
        assert (
            budget["per_arm"]["R2"]["rationale_supervised_events"]
            == budget["per_arm"]["R3"]["rationale_supervised_events"]
            == 4_803
        )
        for epoch in seed["r2_r3_frequency_control"]["individual_epochs"]:
            assert epoch["active_rationale_events"] == 1_601
            assert not any(epoch["frequency_l1_difference"].values())

    forbidden_public_keys = {
        "record_id",
        "reference_id",
        "base_event_id",
        "target_text",
        "reason",
        "rationale",
    }

    def collect_keys(value):
        if isinstance(value, dict):
            return set(value) | set().union(
                *(collect_keys(item) for item in value.values())
            )
        if isinstance(value, list):
            return set().union(*(collect_keys(item) for item in value))
        return set()

    assert not (collect_keys(report) & forbidden_public_keys)
    assert not (collect_keys(lock) & forbidden_public_keys)
