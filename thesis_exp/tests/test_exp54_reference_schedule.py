from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from thesis_exp.exp54_rar_sft.build_reference_schedule_events import cross_seed_checks
from thesis_exp.exp54_rar_sft.reference_schedule import (
    FORMAL_SEEDS,
    SCHEDULE_SCHEMA_VERSION,
    build_schedule_events,
    schedule_event_id,
    schedule_index,
    schedule_offset,
    validate_reference_inventory,
    validate_schedule_events,
)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _reference(record_id: str, rater_id: str, reason: str) -> dict:
    return {
        "reference_id": f"{record_id}:{rater_id}",
        "rater_id": rater_id,
        "reason": reason,
        "clean_reason_sha256": _sha256(reason),
    }


def _row(record_id: str, reasons: list[tuple[str, str]]) -> dict:
    references = [
        _reference(record_id, rater_id, reason) for rater_id, reason in reasons
    ]
    return {
        "record_id": record_id,
        "normalized_qa_key": _sha256(f"qa:{record_id}"),
        "label_5": 2,
        "metric_id": "IFTC",
        "language": "zh",
        "reference_count": len(references),
        "references": references,
    }


def _small_inventory() -> list[dict]:
    return [
        _row("no-reason", []),
        _row("two", [("human_2", "two-b"), ("human_1", "two-a")]),
        _row(
            "three",
            [
                ("human_3", "three-c"),
                ("human_1", "three-a"),
                ("human_2", "three-b"),
            ],
        ),
    ]


def test_frozen_schedule_probe_matches_exact_sha256_prefix_behavior() -> None:
    assert schedule_offset(42, "sample-a") == 746608230635606418
    assert [schedule_index(42, "sample-a", epoch, 3) for epoch in range(3)] == [
        0,
        1,
        2,
    ]
    assert [schedule_index(43, "sample-a", epoch, 3) for epoch in range(3)] == [
        1,
        2,
        0,
    ]


def test_event_id_is_deterministic_and_epoch_specific() -> None:
    first = schedule_event_id(
        seed=42,
        epoch_index=0,
        row_position=7,
        record_id="row-a",
        selected_reference_id="row-a:human_1",
    )
    same = schedule_event_id(
        seed=42,
        epoch_index=0,
        row_position=7,
        record_id="row-a",
        selected_reference_id="row-a:human_1",
    )
    later_epoch = schedule_event_id(
        seed=42,
        epoch_index=1,
        row_position=7,
        record_id="row-a",
        selected_reference_id="row-a:human_1",
    )
    assert first == same
    assert first != later_epoch
    assert len(first) == 64


def test_builds_epoch_major_events_and_rotates_sorted_raters() -> None:
    rows = _small_inventory()
    events = build_schedule_events(rows, seed=42, epochs=3)
    assert len(events) == 9
    assert [event["record_id"] for event in events] == [
        "no-reason",
        "two",
        "three",
    ] * 3
    assert [event["row_position"] for event in events] == [0, 1, 2] * 3

    no_reason_events = [event for event in events if event["record_id"] == "no-reason"]
    assert all(event["selected_reference_id"] is None for event in no_reason_events)
    assert all(event["rationale_available"] is False for event in no_reason_events)

    selected_two = [
        event["selected_reference_id"] for event in events if event["record_id"] == "two"
    ]
    assert set(selected_two) == {"two:human_1", "two:human_2"}
    assert sorted(Counter(selected_two).values()) == [1, 2]

    selected_three = [
        event["selected_reference_id"]
        for event in events
        if event["record_id"] == "three"
    ]
    assert Counter(selected_three) == {
        "three:human_1": 1,
        "three:human_2": 1,
        "three:human_3": 1,
    }
    assert len({event["event_id"] for event in events}) == len(events)


def test_validation_rejects_duplicate_missing_or_modified_events() -> None:
    rows = _small_inventory()
    events = build_schedule_events(rows, seed=42, epochs=3)
    validate_schedule_events(rows, events, seed=42, epochs=3)

    with pytest.raises(ValueError, match="differ"):
        validate_schedule_events(rows, events[:-1], seed=42, epochs=3)

    duplicated = [*events[:-1], events[-2]]
    with pytest.raises(ValueError, match="differ"):
        validate_schedule_events(rows, duplicated, seed=42, epochs=3)

    modified = [dict(event) for event in events]
    modified[0]["row_position"] = 99
    with pytest.raises(ValueError, match="differ"):
        validate_schedule_events(rows, modified, seed=42, epochs=3)


def test_inventory_hard_fails_on_duplicate_ids_and_reason_hash_changes() -> None:
    rows = _small_inventory()
    duplicated = [rows[0], dict(rows[0])]
    with pytest.raises(ValueError, match="duplicate record_id"):
        validate_reference_inventory(duplicated)

    changed = _small_inventory()
    changed[1]["references"][0]["clean_reason_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bytes hash mismatch"):
        validate_reference_inventory(changed)


def test_cross_seed_base_metadata_is_identical() -> None:
    rows = _small_inventory()
    schedules = {
        seed: build_schedule_events(rows, seed=seed, epochs=3)
        for seed in FORMAL_SEEDS
    }
    result = cross_seed_checks(rows, schedules)
    assert all(result["checks"].values())


def test_event_schema_accepts_generated_events() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = Path(
        "thesis_exp/exp54_rar_sft/schemas/reference_schedule_event.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    events = build_schedule_events(_small_inventory(), seed=42, epochs=3)
    for event in events:
        jsonschema.validate(event, schema)
    assert events[0]["schedule_schema_version"] == SCHEDULE_SCHEMA_VERSION


def test_real_train_inventory_expands_to_exact_formal_counts() -> None:
    path = Path(
        "thesis_exp/outputs/exp54_rar_sft/rar_v2/data/"
        "label_consistent_reference_sets.jsonl"
    )
    if not path.exists():
        pytest.skip("private train-only reference inventory is unavailable")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 2654
    for seed in FORMAL_SEEDS:
        events = build_schedule_events(rows, seed=seed, epochs=3)
        summary = validate_schedule_events(rows, events, seed=seed, epochs=3)
        assert summary["row_events"] == 7962
        assert summary["unique_row_ids"] == 2654
        assert summary["rationale_available_events"] == 4836
        assert summary["score_only_events"] == 3126
        assert summary["unique_selected_reference_ids"] == 3934
        assert summary["repeated_reference_occurrences"] == 902
        assert all(summary["checks"].values())
