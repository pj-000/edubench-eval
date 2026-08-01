"""Shared deterministic multi-reference schedule for Exp54 RAR-SFT.

This module is the only production source of truth for selecting one human
reference for a row at a given seed and epoch. Schedule artifacts are
train-only and pre-tokenization: they identify the selected cleaned rationale
by hashes and IDs but do not serialize model targets or authorize training.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Iterable


SCHEDULE_SCHEMA_VERSION = "exp54-reference-event-v1"
FORMAL_SEEDS = (42, 43, 44)
FORMAL_EPOCHS = 3
EXPECTED_TRAIN_ROWS = 2654


def canonical_json_bytes(value: Any) -> bytes:
    """Return the unambiguous UTF-8 encoding used by all schedule hashes."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def schedule_offset(seed: int, record_id: str) -> int:
    """Return the frozen offset from the first 16 SHA-256 hex characters."""
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not record_id:
        raise ValueError("record_id must be nonempty")
    payload = f"{seed}|{record_id}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def schedule_index(
    seed: int,
    record_id: str,
    epoch_index: int,
    reference_count: int,
) -> int:
    """Select a zero-based reference position for a zero-based epoch."""
    if epoch_index < 0:
        raise ValueError("epoch_index must be nonnegative")
    if reference_count < 1:
        raise ValueError("reference_count must be positive")
    return (schedule_offset(seed, record_id) + epoch_index) % reference_count


def schedule_event_id(
    *,
    seed: int,
    epoch_index: int,
    row_position: int,
    record_id: str,
    selected_reference_id: str | None,
) -> str:
    """Hash one scheduled row occurrence without delimiter ambiguity."""
    if row_position < 0:
        raise ValueError("row_position must be nonnegative")
    payload = [
        SCHEDULE_SCHEMA_VERSION,
        seed,
        epoch_index,
        row_position,
        record_id,
        selected_reference_id,
    ]
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _ordered_references(
    row: dict[str, Any],
    *,
    record_id: str,
) -> list[dict[str, Any]]:
    references = list(row.get("references") or [])
    declared_count = int(row.get("reference_count", len(references)))
    if declared_count != len(references):
        raise ValueError(
            f"{record_id}: reference_count={declared_count} but found {len(references)}"
        )
    ordered = sorted(references, key=lambda reference: str(reference.get("rater_id") or ""))
    rater_ids = [str(reference.get("rater_id") or "") for reference in ordered]
    if any(not rater_id for rater_id in rater_ids):
        raise ValueError(f"{record_id}: empty rater_id")
    if len(rater_ids) != len(set(rater_ids)):
        raise ValueError(f"{record_id}: duplicate rater_id")
    return ordered


def validate_reference_inventory(reference_rows: list[dict[str, Any]]) -> None:
    """Hard-fail on an ambiguous or internally inconsistent reference inventory."""
    seen_record_ids: set[str] = set()
    seen_reference_ids: set[str] = set()
    for row_position, row in enumerate(reference_rows):
        record_id = str(row.get("record_id") or "")
        if not record_id:
            raise ValueError(f"row {row_position}: missing record_id")
        if record_id in seen_record_ids:
            raise ValueError(f"duplicate record_id: {record_id}")
        seen_record_ids.add(record_id)
        normalized_qa_key = str(row.get("normalized_qa_key") or "")
        if not normalized_qa_key:
            raise ValueError(f"{record_id}: missing normalized_qa_key")
        label = int(row["label_5"])
        if label not in range(1, 6):
            raise ValueError(f"{record_id}: label_5 outside 1-5")
        if not str(row.get("metric_id") or ""):
            raise ValueError(f"{record_id}: missing metric_id")
        if not str(row.get("language") or ""):
            raise ValueError(f"{record_id}: missing language")

        for reference in _ordered_references(row, record_id=record_id):
            reference_id = str(reference.get("reference_id") or "")
            if not reference_id:
                raise ValueError(f"{record_id}: missing reference_id")
            if reference_id in seen_reference_ids:
                raise ValueError(f"duplicate reference_id: {reference_id}")
            seen_reference_ids.add(reference_id)
            reason = str(reference.get("reason") or "")
            reason_hash = str(reference.get("clean_reason_sha256") or "")
            if not reason or not reason_hash:
                raise ValueError(f"{reference_id}: missing cleaned reason or hash")
            actual_hash = hashlib.sha256(reason.encode("utf-8")).hexdigest()
            if actual_hash != reason_hash:
                raise ValueError(f"{reference_id}: cleaned reason bytes hash mismatch")


def build_schedule_events(
    reference_rows: list[dict[str, Any]],
    *,
    seed: int,
    epochs: int = FORMAL_EPOCHS,
) -> list[dict[str, Any]]:
    """Expand ordered rows into epoch-major scheduled training occurrences."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    validate_reference_inventory(reference_rows)
    events: list[dict[str, Any]] = []
    for epoch_index in range(epochs):
        for row_position, row in enumerate(reference_rows):
            record_id = str(row["record_id"])
            references = _ordered_references(row, record_id=record_id)
            selected: dict[str, Any] | None = None
            if references:
                selected = references[
                    schedule_index(seed, record_id, epoch_index, len(references))
                ]
            selected_reference_id = (
                str(selected["reference_id"]) if selected is not None else None
            )
            event = {
                "schedule_schema_version": SCHEDULE_SCHEMA_VERSION,
                "event_id": schedule_event_id(
                    seed=seed,
                    epoch_index=epoch_index,
                    row_position=row_position,
                    record_id=record_id,
                    selected_reference_id=selected_reference_id,
                ),
                "seed": seed,
                "epoch_index": epoch_index,
                "epoch_number": epoch_index + 1,
                "row_position": row_position,
                "record_id": record_id,
                "selected_reference_id": selected_reference_id,
                "selected_rater_id": (
                    str(selected["rater_id"]) if selected is not None else None
                ),
                "reference_count": len(references),
                "rationale_available": selected is not None,
                "selected_reason_bytes_sha256": (
                    str(selected["clean_reason_sha256"])
                    if selected is not None
                    else None
                ),
                "label_5": int(row["label_5"]),
                "metric_id": str(row["metric_id"]),
                "language": str(row["language"]),
                "normalized_qa_key": str(row["normalized_qa_key"]),
                "stratum_id": (
                    f"seed={seed}|epoch={epoch_index + 1}|"
                    f"label={int(row['label_5'])}|metric={row['metric_id']}|"
                    f"language={row['language']}"
                ),
            }
            events.append(event)
    return events


def vector_sha256(values: Iterable[Any]) -> str:
    return sha256_json(list(values))


def validate_schedule_events(
    reference_rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    seed: int,
    epochs: int = FORMAL_EPOCHS,
) -> dict[str, Any]:
    """Validate exact completeness, order, rotation, and deterministic identity."""
    expected = build_schedule_events(reference_rows, seed=seed, epochs=epochs)
    if events != expected:
        raise ValueError("schedule events differ from the shared deterministic schedule")

    expected_count = len(reference_rows) * epochs
    event_ids = [str(event["event_id"]) for event in events]
    if len(events) != expected_count:
        raise ValueError(f"expected {expected_count} events, found {len(events)}")
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("duplicate event_id")

    ordered_row_ids = [str(row["record_id"]) for row in reference_rows]
    by_epoch: dict[int, list[dict[str, Any]]] = {
        epoch_index: [
            event for event in events if int(event["epoch_index"]) == epoch_index
        ]
        for epoch_index in range(epochs)
    }
    for epoch_index, epoch_events in by_epoch.items():
        if len(epoch_events) != len(reference_rows):
            raise ValueError(
                f"epoch {epoch_index}: expected {len(reference_rows)} events, "
                f"found {len(epoch_events)}"
            )
        if [event["record_id"] for event in epoch_events] != ordered_row_ids:
            raise ValueError(f"epoch {epoch_index}: ordered row vector changed")
        if [event["row_position"] for event in epoch_events] != list(
            range(len(reference_rows))
        ):
            raise ValueError(f"epoch {epoch_index}: row positions changed")

    reference_frequency = Counter(
        str(event["selected_reference_id"])
        for event in events
        if event["rationale_available"]
    )
    inventory_reference_ids = {
        str(reference["reference_id"])
        for row in reference_rows
        for reference in row.get("references") or []
    }
    if set(reference_frequency) != inventory_reference_ids:
        raise ValueError("scheduled reference set differs from reference inventory")

    expected_frequency_by_count = {
        2: sorted((1, 2)),
        3: sorted((1, 1, 1)),
    }
    for row in reference_rows:
        references = _ordered_references(row, record_id=str(row["record_id"]))
        if len(references) in expected_frequency_by_count and epochs == 3:
            actual = sorted(
                reference_frequency[str(reference["reference_id"])]
                for reference in references
            )
            if actual != expected_frequency_by_count[len(references)]:
                raise ValueError(
                    f"{row['record_id']}: three-epoch rotation frequency mismatch"
                )

    rationale_events = [event for event in events if event["rationale_available"]]
    return {
        "seed": seed,
        "epochs": epochs,
        "row_events": len(events),
        "unique_row_ids": len(ordered_row_ids),
        "unique_event_ids": len(set(event_ids)),
        "rationale_available_events": len(rationale_events),
        "score_only_events": len(events) - len(rationale_events),
        "unique_selected_reference_ids": len(reference_frequency),
        "repeated_reference_occurrences": (
            len(rationale_events) - len(reference_frequency)
        ),
        "ordered_row_id_vector_sha256": vector_sha256(ordered_row_ids),
        "row_id_set_sha256": vector_sha256(sorted(ordered_row_ids)),
        "event_id_vector_sha256": vector_sha256(event_ids),
        "selected_reference_vector_sha256": vector_sha256(
            [event["selected_reference_id"] for event in events]
        ),
        "selected_reason_hash_vector_sha256": vector_sha256(
            [event["selected_reason_bytes_sha256"] for event in events]
        ),
        "checks": {
            "event_count_exact": len(events) == expected_count,
            "event_ids_unique": len(event_ids) == len(set(event_ids)),
            "row_order_identical_each_epoch": True,
            "selection_matches_shared_schedule": True,
            "all_inventory_references_selected": (
                set(reference_frequency) == inventory_reference_ids
            ),
            "three_epoch_rotation_exact": True,
        },
    }
