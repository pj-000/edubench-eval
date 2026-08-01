"""Shared pure contract for resolving each arm's semantic target source."""

from __future__ import annotations

from typing import Any

from thesis_exp.exp54_rar_sft.reference_schedule import schedule_index
from thesis_exp.exp54_rar_sft.training_contract import sha256_bytes


ARMS = ("S0", "R1", "R2", "R3")


def index_unique_rows(
    rows: list[dict[str, Any]],
    key: str,
    *,
    source: str,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key) or "")
        if not value:
            raise ValueError(f"{source}: empty {key}")
        if value in output:
            raise ValueError(f"{source}: duplicate {key}: {value}")
        output[value] = row
    return output


def reference_indexes(
    rows: list[dict[str, Any]],
    *,
    source: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_record = index_unique_rows(rows, "record_id", source=source)
    by_reference: dict[str, dict[str, Any]] = {}
    for record_id, row in by_record.items():
        references = list(row.get("references") or [])
        if int(row["reference_count"]) != len(references):
            raise ValueError(f"{record_id}: declared reference_count differs")
        seen_raters: set[str] = set()
        for reference in references:
            reference_id = str(reference.get("reference_id") or "")
            rater_id = str(reference.get("rater_id") or "")
            reason = str(reference.get("reason") or "")
            if not reference_id or not rater_id or not reason:
                raise ValueError(f"{record_id}: incomplete reference")
            if rater_id in seen_raters:
                raise ValueError(f"{record_id}: duplicate rater_id {rater_id}")
            if reference_id in by_reference:
                raise ValueError(f"{source}: duplicate reference_id {reference_id}")
            seen_raters.add(rater_id)
            expected_hash = str(reference.get("clean_reason_sha256") or "")
            actual_hash = sha256_bytes(reason.encode("utf-8"))
            if expected_hash != actual_hash:
                raise ValueError(f"{reference_id}: clean reason hash differs")
            by_reference[reference_id] = {
                **reference,
                "record_id": record_id,
                "label_5": int(row["label_5"]),
                "metric_id": str(row["metric_id"]),
                "language": str(row["language"]),
                "normalized_qa_key": str(row["normalized_qa_key"]),
            }
    return by_record, by_reference


def resolve_expected_arm_source(
    *,
    arm: str,
    train_row: dict[str, Any],
    base_event: dict[str, Any],
    all_references_by_record: dict[str, dict[str, Any]],
    consistent_references_by_id: dict[str, dict[str, Any]],
    donor_by_event: dict[str, dict[str, Any]],
    mask_by_event: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Resolve the exact raw rationale/provenance expected for one arm event."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    event_id = str(base_event["event_id"])
    record_id = str(base_event["record_id"])
    if record_id != str(train_row["record_id"]):
        raise ValueError(f"{event_id}: base event belongs to another train row")
    if arm == "S0":
        return {
            "rationale": "",
            "rationale_active": False,
            "arm_selected_reference_id": None,
            "arm_rationale_source_event_id": None,
            "inactive_reason": "score_only_arm",
        }

    if arm == "R1":
        reference_row = all_references_by_record[record_id]
        references = list(reference_row.get("references") or [])
        if not references:
            return {
                "rationale": "",
                "rationale_active": False,
                "arm_selected_reference_id": None,
                "arm_rationale_source_event_id": None,
                "inactive_reason": "no_human_reference",
            }
        reference = references[
            schedule_index(
                int(base_event["seed"]),
                record_id,
                int(base_event["epoch_index"]),
                len(references),
            )
        ]
        return {
            "rationale": str(reference["reason"]),
            "rationale_active": True,
            "arm_selected_reference_id": str(reference["reference_id"]),
            "arm_rationale_source_event_id": event_id,
            "inactive_reason": "",
        }

    mask = mask_by_event[event_id]
    donor = donor_by_event[event_id]
    active = bool(mask["rationale_active"])
    if bool(donor["active"]) != active:
        raise ValueError(f"{event_id}: donor and R2/R3 mask activity differ")
    if not active:
        return {
            "rationale": "",
            "rationale_active": False,
            "arm_selected_reference_id": None,
            "arm_rationale_source_event_id": None,
            "inactive_reason": str(mask["inactive_reason"]),
        }

    reference_id = (
        str(donor["donor_reference_id"])
        if arm == "R2"
        else str(base_event["selected_reference_id"])
    )
    source_event_id = (
        str(donor["donor_event_id"]) if arm == "R2" else event_id
    )
    reference = consistent_references_by_id.get(reference_id)
    if reference is None:
        raise ValueError(f"{event_id}: missing {arm} reference {reference_id}")
    expected_reason_hash = (
        str(donor["donor_reason_bytes_sha256"])
        if arm == "R2"
        else str(base_event["selected_reason_bytes_sha256"])
    )
    if expected_reason_hash != reference["clean_reason_sha256"]:
        raise ValueError(f"{event_id}: {arm} reason hash differs from source")
    if arm == "R2":
        source_event = donor_by_event.get(source_event_id)
        if source_event is None:
            raise ValueError(f"{event_id}: donor source event is outside closure")
        if str(source_event["recipient_reference_id"]) != reference_id:
            raise ValueError(f"{event_id}: donor event/reference backlink differs")
    return {
        "rationale": str(reference["reason"]),
        "rationale_active": True,
        "arm_selected_reference_id": reference_id,
        "arm_rationale_source_event_id": source_event_id,
        "inactive_reason": "",
    }


def source_fingerprint(source: dict[str, Any]) -> dict[str, Any]:
    """Return a non-text source representation suitable for vector hashing."""
    rationale = str(source["rationale"])
    return {
        "rationale_bytes_sha256": sha256_bytes(rationale.encode("utf-8")),
        "rationale_active": bool(source["rationale_active"]),
        "arm_selected_reference_id": source["arm_selected_reference_id"],
        "arm_rationale_source_event_id": source[
            "arm_rationale_source_event_id"
        ],
        "inactive_reason": str(source["inactive_reason"]),
    }
