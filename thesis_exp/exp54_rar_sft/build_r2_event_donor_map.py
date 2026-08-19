"""Build per-seed, per-epoch R2 event-level strict donor permutations.

The production matching unit is one scheduled rationale occurrence, not one
static reference identity. Matching is independent within the structured tuple
``seed × epoch_index × label_5 × metric_id × language``. Row-level maps stay
private; public artifacts contain only aggregate validation and hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import scipy

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.audit_rar0_alignment import (
    file_sha256,
    read_jsonl,
    reject_eval_path,
    write_json,
    write_jsonl,
)
from thesis_exp.exp54_rar_sft.build_r2_donor_map import (
    match_stratum,
    tokenizer_length_function,
)
from thesis_exp.exp54_rar_sft.reference_schedule import (
    FORMAL_EPOCHS,
    FORMAL_SEEDS,
    validate_schedule_events,
)


DEFAULT_REFERENCES = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/data/"
    "label_consistent_reference_sets.jsonl"
)
DEFAULT_OUTPUT = REPO_ROOT / "thesis_exp/outputs/exp54_rar_sft/rar_v2"
DEFAULT_TOKENIZER_LOCK_SPEC = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/qwen_tokenizer_lock_spec.json"
)
DEFAULT_ORACLE_REPORT = (
    DEFAULT_OUTPUT / "audit/r2_event_solver_oracle_report.json"
)
CORE_MATCHER_SOURCE = (
    REPO_ROOT / "thesis_exp/exp54_rar_sft/build_r2_donor_map.py"
)

RationaleFeatureFn = Callable[[str], tuple[int, str]]


def event_group_key(item: dict[str, Any]) -> tuple[int, int, int, str, str]:
    return (
        int(item["seed"]),
        int(item["epoch_index"]),
        int(item["label_5"]),
        str(item["metric_id"]),
        str(item["language"]),
    )


def event_stratum_id(item: dict[str, Any]) -> str:
    seed, epoch_index, label, metric, language = event_group_key(item)
    return (
        f"seed={seed}|epoch={epoch_index + 1}|label={label}|"
        f"metric={metric}|language={language}"
    )


def _reference_index(
    reference_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in reference_rows:
        record_id = str(row["record_id"])
        for reference in row.get("references") or []:
            reference_id = str(reference["reference_id"])
            if reference_id in output:
                raise ValueError(f"duplicate reference_id: {reference_id}")
            output[reference_id] = {
                "record_id": record_id,
                "normalized_qa_key": str(row["normalized_qa_key"]),
                "label_5": int(row["label_5"]),
                "metric_id": str(row["metric_id"]),
                "language": str(row["language"]),
                "reference_id": reference_id,
                "reason": str(reference["reason"]),
                "reason_bytes_sha256": str(reference["clean_reason_sha256"]),
            }
    return output


def prepare_event_items(
    reference_rows: list[dict[str, Any]],
    schedule_events: list[dict[str, Any]],
    *,
    seed: int,
    feature_fn: RationaleFeatureFn,
) -> list[dict[str, Any]]:
    validate_schedule_events(
        reference_rows,
        schedule_events,
        seed=seed,
        epochs=FORMAL_EPOCHS,
    )
    references = _reference_index(reference_rows)
    feature_by_reference: dict[str, tuple[int, str]] = {}
    items: list[dict[str, Any]] = []
    for event in schedule_events:
        if not event["rationale_available"]:
            if event["selected_reference_id"] is not None:
                raise ValueError("score-only event unexpectedly has a selected reference")
            continue
        reference_id = str(event["selected_reference_id"])
        if reference_id not in references:
            raise ValueError(f"scheduled reference missing from inventory: {reference_id}")
        reference = references[reference_id]
        if str(event["record_id"]) != reference["record_id"]:
            raise ValueError(f"{event['event_id']}: selected reference belongs to another row")
        for field in ("normalized_qa_key", "label_5", "metric_id", "language"):
            if event[field] != reference[field]:
                raise ValueError(f"{event['event_id']}: {field} differs from inventory")
        if (
            str(event["selected_reason_bytes_sha256"])
            != reference["reason_bytes_sha256"]
        ):
            raise ValueError(f"{event['event_id']}: reason bytes hash differs from inventory")

        if reference_id not in feature_by_reference:
            token_length, token_ids_sha256 = feature_fn(reference["reason"])
            if token_length < 1:
                raise ValueError(f"{reference_id}: rationale token length must be positive")
            if len(token_ids_sha256) != 64:
                raise ValueError(f"{reference_id}: invalid rationale token-ID hash")
            feature_by_reference[reference_id] = (
                int(token_length),
                str(token_ids_sha256),
            )
        token_length, token_ids_sha256 = feature_by_reference[reference_id]
        items.append(
            {
                "event_id": str(event["event_id"]),
                "seed": int(event["seed"]),
                "epoch_index": int(event["epoch_index"]),
                "epoch_number": int(event["epoch_number"]),
                "row_position": int(event["row_position"]),
                "record_id": str(event["record_id"]),
                "normalized_qa_key": str(event["normalized_qa_key"]),
                "reference_id": reference_id,
                "label_5": int(event["label_5"]),
                "metric_id": str(event["metric_id"]),
                "language": str(event["language"]),
                "reason": reference["reason"],
                "reason_bytes_sha256": reference["reason_bytes_sha256"],
                "rationale_token_length": token_length,
                "rationale_token_ids_sha256": token_ids_sha256,
            }
        )
    if len({item["event_id"] for item in items}) != len(items):
        raise ValueError("duplicate rationale event ID")
    return items


def match_event_stratum(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Match one structured event stratum through the reviewed core solver."""
    if not items:
        return []
    group_keys = {event_group_key(item) for item in items}
    if len(group_keys) != 1:
        raise ValueError("match_event_stratum received multiple structured strata")
    by_event = {str(item["event_id"]): item for item in items}
    if len(by_event) != len(items):
        raise ValueError("duplicate event_id in event stratum")
    length_by_reason = {
        str(item["reason"]): int(item["rationale_token_length"]) for item in items
    }
    core_items = [
        {
            "record_id": item["record_id"],
            "normalized_qa_key": item["normalized_qa_key"],
            "reference_id": item["event_id"],
            "reference_index": item["row_position"],
            "label_5": item["label_5"],
            "metric_id": item["metric_id"],
            "language": item["language"],
            "reason": item["reason"],
            "reason_sha256": item["reason_bytes_sha256"],
        }
        for item in items
    ]
    core_mapping = match_stratum(
        core_items,
        lambda reason: length_by_reason[str(reason)],
    )
    output = []
    for row in core_mapping:
        recipient = by_event[str(row["recipient_reference_id"])]
        donor = (
            by_event[str(row["donor_reference_id"])] if row["active"] else None
        )
        output.append(
            {
                "stratum_id": event_stratum_id(recipient),
                "recipient_event_id": recipient["event_id"],
                "recipient_record_id": recipient["record_id"],
                "recipient_reference_id": recipient["reference_id"],
                "recipient_normalized_qa_key": recipient["normalized_qa_key"],
                "donor_event_id": donor["event_id"] if donor else None,
                "donor_record_id": donor["record_id"] if donor else None,
                "donor_reference_id": donor["reference_id"] if donor else None,
                "donor_normalized_qa_key": (
                    donor["normalized_qa_key"] if donor else None
                ),
                "seed": recipient["seed"],
                "epoch_index": recipient["epoch_index"],
                "epoch_number": recipient["epoch_number"],
                "row_position": recipient["row_position"],
                "label_5": recipient["label_5"],
                "metric_id": recipient["metric_id"],
                "language": recipient["language"],
                "recipient_reason_bytes_sha256": recipient["reason_bytes_sha256"],
                "donor_reason_bytes_sha256": (
                    donor["reason_bytes_sha256"] if donor else None
                ),
                "recipient_rationale_token_ids_sha256": recipient[
                    "rationale_token_ids_sha256"
                ],
                "donor_rationale_token_ids_sha256": (
                    donor["rationale_token_ids_sha256"] if donor else None
                ),
                "recipient_token_length": recipient["rationale_token_length"],
                "donor_token_length": (
                    donor["rationale_token_length"] if donor else None
                ),
                "absolute_token_length_difference": (
                    int(row["absolute_length_difference"]) if donor else None
                ),
                "active": bool(row["active"]),
                "inactive_reason": (
                    "" if row["active"] else "no_legal_strict_event_permutation"
                ),
                "cycle_id": str(row["cycle_id"]),
                "solver_name": str(row["solver_name"]),
                "solver_version": str(row["solver_version"]),
            }
        )
    return sorted(output, key=lambda row: row["recipient_event_id"])


def build_event_donor_map(
    reference_rows: list[dict[str, Any]],
    schedule_events: list[dict[str, Any]],
    *,
    seed: int,
    feature_fn: RationaleFeatureFn,
) -> list[dict[str, Any]]:
    items = prepare_event_items(
        reference_rows,
        schedule_events,
        seed=seed,
        feature_fn=feature_fn,
    )
    grouped: dict[tuple[int, int, int, str, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for item in items:
        grouped[event_group_key(item)].append(item)
    matched_by_event: dict[str, dict[str, Any]] = {}
    for key in sorted(grouped):
        for row in match_event_stratum(grouped[key]):
            event_id = str(row["recipient_event_id"])
            if event_id in matched_by_event:
                raise ValueError(f"duplicate matched recipient event: {event_id}")
            matched_by_event[event_id] = row

    output = []
    for event in schedule_events:
        event_id = str(event["event_id"])
        if event["rationale_available"]:
            if event_id not in matched_by_event:
                raise ValueError(f"eligible event missing matcher row: {event_id}")
            output.append(matched_by_event[event_id])
        else:
            output.append(
                {
                    "stratum_id": str(event["stratum_id"]),
                    "recipient_event_id": event_id,
                    "recipient_record_id": str(event["record_id"]),
                    "recipient_reference_id": None,
                    "recipient_normalized_qa_key": str(event["normalized_qa_key"]),
                    "donor_event_id": None,
                    "donor_record_id": None,
                    "donor_reference_id": None,
                    "donor_normalized_qa_key": None,
                    "seed": int(event["seed"]),
                    "epoch_index": int(event["epoch_index"]),
                    "epoch_number": int(event["epoch_number"]),
                    "row_position": int(event["row_position"]),
                    "label_5": int(event["label_5"]),
                    "metric_id": str(event["metric_id"]),
                    "language": str(event["language"]),
                    "recipient_reason_bytes_sha256": None,
                    "donor_reason_bytes_sha256": None,
                    "recipient_rationale_token_ids_sha256": None,
                    "donor_rationale_token_ids_sha256": None,
                    "recipient_token_length": None,
                    "donor_token_length": None,
                    "absolute_token_length_difference": None,
                    "active": False,
                    "inactive_reason": "no_label_consistent_reference",
                    "cycle_id": "",
                    "solver_name": "not_applicable",
                    "solver_version": "",
                }
            )
    return output


def _counter_delta(left: Counter[str], right: Counter[str]) -> tuple[int, int]:
    keys = set(left) | set(right)
    return (
        sum(abs(left[key] - right[key]) for key in keys),
        sum(left[key] != right[key] for key in keys),
    )


def _frequency_report(
    donor_rows: list[dict[str, Any]],
    *,
    from_epoch_index: int = 0,
    upto_epoch_index: int,
) -> dict[str, Any]:
    active = [
        row
        for row in donor_rows
        if row["active"]
        and from_epoch_index <= int(row["epoch_index"]) <= upto_epoch_index
    ]
    recipient_bytes = Counter(
        str(row["recipient_reason_bytes_sha256"]) for row in active
    )
    donor_bytes = Counter(str(row["donor_reason_bytes_sha256"]) for row in active)
    recipient_tokens = Counter(
        str(row["recipient_rationale_token_ids_sha256"]) for row in active
    )
    donor_tokens = Counter(
        str(row["donor_rationale_token_ids_sha256"]) for row in active
    )
    bytes_l1, bytes_different = _counter_delta(recipient_bytes, donor_bytes)
    tokens_l1, tokens_different = _counter_delta(recipient_tokens, donor_tokens)
    return {
        "from_epoch_number": from_epoch_index + 1,
        "checkpoint_epoch_number": upto_epoch_index + 1,
        "active_rationale_events_per_arm": len(active),
        "rationale_bytes_counter_identical": recipient_bytes == donor_bytes,
        "rationale_bytes_frequency_delta_l1": bytes_l1,
        "rationale_bytes_hashes_with_different_frequency": bytes_different,
        "rationale_token_ids_counter_identical": recipient_tokens == donor_tokens,
        "rationale_token_ids_frequency_delta_l1": tokens_l1,
        "rationale_token_hashes_with_different_frequency": tokens_different,
        "recipient_supervised_tokens": sum(
            int(row["recipient_token_length"]) for row in active
        ),
        "donor_supervised_tokens": sum(
            int(row["donor_token_length"]) for row in active
        ),
    }


def _expected_cycle_ids(donor_rows: list[dict[str, Any]]) -> dict[str, str]:
    active_by_event = {
        str(row["recipient_event_id"]): row for row in donor_rows if row["active"]
    }
    remaining = set(active_by_event)
    expected: dict[str, str] = {}
    while remaining:
        start = min(remaining)
        cycle = []
        current = start
        while current not in cycle:
            if current not in remaining:
                raise AssertionError("active event mapping does not close into disjoint cycles")
            cycle.append(current)
            current = str(active_by_event[current]["donor_event_id"])
        if current != start:
            raise AssertionError("active event mapping enters a previously visited cycle")
        canonical_start = min(range(len(cycle)), key=cycle.__getitem__)
        canonical = cycle[canonical_start:] + cycle[:canonical_start]
        cycle_id = "cycle-" + hashlib.sha256(
            "\n".join(canonical).encode("utf-8")
        ).hexdigest()[:16]
        for event_id in cycle:
            expected[event_id] = cycle_id
            remaining.remove(event_id)
    return expected


def validate_event_map(
    schedule_events: list[dict[str, Any]],
    donor_rows: list[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    schedule_by_event = {str(row["event_id"]): row for row in schedule_events}
    map_by_event = {str(row["recipient_event_id"]): row for row in donor_rows}
    if len(schedule_by_event) != len(schedule_events):
        raise ValueError("duplicate base event ID")
    if len(map_by_event) != len(donor_rows):
        raise ValueError("duplicate recipient event ID")
    if set(schedule_by_event) != set(map_by_event):
        raise ValueError("event donor map does not exactly cover base schedule")
    if len(donor_rows) != len(schedule_events):
        raise ValueError("event donor map row count differs from base schedule")

    active = [row for row in donor_rows if row["active"]]
    inactive = [row for row in donor_rows if not row["active"]]
    eligible = [
        row
        for row in donor_rows
        if schedule_by_event[str(row["recipient_event_id"])]["rationale_available"]
    ]
    active_recipient_ids = {str(row["recipient_event_id"]) for row in active}
    active_donor_ids = {str(row["donor_event_id"]) for row in active}
    expected_cycle_ids = _expected_cycle_ids(donor_rows)
    checks = {
        "one_map_row_per_base_event": len(map_by_event)
        == len(schedule_by_event)
        == len(donor_rows),
        "base_event_set_exact": set(map_by_event) == set(schedule_by_event),
        "base_event_order_unchanged": [
            row["recipient_event_id"] for row in donor_rows
        ]
        == [row["event_id"] for row in schedule_events],
        "seed_exact": all(int(row["seed"]) == seed for row in donor_rows),
        "recipient_metadata_matches_base_schedule": all(
            (
                row["seed"],
                row["epoch_index"],
                row["epoch_number"],
                row["row_position"],
                row["recipient_record_id"],
                row["recipient_reference_id"],
                row["recipient_normalized_qa_key"],
                row["label_5"],
                row["metric_id"],
                row["language"],
                row["recipient_reason_bytes_sha256"],
            )
            == (
                schedule_by_event[str(row["recipient_event_id"])]["seed"],
                schedule_by_event[str(row["recipient_event_id"])]["epoch_index"],
                schedule_by_event[str(row["recipient_event_id"])]["epoch_number"],
                schedule_by_event[str(row["recipient_event_id"])]["row_position"],
                schedule_by_event[str(row["recipient_event_id"])]["record_id"],
                schedule_by_event[str(row["recipient_event_id"])][
                    "selected_reference_id"
                ],
                schedule_by_event[str(row["recipient_event_id"])][
                    "normalized_qa_key"
                ],
                schedule_by_event[str(row["recipient_event_id"])]["label_5"],
                schedule_by_event[str(row["recipient_event_id"])]["metric_id"],
                schedule_by_event[str(row["recipient_event_id"])]["language"],
                schedule_by_event[str(row["recipient_event_id"])][
                    "selected_reason_bytes_sha256"
                ],
            )
            for row in donor_rows
        ),
        "active_donor_events_unique": len(active_donor_ids) == len(active),
        "active_subset_permutation": active_recipient_ids == active_donor_ids,
        "no_fixed_points": all(
            row["recipient_event_id"] != row["donor_event_id"] for row in active
        ),
        "no_same_record_donors": all(
            row["recipient_record_id"] != row["donor_record_id"] for row in active
        ),
        "no_same_content_donors": all(
            row["recipient_normalized_qa_key"]
            != row["donor_normalized_qa_key"]
            for row in active
        ),
        "structured_strata_preserved": all(
            (
                row["seed"],
                row["epoch_index"],
                row["label_5"],
                row["metric_id"],
                row["language"],
            )
            == (
                map_by_event[str(row["donor_event_id"])]["seed"],
                map_by_event[str(row["donor_event_id"])]["epoch_index"],
                map_by_event[str(row["donor_event_id"])]["label_5"],
                map_by_event[str(row["donor_event_id"])]["metric_id"],
                map_by_event[str(row["donor_event_id"])]["language"],
            )
            for row in active
        ),
        "active_rows_have_cycle_ids": all(row["cycle_id"] for row in active),
        "cycle_ids_match_event_graph": all(
            row["cycle_id"]
            == expected_cycle_ids[str(row["recipient_event_id"])]
            for row in active
        ),
        "donor_metadata_matches_donor_event": all(
            (
                row["donor_record_id"],
                row["donor_reference_id"],
                row["donor_normalized_qa_key"],
                row["donor_reason_bytes_sha256"],
                row["donor_rationale_token_ids_sha256"],
                row["donor_token_length"],
            )
            == (
                map_by_event[str(row["donor_event_id"])]["recipient_record_id"],
                map_by_event[str(row["donor_event_id"])]["recipient_reference_id"],
                map_by_event[str(row["donor_event_id"])][
                    "recipient_normalized_qa_key"
                ],
                map_by_event[str(row["donor_event_id"])][
                    "recipient_reason_bytes_sha256"
                ],
                map_by_event[str(row["donor_event_id"])][
                    "recipient_rationale_token_ids_sha256"
                ],
                map_by_event[str(row["donor_event_id"])]["recipient_token_length"],
            )
            for row in active
        ),
        "inactive_rows_have_no_donor": all(
            row["donor_event_id"] is None
            and row["donor_record_id"] is None
            and row["donor_reference_id"] is None
            for row in inactive
        ),
        "score_only_base_events_inactive": all(
            not map_by_event[event_id]["active"]
            and map_by_event[event_id]["inactive_reason"]
            == "no_label_consistent_reference"
            for event_id, event in schedule_by_event.items()
            if not event["rationale_available"]
        ),
    }
    prefix_reports = [
        _frequency_report(donor_rows, upto_epoch_index=epoch_index)
        for epoch_index in range(FORMAL_EPOCHS)
    ]
    epoch_reports = [
        _frequency_report(
            donor_rows,
            from_epoch_index=epoch_index,
            upto_epoch_index=epoch_index,
        )
        for epoch_index in range(FORMAL_EPOCHS)
    ]
    active_by_group: dict[
        tuple[int, int, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in active:
        active_by_group[
            (
                int(row["epoch_index"]),
                int(row["label_5"]),
                str(row["metric_id"]),
                str(row["language"]),
            )
        ].append(row)
    group_frequency_equal = all(
        Counter(str(row["recipient_reason_bytes_sha256"]) for row in rows)
        == Counter(str(row["donor_reason_bytes_sha256"]) for row in rows)
        and Counter(
            str(row["recipient_rationale_token_ids_sha256"]) for row in rows
        )
        == Counter(str(row["donor_rationale_token_ids_sha256"]) for row in rows)
        and sum(int(row["recipient_token_length"]) for row in rows)
        == sum(int(row["donor_token_length"]) for row in rows)
        for rows in active_by_group.values()
    )
    checks.update(
        {
            "all_structured_group_counters_and_tokens_identical": (
                group_frequency_equal
            ),
            "all_individual_epoch_bytes_counters_identical": all(
                report["rationale_bytes_counter_identical"]
                for report in epoch_reports
            ),
            "all_individual_epoch_token_counters_identical": all(
                report["rationale_token_ids_counter_identical"]
                for report in epoch_reports
            ),
            "all_individual_epoch_supervised_tokens_identical": all(
                report["recipient_supervised_tokens"]
                == report["donor_supervised_tokens"]
                for report in epoch_reports
            ),
            "all_checkpoint_prefix_bytes_counters_identical": all(
                report["rationale_bytes_counter_identical"]
                for report in prefix_reports
            ),
            "all_checkpoint_prefix_token_counters_identical": all(
                report["rationale_token_ids_counter_identical"]
                for report in prefix_reports
            ),
            "all_checkpoint_prefix_supervised_tokens_identical": all(
                report["recipient_supervised_tokens"]
                == report["donor_supervised_tokens"]
                for report in prefix_reports
            ),
        }
    )
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"event donor map checks failed: {failed}")

    active_length_differences = [
        int(row["absolute_token_length_difference"]) for row in active
    ]
    inactive_eligible = [row for row in eligible if not row["active"]]
    strata_counts = Counter(
        (
            row["epoch_index"],
            row["label_5"],
            row["metric_id"],
            row["language"],
        )
        for row in eligible
    )
    inactive_strata = {
        (
            row["epoch_index"],
            row["label_5"],
            row["metric_id"],
            row["language"],
        )
        for row in inactive_eligible
    }
    eligible_by_label = Counter(int(row["label_5"]) for row in eligible)
    active_by_label = Counter(int(row["label_5"]) for row in active)
    inactive_by_stratum = Counter(
        (
            int(row["epoch_number"]),
            int(row["label_5"]),
            str(row["metric_id"]),
            str(row["language"]),
        )
        for row in inactive_eligible
    )
    return {
        "checks": checks,
        "counts": {
            "base_row_events": len(donor_rows),
            "rationale_eligible_events": len(eligible),
            "score_only_events": len(donor_rows) - len(eligible),
            "active_rationale_events": len(active),
            "inactive_eligible_rationale_events": len(inactive_eligible),
            "active_event_coverage": len(active) / len(eligible) if eligible else 0.0,
            "event_strata": len(strata_counts),
            "event_strata_with_deactivation": len(inactive_strata),
            "max_stratum_events": max(strata_counts.values(), default=0),
            "max_absolute_token_length_difference": max(
                active_length_differences, default=0
            ),
            "mean_absolute_token_length_difference": (
                sum(active_length_differences) / len(active_length_differences)
                if active_length_differences
                else 0.0
            ),
        },
        "coverage_by_label": [
            {
                "label_5": label,
                "eligible_rationale_events": eligible_by_label[label],
                "active_rationale_events": active_by_label[label],
                "inactive_eligible_rationale_events": (
                    eligible_by_label[label] - active_by_label[label]
                ),
                "active_event_coverage": (
                    active_by_label[label] / eligible_by_label[label]
                    if eligible_by_label[label]
                    else 0.0
                ),
            }
            for label in sorted(eligible_by_label)
        ],
        "inactive_by_stratum": [
            {
                "epoch_number": key[0],
                "label_5": key[1],
                "metric_id": key[2],
                "language": key[3],
                "inactive_eligible_rationale_events": count,
            }
            for key, count in sorted(inactive_by_stratum.items())
        ],
        "individual_epoch_frequency": epoch_reports,
        "checkpoint_prefix_frequency": prefix_reports,
    }


def character_feature(text: str) -> tuple[int, str]:
    codepoints = [ord(character) for character in text]
    return (
        len(text),
        hashlib.sha256(
            json.dumps(codepoints, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    )


def tokenizer_feature_function(
    tokenizer_path: Path,
    lock_spec: dict[str, Any],
) -> tuple[RationaleFeatureFn, dict[str, Any]]:
    length_fn, tokenizer_lock = tokenizer_length_function(tokenizer_path, lock_spec)
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("formal tokenizer mode requires transformers") from error
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        trust_remote_code=True,
        local_files_only=True,
    )

    def feature_fn(text: str) -> tuple[int, str]:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        length = len(token_ids)
        if length != length_fn(text):
            raise AssertionError("tokenizer length functions disagree")
        return (
            length,
            hashlib.sha256(
                json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        )

    return feature_fn, tokenizer_lock


def _schedule_path(output_dir: Path, seed: int) -> Path:
    return output_dir / "data" / f"base_event_schedule_seed{seed}.jsonl"


def _map_path(output_dir: Path, seed: int, *, diagnostic: bool) -> Path:
    suffix = "_character_diagnostic" if diagnostic else ""
    return (
        output_dir
        / "data"
        / f"r2_event_donor_map_seed{seed}{suffix}.jsonl"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--length-mode",
        choices=("character-diagnostic", "tokenizer"),
        default="character-diagnostic",
    )
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument(
        "--tokenizer-lock-spec",
        type=Path,
        default=DEFAULT_TOKENIZER_LOCK_SPEC,
    )
    parser.add_argument("--oracle-report", type=Path, default=DEFAULT_ORACLE_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reject_eval_path(args.references)
    if not args.references.exists():
        raise FileNotFoundError(args.references)
    reference_rows = read_jsonl(args.references, protect_split=True)

    if args.length_mode == "tokenizer":
        if args.tokenizer_path is None:
            raise ValueError("--tokenizer-path is required for formal tokenizer mode")
        for path in (args.tokenizer_lock_spec, args.oracle_report):
            if not path.exists():
                raise FileNotFoundError(path)
        oracle_report = json.loads(args.oracle_report.read_text(encoding="utf-8"))
        if oracle_report.get("status") != "R2_EVENT_MATCHER_ORACLE_PASS":
            raise ValueError("formal mode requires R2_EVENT_MATCHER_ORACLE_PASS")
        if oracle_report.get("event_matcher_source_sha256") != file_sha256(
            Path(__file__)
        ):
            raise ValueError("Oracle does not cover current event matcher source")
        if oracle_report.get("core_matcher_source_sha256") != file_sha256(
            CORE_MATCHER_SOURCE
        ):
            raise ValueError("Oracle does not cover current core matcher source")
        lock_spec = json.loads(args.tokenizer_lock_spec.read_text(encoding="utf-8"))
        feature_fn, tokenizer_lock = tokenizer_feature_function(
            args.tokenizer_path,
            lock_spec,
        )
        diagnostic = False
        status_name = "R2_EVENT_DONOR_MAP_CANDIDATE_READY"
    else:
        feature_fn = character_feature
        tokenizer_lock = {
            "mode": "character_length_only",
            "formal_training_allowed": False,
        }
        diagnostic = True
        status_name = "R2_EVENT_DONOR_DIAGNOSTIC_ONLY"

    aggregate = []
    map_hashes: dict[str, str] = {}
    for seed in FORMAL_SEEDS:
        schedule_path = _schedule_path(args.out_dir, seed)
        reject_eval_path(schedule_path)
        if not schedule_path.exists():
            raise FileNotFoundError(schedule_path)
        schedule_events = read_jsonl(schedule_path, protect_split=True)
        donor_rows = build_event_donor_map(
            reference_rows,
            schedule_events,
            seed=seed,
            feature_fn=feature_fn,
        )
        validation = validate_event_map(schedule_events, donor_rows, seed=seed)
        map_path = _map_path(args.out_dir, seed, diagnostic=diagnostic)
        write_jsonl(map_path, donor_rows)
        map_hash = file_sha256(map_path)
        map_hashes[f"seed{seed}"] = map_hash
        seed_report = {
            "status": status_name,
            "seed": seed,
            "length_mode": args.length_mode,
            "base_schedule_sha256": file_sha256(schedule_path),
            "event_donor_map_sha256": map_hash,
            "reference_source_sha256": file_sha256(args.references),
            "event_matcher_source_sha256": file_sha256(Path(__file__)),
            "core_matcher_source_sha256": file_sha256(CORE_MATCHER_SOURCE),
            "tokenizer_lock": tokenizer_lock,
            "manifest_freeze_allowed": False,
            "formal_training_allowed": False,
            "dev_accessed": False,
            "test_accessed": False,
            "training_used": False,
            **validation,
        }
        aggregate.append(seed_report)
        report_name = (
            f"r2_event_donor_match_report_seed{seed}.json"
            if not diagnostic
            else f"r2_event_donor_match_diagnostic_seed{seed}.json"
        )
        write_json(args.out_dir / "audit" / report_name, seed_report)

    if not diagnostic:
        write_json(
            args.out_dir / "protocol/r2_event_donor_map_candidate_lock.json",
            {
                "status": status_name,
                "formal_seeds": list(FORMAL_SEEDS),
                "epochs": FORMAL_EPOCHS,
                "reference_source_sha256": file_sha256(args.references),
                "event_matcher_source_sha256": file_sha256(Path(__file__)),
                "core_matcher_source_sha256": file_sha256(CORE_MATCHER_SOURCE),
                "oracle_report_sha256": file_sha256(args.oracle_report),
                "tokenizer_lock_sha256": tokenizer_lock["tokenizer_lock_sha256"],
                "event_donor_map_sha256_by_seed": map_hashes,
                "candidate_event_mask_build_allowed": True,
                "manifest_freeze_allowed": False,
                "formal_training_allowed": False,
                "dev_accessed": False,
                "test_accessed": False,
                "training_used": False,
                "runtime": {
                    "python": platform.python_version(),
                    "scipy": scipy.__version__,
                },
            },
        )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
