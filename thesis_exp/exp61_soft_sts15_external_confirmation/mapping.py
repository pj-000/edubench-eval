"""Train-only deterministic maximum-mismatch mapping for six-class targets."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from thesis_exp.exp61_soft_sts15_external_confirmation import LABELS
from thesis_exp.exp61_soft_sts15_external_confirmation.data import rows_contract_sha256


def theoretical_maximum_changes(counts: Counter[tuple[int, ...]]) -> int:
    total = sum(counts.values())
    if total == 0:
        return 0
    largest = max(counts.values())
    return total - max(0, 2 * largest - total)


def mapping_sha256(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda value: value["recipient_record_id"]):
        digest.update(
            (
                f"{row['hard_label']}\t{row['recipient_record_id']}\t"
                f"{row['donor_record_id']}\t{row['shuffled_target_fifths']}\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def build_maximum_mismatch_mapping(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if any(row.get("split") != "train" for row in rows):
        raise PermissionError("Exp61 mismatch mapping is train-only")
    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        record_id = str(row["record_id"])
        if record_id in seen:
            raise ValueError(f"duplicate record_id: {record_id}")
        seen.add(record_id)
        label = int(row["label"])
        if label not in LABELS:
            raise ValueError("hard label outside [0, 5]")
        fifths = tuple(int(value) for value in row["target_fifths"])
        if len(fifths) != 6 or sum(fifths) != 5 or min(fifths) < 0:
            raise ValueError("invalid six-class target fifths")
        by_label[label].append(row)

    mapping: list[dict[str, Any]] = []
    by_label_audit: dict[str, Any] = {}
    total_changed = 0
    for label in sorted(by_label):
        ordered = sorted(
            by_label[label],
            key=lambda row: (tuple(row["target_fifths"]), str(row["record_id"])),
        )
        counts = Counter(tuple(row["target_fifths"]) for row in ordered)
        shift = max(counts.values())
        donors = ordered[shift:] + ordered[:shift]
        changed = 0
        for recipient, donor in zip(ordered, donors):
            original = tuple(recipient["target_fifths"])
            shuffled = tuple(donor["target_fifths"])
            is_changed = original != shuffled
            changed += int(is_changed)
            mapping.append(
                {
                    "hard_label": label,
                    "recipient_record_id": str(recipient["record_id"]),
                    "donor_record_id": str(donor["record_id"]),
                    "original_target_fifths": list(original),
                    "shuffled_target_fifths": list(shuffled),
                    "effectively_changed": is_changed,
                    "self_assignment": recipient["record_id"] == donor["record_id"],
                }
            )
        maximum = theoretical_maximum_changes(counts)
        if changed != maximum:
            raise AssertionError(f"label {label}: mapping is not maximum mismatch")
        total_changed += changed
        by_label_audit[str(label)] = {
            "rows": len(ordered),
            "rotation": shift,
            "effective_target_changes": changed,
            "theoretical_maximum_changes": maximum,
            "target_state_count": len(counts),
        }

    recipient_ids = [row["recipient_record_id"] for row in mapping]
    donor_ids = [row["donor_record_id"] for row in mapping]
    donor_counts = Counter(donor_ids)
    checks = {
        "all_train_rows_mapped_once": len(mapping) == len(rows),
        "recipient_set_preserved": set(recipient_ids) == seen,
        "donor_set_preserved": set(donor_ids) == seen,
        "every_donor_used_once": len(donor_counts) == len(rows)
        and all(value == 1 for value in donor_counts.values()),
        "maximum_mismatch_in_every_label": True,
        "six_dimensional_target_multiset_preserved": True,
        "no_dev_or_test_information_used": True,
        "no_model_outcome_used": True,
    }
    if not all(checks.values()):
        raise AssertionError("Exp61 maximum-mismatch mapping contract failed")
    ordered_mapping = sorted(mapping, key=lambda value: value["recipient_record_id"])
    audit = {
        "status": "EXP61_TRAIN_MAXIMUM_MISMATCH_MAPPING_PASS",
        "algorithm": "within-label target blocks rotated by largest block size",
        "tie_break": "target_fifths then record_id lexicographic order",
        "rows": len(rows),
        "source_contract_sha256": rows_contract_sha256(rows),
        "mapping_sha256": mapping_sha256(ordered_mapping),
        "effective_target_changes": total_changed,
        "effective_change_rate": total_changed / len(rows),
        "by_hard_label": by_label_audit,
        "checks": checks,
        "allowed_splits": ["train"],
        "test_access_count": 0,
    }
    return ordered_mapping, audit


def mapping_target_lookup(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    return {
        str(row["recipient_record_id"]): [
            int(value) / 5.0 for value in row["shuffled_target_fifths"]
        ]
        for row in rows
    }


def write_mapping(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
