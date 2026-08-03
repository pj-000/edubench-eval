"""Deterministic maximum-mismatch residual mapping for Exp60.

The mapping is constructed independently inside each hard-label stratum.  It
preserves the exact empirical soft-target multiset while maximizing the number
of recipients whose target vector changes.  No model output or development
metric is used.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from thesis_exp.exp57_cbrd.data_audit import (
    model_rows,
    row_target,
    stable_row_id,
    target_state,
    thirds,
    write_json,
)
from thesis_exp.exp60_geometry_matched_shuffle import (
    MAPPING_AUDIT_PATH,
    MAPPING_PATH,
)


def source_contract_sha256(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=stable_row_id):
        digest.update(
            (
                f"{stable_row_id(row)}\t{int(row['label_5'])}\t"
                f"{list(thirds(row_target(row)))}\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def mapping_sha256(mapping: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(mapping, key=lambda item: str(item["recipient_record_id"])):
        digest.update(
            (
                f"{row['hard_label']}\t{row['recipient_record_id']}\t"
                f"{row['donor_record_id']}\t{row['shuffled_target_thirds']}\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def theoretical_maximum_changes(target_counts: Counter[tuple[int, ...]]) -> int:
    """Maximum mismatches in a permutation of a categorical multiset."""

    total = sum(target_counts.values())
    if total == 0:
        return 0
    largest = max(target_counts.values())
    unavoidable_matches = max(0, 2 * largest - total)
    return total - unavoidable_matches


def build_maximum_mismatch_mapping(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        record_id = stable_row_id(row)
        if record_id in seen:
            raise ValueError(f"Duplicate record id: {record_id}")
        seen.add(record_id)
        label = int(row["label_5"])
        if label not in (1, 2, 3, 4, 5):
            raise ValueError(f"Hard label outside 1--5: {label}")
        by_label[label].append(row)

    mapping: list[dict[str, Any]] = []
    label_audits: dict[str, Any] = {}
    total_changed = 0
    total_self_assignments = 0
    transitions: Counter[str] = Counter()
    for label in sorted(by_label):
        ordered = sorted(
            by_label[label],
            key=lambda row: (thirds(row_target(row)), stable_row_id(row)),
        )
        counts = Counter(thirds(row_target(row)) for row in ordered)
        shift = max(counts.values())
        donors = ordered[shift:] + ordered[:shift]
        if Counter(thirds(row_target(row)) for row in donors) != counts:
            raise AssertionError(f"Target multiset changed for hard label {label}")
        changed = 0
        self_assignments = 0
        for recipient, donor in zip(ordered, donors):
            original = thirds(row_target(recipient))
            shuffled = thirds(row_target(donor))
            original_state = target_state(label, [value / 3.0 for value in original])
            shuffled_state = target_state(label, [value / 3.0 for value in shuffled])
            is_changed = original != shuffled
            is_self = stable_row_id(recipient) == stable_row_id(donor)
            changed += int(is_changed)
            self_assignments += int(is_self)
            transitions[f"{original_state}->{shuffled_state}"] += 1
            mapping.append(
                {
                    "hard_label": label,
                    "recipient_record_id": stable_row_id(recipient),
                    "donor_record_id": stable_row_id(donor),
                    "original_target_thirds": list(original),
                    "shuffled_target_thirds": list(shuffled),
                    "original_state": original_state,
                    "shuffled_state": shuffled_state,
                    "effectively_changed": is_changed,
                    "self_assignment": is_self,
                }
            )
        theoretical = theoretical_maximum_changes(counts)
        if changed != theoretical:
            raise AssertionError(
                f"Rotation failed maximum-mismatch proof for label {label}: "
                f"{changed} != {theoretical}"
            )
        total_changed += changed
        total_self_assignments += self_assignments
        label_audits[str(label)] = {
            "rows": len(ordered),
            "target_counts": {str(list(key)): value for key, value in sorted(counts.items())},
            "rotation": shift,
            "effective_target_changes": changed,
            "theoretical_maximum_changes": theoretical,
            "effective_change_rate": changed / len(ordered),
            "self_assignments": self_assignments,
        }

    source_by_id = {stable_row_id(row): row for row in rows}
    recipient_ids = [str(row["recipient_record_id"]) for row in mapping]
    donor_ids = [str(row["donor_record_id"]) for row in mapping]
    donor_counts = Counter(donor_ids)
    donor_targets_exact = all(
        list(thirds(row_target(source_by_id[str(item["donor_record_id"])])))
        == list(item["shuffled_target_thirds"])
        for item in mapping
    )
    hard_labels_match = all(
        int(source_by_id[str(item["recipient_record_id"])]["label_5"])
        == int(item["hard_label"])
        == int(source_by_id[str(item["donor_record_id"])]["label_5"])
        for item in mapping
    )
    audit = {
        "status": "EXP60_MAXIMUM_MISMATCH_MAPPING_PASS",
        "algorithm": "within-hard-label lexicographic target blocks rotated by the largest block size",
        "tie_break": "target_thirds then recipient record_id lexicographic order",
        "rows": len(rows),
        "source_contract_sha256": source_contract_sha256(rows),
        "mapping_sha256": mapping_sha256(mapping),
        "effective_target_changes": total_changed,
        "effective_change_rate": total_changed / len(rows),
        "self_assignments": total_self_assignments,
        "state_transition_counts": dict(sorted(transitions.items())),
        "by_hard_label": label_audits,
        "checks": {
            "all_rows_mapped_once": len(mapping) == len(rows),
            "recipient_id_set_equals_source_id_set": set(recipient_ids) == set(source_by_id),
            "donor_id_set_equals_recipient_id_set": set(donor_ids) == set(recipient_ids),
            "every_donor_used_exactly_once": all(count == 1 for count in donor_counts.values())
            and len(donor_counts) == len(rows),
            "donor_and_recipient_hard_labels_match": hard_labels_match,
            "shuffled_target_exactly_equals_donor_source_target": donor_targets_exact,
            "within_hard_label_target_multisets_preserved": True,
            "maximum_mismatch_achieved_in_every_label": True,
            "no_model_or_dev_information_used": True,
            "no_test_access": True,
        },
        "allowed_splits": ["train"],
        "test_access_count": 0,
    }
    return sorted(mapping, key=lambda row: str(row["recipient_record_id"])), audit


def mapping_target_lookup(mapping: list[dict[str, Any]]) -> dict[str, list[float]]:
    return {
        str(row["recipient_record_id"]): [
            int(value) / 3.0 for value in row["shuffled_target_thirds"]
        ]
        for row in mapping
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    rows = model_rows("train")
    mapping, audit = build_maximum_mismatch_mapping(rows)
    write_jsonl(MAPPING_PATH, mapping)
    write_json(MAPPING_AUDIT_PATH, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
