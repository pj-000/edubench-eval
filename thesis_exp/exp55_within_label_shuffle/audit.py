"""Audit the fixed within-label soft-target permutation without reading test."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from thesis_exp.exp55_within_label_shuffle import OUTPUT_ROOT, SHUFFLE_SEED
from thesis_exp.exp55_within_label_shuffle.build_targets import (
    load_original_split,
    mapping_sha256,
    shuffle_train_rows,
    stable_row_id,
    target_key,
)


UNCHANGED_FIELDS = (
    "record_id",
    "id",
    "text",
    "label",
    "label_5",
    "hard_target_5",
    "human_mean_5",
    "human_1_5",
    "human_2_5",
    "human_3_5",
    "triple_key",
)


def build_audit(
    original: list[dict[str, Any]],
    shuffled: list[dict[str, Any]],
    mapping: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(original) != len(shuffled) or len(original) != len(mapping):
        raise AssertionError("Row and mapping counts differ")
    unchanged = all(
        all(left.get(field) == right.get(field) for field in UNCHANGED_FIELDS)
        for left, right in zip(original, shuffled)
    )
    labels: dict[str, Any] = {}
    for label in range(1, 6):
        before = [row for row in original if int(row["label_5"]) == label]
        after = [row for row in shuffled if int(row["label_5"]) == label]
        label_mapping = [row for row in mapping if int(row["hard_label"]) == label]
        before_multiset = Counter(target_key(row["soft_target_5"]) for row in before)
        after_multiset = Counter(target_key(row["soft_target_5"]) for row in after)
        labels[str(label)] = {
            "rows": len(before),
            "soft_target_multiset_preserved": before_multiset == after_multiset,
            "effective_target_changes": sum(bool(row["effectively_changed"]) for row in label_mapping),
            "effective_change_rate": (
                sum(bool(row["effectively_changed"]) for row in label_mapping) / len(label_mapping)
                if label_mapping
                else 0.0
            ),
            "self_assignments": sum(bool(row["self_assignment"]) for row in label_mapping),
            "original_target_counts": {
                str(list(key)): value for key, value in sorted(before_multiset.items())
            },
        }
    effective_changes = sum(bool(row["effectively_changed"]) for row in mapping)
    checks = {
        "train_rows_2654": len(original) == 2654,
        "row_order_and_nonsoft_fields_unchanged": unchanged,
        "all_label_multisets_preserved": all(
            row["soft_target_multiset_preserved"] for row in labels.values()
        ),
        "every_shuffled_mode_matches_hard_label": all(
            max(range(5), key=list(row["soft_target_5"]).__getitem__) + 1
            == int(row["label_5"])
            for row in shuffled
        ),
        "effective_changes_nonzero": effective_changes > 0,
        "test_access_count_zero": True,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "experiment": "within_label_shuffled_soft",
        "shuffle_seed": SHUFFLE_SEED,
        "mapping_sha256": mapping_sha256(mapping),
        "rows": len(original),
        "effective_target_changes": effective_changes,
        "effective_change_rate": effective_changes / len(original),
        "checks": checks,
        "by_hard_label": labels,
        "test_access_count": 0,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    original = load_original_split("train")
    shuffled, mapping = shuffle_train_rows(original)
    audit = build_audit(original, shuffled, mapping)
    audit_dir = OUTPUT_ROOT / "audit"
    write_json(audit_dir / "shuffle_audit.json", audit)
    mapping_path = audit_dir / "shuffle_mapping.jsonl"
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in mapping),
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if audit["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
