"""Deterministically permute HMSA soft targets within each hard-label class.

The control preserves the complete multiset of empirical human distributions
inside every hard-label class. It changes only which training example receives
which distribution. Development rows remain untouched and test access stays
forbidden by the inherited Exp51 loader.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Any, Iterable

from thesis_exp.exp55_within_label_shuffle import SHUFFLE_SEED


def target_key(target: Iterable[float]) -> tuple[int, ...]:
    """Represent thirds exactly enough for stable counting and hashing."""

    values = tuple(int(round(float(value) * 3)) for value in target)
    if len(values) != 5 or sum(values) != 3:
        raise ValueError(f"Invalid five-class three-rater target: {values}")
    return values


def stable_row_id(row: dict[str, Any]) -> str:
    value = row.get("record_id") or row.get("id")
    if not value:
        raise ValueError("Every shuffled row requires record_id or id")
    return str(value)


def donor_sort_key(row_id: str, label: int, shuffle_seed: int) -> str:
    payload = f"{shuffle_seed}\t{label}\t{row_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def shuffle_train_rows(
    rows: list[dict[str, Any]], *, shuffle_seed: int = SHUFFLE_SEED
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return copied rows and a stable recipient-to-donor mapping."""

    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen_ids: set[str] = set()
    for row in rows:
        row_id = stable_row_id(row)
        if row_id in seen_ids:
            raise ValueError(f"Duplicate row identifier: {row_id}")
        seen_ids.add(row_id)
        label = int(row["label_5"])
        if label not in (1, 2, 3, 4, 5):
            raise ValueError(f"Hard label outside 1--5: {label}")
        if max(range(5), key=list(row["soft_target_5"]).__getitem__) + 1 != label:
            raise ValueError(f"Original soft-target mode differs from hard label for {row_id}")
        by_label[label].append(row)

    shuffled_by_id: dict[str, dict[str, Any]] = {}
    mapping: list[dict[str, Any]] = []
    for label in sorted(by_label):
        recipients = sorted(by_label[label], key=stable_row_id)
        donors = sorted(
            by_label[label],
            key=lambda row: donor_sort_key(stable_row_id(row), label, shuffle_seed),
        )
        if len(recipients) != len(donors):
            raise AssertionError("Recipient and donor counts differ")
        original_multiset = Counter(target_key(row["soft_target_5"]) for row in recipients)
        assigned_multiset = Counter(target_key(row["soft_target_5"]) for row in donors)
        if original_multiset != assigned_multiset:
            raise AssertionError(f"Soft-target multiset changed for label {label}")

        for recipient, donor in zip(recipients, donors):
            recipient_id = stable_row_id(recipient)
            donor_id = stable_row_id(donor)
            original_key = target_key(recipient["soft_target_5"])
            shuffled_key = target_key(donor["soft_target_5"])
            copied = dict(recipient)
            copied["soft_target_5"] = [float(value) for value in donor["soft_target_5"]]
            copied["soft_target_source_record_id"] = donor_id
            copied["soft_target_shuffle_seed"] = shuffle_seed
            copied["soft_target_effectively_changed"] = original_key != shuffled_key
            if max(range(5), key=copied["soft_target_5"].__getitem__) + 1 != label:
                raise AssertionError(f"Shuffled soft-target mode differs from hard label for {recipient_id}")
            shuffled_by_id[recipient_id] = copied
            mapping.append(
                {
                    "hard_label": label,
                    "recipient_record_id": recipient_id,
                    "donor_record_id": donor_id,
                    "self_assignment": recipient_id == donor_id,
                    "original_target_thirds": list(original_key),
                    "shuffled_target_thirds": list(shuffled_key),
                    "effectively_changed": original_key != shuffled_key,
                }
            )

    shuffled = [shuffled_by_id[stable_row_id(row)] for row in rows]
    return shuffled, mapping


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


def load_original_split(split: str) -> list[dict[str, Any]]:
    # Delayed import keeps the deterministic shuffle helpers locally testable
    # while reusing the locked Exp51 loader on the training server.
    from thesis_exp.exp51_hmsa.build_targets import load_split as load_exp51_split

    return load_exp51_split(split)


def load_split(split: str) -> list[dict[str, Any]]:
    rows = load_original_split(split)
    if split == "train":
        return shuffle_train_rows(rows)[0]
    return rows
