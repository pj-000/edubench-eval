"""Audit whether R2/R3 multi-reference schedules preserve rationale frequencies.

The frozen donor map is a strict permutation over active references. Training,
however, selects one reference per row and epoch. If a donor reference from a
two-reference row is assigned to a recipient position from a three-reference
row, the realized rationale frequencies can differ even though the inventory
multisets are identical. This train-only audit quantifies that interaction and
compares deterministic alternatives without authorizing a protocol change.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.audit_rar0_alignment import (
    file_sha256,
    read_jsonl,
    reject_eval_path,
    write_json,
)
from thesis_exp.exp54_rar_sft.build_r2_donor_map import (
    flatten_references,
    match_stratum,
)
from thesis_exp.exp54_rar_sft.reference_schedule import schedule_index


DEFAULT_REFERENCES = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/data/"
    "label_consistent_reference_sets.jsonl"
)
DEFAULT_DONOR_MAP = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/data/"
    "shuffled_rationale_donor_map.jsonl"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/audit/"
    "manifest_schedule_compatibility_report.json"
)
SEEDS = (42, 43, 44)


def selected_frequency(
    seed: int,
    sample_id: str,
    reference_id: str,
    references: list[dict[str, Any]],
    epochs: int,
) -> int:
    ordered = sorted(references, key=lambda row: row["rater_id"])
    return sum(
        ordered[schedule_index(seed, sample_id, epoch, len(ordered))]["reference_id"]
        == reference_id
        for epoch in range(epochs)
    )


def coverage_summary(
    reference_rows: list[dict[str, Any]],
    mapping: list[dict[str, Any]],
) -> dict[str, Any]:
    active = [row for row in mapping if row["active"]]
    source_samples_by_label: dict[int, set[str]] = defaultdict(set)
    active_samples_by_label: dict[int, set[str]] = defaultdict(set)
    for row in reference_rows:
        if row["references"]:
            source_samples_by_label[int(row["label_5"])].add(str(row["record_id"]))
    for row in active:
        active_samples_by_label[int(row["label_5"])].add(
            str(row["recipient_record_id"])
        )
    return {
        "source_references": len(mapping),
        "active_references": len(active),
        "inactive_references": len(mapping) - len(active),
        "active_coverage": len(active) / len(mapping) if mapping else 0.0,
        "sample_coverage_by_label": [
            {
                "label_5": label,
                "source_reason_samples": len(source_samples_by_label[label]),
                "active_reason_samples": len(active_samples_by_label[label]),
                "fully_inactive_samples": len(
                    source_samples_by_label[label] - active_samples_by_label[label]
                ),
            }
            for label in sorted(source_samples_by_label)
        ],
    }


def frequency_audit(
    reference_rows: list[dict[str, Any]],
    mapping: list[dict[str, Any]],
    *,
    epochs: int,
    seeds: tuple[int, ...] = SEEDS,
) -> dict[str, Any]:
    map_by_recipient = {
        str(row["recipient_reference_id"]): row for row in mapping
    }
    reason_hash_by_reference = {
        str(reference["reference_id"]): str(reference["clean_reason_sha256"])
        for row in reference_rows
        for reference in row["references"]
    }
    seed_reports = []
    for seed in seeds:
        r2_frequency: Counter[str] = Counter()
        r3_frequency: Counter[str] = Counter()
        active_events = 0
        for row in reference_rows:
            references = sorted(row["references"], key=lambda value: value["rater_id"])
            if not references:
                continue
            for epoch in range(epochs):
                reference = references[
                    schedule_index(seed, str(row["record_id"]), epoch, len(references))
                ]
                donor = map_by_recipient[str(reference["reference_id"])]
                if not donor["active"]:
                    continue
                r3_frequency[str(reference["clean_reason_sha256"])] += 1
                r2_frequency[
                    reason_hash_by_reference[str(donor["donor_reference_id"])]
                ] += 1
                active_events += 1
        all_hashes = set(r2_frequency) | set(r3_frequency)
        absolute_frequency_delta = sum(
            abs(r2_frequency[key] - r3_frequency[key]) for key in all_hashes
        )
        differing_reason_hashes = sum(
            r2_frequency[key] != r3_frequency[key] for key in all_hashes
        )
        seed_reports.append(
            {
                "seed": seed,
                "epochs": epochs,
                "active_rationale_events_per_arm": active_events,
                "rationale_frequency_multiset_identical": r2_frequency == r3_frequency,
                "absolute_frequency_delta_l1": absolute_frequency_delta,
                "reason_hashes_with_different_frequency": differing_reason_hashes,
            }
        )
    return {
        "epochs": epochs,
        "all_seeds_frequency_identical": all(
            row["rationale_frequency_multiset_identical"] for row in seed_reports
        ),
        "seed_reports": seed_reports,
    }


def build_candidate_map(
    reference_rows: list[dict[str, Any]],
    formal_mapping: list[dict[str, Any]],
    *,
    mode: str,
    epochs: int,
) -> list[dict[str, Any]]:
    if mode not in {"same_reference_count", "schedule_signature"}:
        raise ValueError(f"unsupported candidate mode: {mode}")
    items = flatten_references(reference_rows)
    rows_by_id = {str(row["record_id"]): row for row in reference_rows}
    reference_count_by_sample = {
        str(row["record_id"]): len(row["references"]) for row in reference_rows
    }
    token_length_by_reference = {
        str(row["recipient_reference_id"]): int(row["recipient_length"])
        for row in formal_mapping
    }
    length_by_reason = {
        str(item["reason"]): token_length_by_reference[str(item["reference_id"])]
        for item in items
    }

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        sample_id = str(item["record_id"])
        key: list[Any] = [
            int(item["label_5"]),
            str(item["metric_id"]),
            str(item["language"]),
            reference_count_by_sample[sample_id],
        ]
        if mode == "schedule_signature":
            row = rows_by_id[sample_id]
            key.extend(
                selected_frequency(
                    seed,
                    sample_id,
                    str(item["reference_id"]),
                    row["references"],
                    epochs,
                )
                for seed in SEEDS
            )
        grouped[tuple(key)].append(item)

    output: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda value: tuple(map(str, value))):
        output.extend(
            match_stratum(
                grouped[key],
                lambda text: length_by_reason[str(text)],
            )
        )
    return sorted(output, key=lambda row: row["recipient_reference_id"])


def cross_reference_count_summary(
    reference_rows: list[dict[str, Any]],
    mapping: list[dict[str, Any]],
) -> dict[str, Any]:
    count_by_sample = {
        str(row["record_id"]): len(row["references"]) for row in reference_rows
    }
    transitions = Counter(
        (
            count_by_sample[str(row["recipient_record_id"])],
            count_by_sample[str(row["donor_record_id"])],
        )
        for row in mapping
        if row["active"]
    )
    mismatch = sum(
        count
        for (recipient_count, donor_count), count in transitions.items()
        if recipient_count != donor_count
    )
    return {
        "active_edges_crossing_reference_count": mismatch,
        "active_edges_total": sum(transitions.values()),
        "transition_counts": [
            {
                "recipient_reference_count": key[0],
                "donor_reference_count": key[1],
                "active_edges": value,
            }
            for key, value in sorted(transitions.items())
        ],
    }


def run_audit(
    reference_rows: list[dict[str, Any]],
    formal_mapping: list[dict[str, Any]],
) -> dict[str, Any]:
    same_count_map = build_candidate_map(
        reference_rows,
        formal_mapping,
        mode="same_reference_count",
        epochs=6,
    )
    signature_map = build_candidate_map(
        reference_rows,
        formal_mapping,
        mode="schedule_signature",
        epochs=3,
    )
    current_frequency = frequency_audit(
        reference_rows,
        formal_mapping,
        epochs=3,
    )
    same_count_frequency = frequency_audit(
        reference_rows,
        same_count_map,
        epochs=6,
    )
    signature_frequency = frequency_audit(
        reference_rows,
        signature_map,
        epochs=3,
    )
    current_compatible = current_frequency["all_seeds_frequency_identical"]
    return {
        "status": (
            "TRAINING_MANIFEST_SCHEDULE_COMPATIBLE"
            if current_compatible
            else "TRAINING_MANIFEST_SCHEDULE_BLOCKED"
        ),
        "problem": (
            "Reference-level permutation does not preserve realized rationale frequencies "
            "under the frozen one-reference-per-row three-epoch schedule."
        ),
        "current_contract": {
            "grouping": "label_5 × metric_id × language",
            "epochs": 3,
            "coverage": coverage_summary(reference_rows, formal_mapping),
            "cross_reference_count": cross_reference_count_summary(
                reference_rows,
                formal_mapping,
            ),
            "frequency": current_frequency,
        },
        "candidate_same_reference_count_six_epochs": {
            "protocol_change": (
                "Add reference_count to donor strata and use six epochs, the LCM of "
                "two- and three-reference schedules."
            ),
            "coverage": coverage_summary(reference_rows, same_count_map),
            "frequency": same_count_frequency,
        },
        "candidate_schedule_signature_three_epochs": {
            "protocol_change": (
                "Add reference_count and the per-seed three-epoch selection-frequency "
                "signature to donor strata."
            ),
            "coverage": coverage_summary(reference_rows, signature_map),
            "frequency": signature_frequency,
        },
        "manifest_freeze_allowed": current_compatible,
        "formal_training_allowed": False,
        "external_review_required": not current_compatible,
        "dev_accessed": False,
        "test_accessed": False,
        "training_used": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--donor-map", type=Path, default=DEFAULT_DONOR_MAP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.references, args.donor_map):
        reject_eval_path(path)
        if not path.exists():
            raise FileNotFoundError(path)
    reference_rows = read_jsonl(args.references, protect_split=True)
    formal_mapping = read_jsonl(args.donor_map, protect_split=True)
    report = {
        **run_audit(reference_rows, formal_mapping),
        "reference_source_sha256": file_sha256(args.references),
        "formal_donor_map_sha256": file_sha256(args.donor_map),
        "audit_source_sha256": file_sha256(Path(__file__)),
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
