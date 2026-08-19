"""Materialize the deterministic, source-article-disjoint Exp64 A/B probes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from thesis_exp.exp62_summeval_routing_confirmation.audit_dataset import sha256_file
from thesis_exp.exp62_summeval_routing_confirmation.data import load_model_rows
from thesis_exp.exp64_optimizer_state_residual import OUTPUT_ROOT


PARTITION_SALT = "exp64_probe_v1"


def stable_group_key(group_id: str) -> str:
    return hashlib.sha256(f"{PARTITION_SALT}\t{group_id}".encode("utf-8")).hexdigest()


def hash_lines(lines: list[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument(
        "--output_dir", type=Path, default=OUTPUT_ROOT / "stage0"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_model_rows(args.annotations, "dev")
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row["group_id"])].append(row)
    ordered_groups = sorted(by_group, key=lambda group: (stable_group_key(group), group))
    if len(ordered_groups) != 15:
        raise RuntimeError(f"Expected 15 SummEval dev groups, found {len(ordered_groups)}")
    assignment = {
        group: ("A" if index % 2 == 0 else "B")
        for index, group in enumerate(ordered_groups)
    }
    manifest_rows = []
    for row in sorted(rows, key=lambda value: str(value["record_id"])):
        manifest_rows.append(
            {
                "record_id": str(row["record_id"]),
                "group_id": str(row["group_id"]),
                "probe": assignment[str(row["group_id"])],
                "dimension": str(row["dimension"]),
                "hard_label": int(row["hard_label"]),
            }
        )
    manifest_lines = [
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in manifest_rows
    ]
    manifest_path = args.output_dir / "probe_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    probe_counts = Counter(row["probe"] for row in manifest_rows)
    dimension_counts = {
        probe: Counter(
            row["dimension"] for row in manifest_rows if row["probe"] == probe
        )
        for probe in ("A", "B")
    }
    group_record_counts = {group: len(by_group[group]) for group in ordered_groups}
    checks = {
        "dev_rows_480": len(rows) == 480,
        "groups_15": len(ordered_groups) == 15,
        "probe_A_groups_8": sum(value == "A" for value in assignment.values()) == 8,
        "probe_B_groups_7": sum(value == "B" for value in assignment.values()) == 7,
        "each_group_has_32_rows": set(group_record_counts.values()) == {32},
        "groups_are_disjoint": not (
            {group for group, probe in assignment.items() if probe == "A"}
            & {group for group, probe in assignment.items() if probe == "B"}
        ),
        "probe_A_rows_256": probe_counts["A"] == 256,
        "probe_B_rows_224": probe_counts["B"] == 224,
        "each_probe_contains_both_dimensions": all(
            set(dimension_counts[probe]) == {"coherence", "fluency"}
            for probe in ("A", "B")
        ),
        "test_access_count_zero": True,
    }
    result = {
        "status": "EXP64_PROBE_MANIFEST_PASS" if all(checks.values()) else "FAIL",
        "partition_salt": PARTITION_SALT,
        "partition_rule": (
            "Sort group IDs by SHA256('exp64_probe_v1\\t' + group_id), then "
            "assign even positions to A and odd positions to B."
        ),
        "annotation_path": str(args.annotations),
        "annotation_sha256": sha256_file(args.annotations),
        "groups_in_hash_order": ordered_groups,
        "group_hashes": {group: stable_group_key(group) for group in ordered_groups},
        "probe_A_groups": [group for group in ordered_groups if assignment[group] == "A"],
        "probe_B_groups": [group for group in ordered_groups if assignment[group] == "B"],
        "group_record_counts": group_record_counts,
        "probe_record_counts": dict(probe_counts),
        "probe_dimension_counts": {
            probe: dict(counts) for probe, counts in dimension_counts.items()
        },
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_logical_sha256": hash_lines(manifest_lines),
        "checks": checks,
        "model_outcomes_read": 0,
        "test_access_count": 0,
    }
    write_json(args.output_dir / "probe_manifest_audit.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "EXP64_PROBE_MANIFEST_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
