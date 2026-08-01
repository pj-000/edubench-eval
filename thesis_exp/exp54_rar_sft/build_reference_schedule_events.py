"""Build train-only candidate reference-schedule events for formal Exp54 seeds.

The row-level JSONL artifacts stay private under the ignored RAR-SFT data
directory. Public outputs contain only aggregate counts and cryptographic
hashes. This stage does not tokenize targets, match R2 donors, freeze training
manifests, inspect dev/test, or authorize training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.audit_rar0_alignment import (
    file_sha256,
    read_jsonl,
    reject_eval_path,
    write_json,
    write_jsonl,
)
from thesis_exp.exp54_rar_sft.reference_schedule import (
    EXPECTED_TRAIN_ROWS,
    FORMAL_EPOCHS,
    FORMAL_SEEDS,
    SCHEDULE_SCHEMA_VERSION,
    build_schedule_events,
    schedule_index,
    schedule_offset,
    validate_schedule_events,
    vector_sha256,
)


DEFAULT_REFERENCES = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/data/"
    "label_consistent_reference_sets.jsonl"
)
DEFAULT_REFERENCE_LOCK = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/protocol/"
    "reference_set_data_lock.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "thesis_exp/outputs/exp54_rar_sft/rar_v2"
SCHEMA_PATH = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/schemas/reference_schedule_event.schema.json"
)
SHARED_SCHEDULE_SOURCE = (
    REPO_ROOT / "thesis_exp/exp54_rar_sft/reference_schedule.py"
)


def _schedule_path(output_dir: Path, seed: int) -> Path:
    return output_dir / "data" / f"base_event_schedule_seed{seed}.jsonl"


def _source_lock_hash(reference_lock: dict[str, Any]) -> str:
    try:
        value = reference_lock["output_hashes"]["label_consistent_reference_sets"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "reference-set lock lacks label_consistent_reference_sets hash"
        ) from exc
    return str(value)


def schedule_probe() -> dict[str, Any]:
    """Freeze the exact SHA-256-prefix behavior on a public synthetic ID."""
    record_id = "sample-a"
    payload = f"42|{record_id}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return {
        "seed": 42,
        "record_id": record_id,
        "payload_utf8_hex": payload.hex(),
        "sha256": digest,
        "sha256_prefix_16": digest[:16],
        "offset_integer": schedule_offset(42, record_id),
        "indices_for_three_epochs_and_three_references": [
            schedule_index(42, record_id, epoch_index, 3)
            for epoch_index in range(3)
        ],
    }


def build_candidate_schedules(
    reference_rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    seeds: tuple[int, ...] = FORMAL_SEEDS,
    epochs: int = FORMAL_EPOCHS,
) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    if seeds != FORMAL_SEEDS:
        raise ValueError(f"formal seeds must be exactly {FORMAL_SEEDS}")
    if epochs != FORMAL_EPOCHS:
        raise ValueError(f"formal epochs must be exactly {FORMAL_EPOCHS}")
    if len(reference_rows) != EXPECTED_TRAIN_ROWS:
        raise ValueError(
            f"expected {EXPECTED_TRAIN_ROWS} train rows, found {len(reference_rows)}"
        )

    summaries: list[dict[str, Any]] = []
    schedules: dict[int, list[dict[str, Any]]] = {}
    for seed in seeds:
        events = build_schedule_events(reference_rows, seed=seed, epochs=epochs)
        summary = validate_schedule_events(
            reference_rows,
            events,
            seed=seed,
            epochs=epochs,
        )
        path = _schedule_path(output_dir, seed)
        write_jsonl(path, events)
        reloaded = read_jsonl(path, protect_split=True)
        if reloaded != events:
            raise ValueError(f"seed {seed}: serialized schedule does not round-trip")
        if not all(summary["checks"].values()):
            raise AssertionError(f"seed {seed}: schedule validation failed")
        summaries.append(
            {
                **summary,
                "private_schedule_path": str(path.relative_to(REPO_ROOT)),
                "private_schedule_sha256": file_sha256(path),
            }
        )
        schedules[seed] = events
    return summaries, schedules


def cross_seed_checks(
    reference_rows: list[dict[str, Any]],
    schedules: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    expected_row_vector = [
        str(row["record_id"])
        for _epoch_index in range(FORMAL_EPOCHS)
        for row in reference_rows
    ]
    row_vectors = {
        seed: [str(event["record_id"]) for event in events]
        for seed, events in schedules.items()
    }
    metadata_vectors = {
        seed: [
            (
                int(event["epoch_index"]),
                int(event["row_position"]),
                str(event["record_id"]),
                int(event["label_5"]),
                str(event["metric_id"]),
                str(event["language"]),
                str(event["normalized_qa_key"]),
            )
            for event in events
        ]
        for seed, events in schedules.items()
    }
    first_seed = FORMAL_SEEDS[0]
    checks = {
        "formal_seed_set_exact": set(schedules) == set(FORMAL_SEEDS),
        "row_event_vector_matches_train_order": all(
            vector == expected_row_vector for vector in row_vectors.values()
        ),
        "cross_seed_base_metadata_identical": all(
            vector == metadata_vectors[first_seed]
            for vector in metadata_vectors.values()
        ),
        "event_ids_disjoint_across_seeds": (
            sum(len({event["event_id"] for event in events}) for events in schedules.values())
            == len(
                {
                    event["event_id"]
                    for events in schedules.values()
                    for event in events
                }
            )
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"cross-seed schedule checks failed: {failed}")
    return {
        "checks": checks,
        "ordered_three_epoch_row_vector_sha256": vector_sha256(expected_row_vector),
        "cross_seed_metadata_vector_sha256": vector_sha256(
            metadata_vectors[first_seed]
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--reference-lock", type=Path, default=DEFAULT_REFERENCE_LOCK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.references, args.reference_lock):
        reject_eval_path(path)
        if not path.exists():
            raise FileNotFoundError(path)
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(SCHEMA_PATH)

    reference_lock = json.loads(args.reference_lock.read_text(encoding="utf-8"))
    if reference_lock.get("dev_accessed") or reference_lock.get("test_accessed"):
        raise PermissionError("reference-set source lock indicates evaluation access")
    actual_reference_hash = file_sha256(args.references)
    if actual_reference_hash != _source_lock_hash(reference_lock):
        raise ValueError("reference inventory differs from its frozen source lock")

    reference_rows = read_jsonl(args.references, protect_split=True)
    seed_summaries, schedules = build_candidate_schedules(
        reference_rows,
        output_dir=args.output_dir,
    )
    cross_seed = cross_seed_checks(reference_rows, schedules)

    source_hashes = {
        "shared_schedule_source_sha256": file_sha256(SHARED_SCHEDULE_SOURCE),
        "builder_source_sha256": file_sha256(Path(__file__)),
        "event_schema_sha256": file_sha256(SCHEMA_PATH),
    }
    schedule_file_hashes = {
        f"base_event_schedule_seed{summary['seed']}": summary[
            "private_schedule_sha256"
        ]
        for summary in seed_summaries
    }
    report = {
        "experiment": "Exp54 RAR-SFT event-level control",
        "stage": "shared train-only reference schedule",
        "status": "CANDIDATE_REFERENCE_SCHEDULE_READY",
        "schedule_schema_version": SCHEDULE_SCHEMA_VERSION,
        "formal_seeds": list(FORMAL_SEEDS),
        "epochs": FORMAL_EPOCHS,
        "expected_rows_per_epoch": EXPECTED_TRAIN_ROWS,
        "expected_row_events_per_seed": EXPECTED_TRAIN_ROWS * FORMAL_EPOCHS,
        "reference_source_sha256": actual_reference_hash,
        "reference_lock_sha256": file_sha256(args.reference_lock),
        "schedule_probe": schedule_probe(),
        "seed_summaries": seed_summaries,
        "cross_seed": cross_seed,
        "source_hashes": source_hashes,
        "private_schedule_file_hashes": schedule_file_hashes,
        "event_matching_allowed": True,
        "manifest_freeze_allowed": False,
        "smoke_training_allowed": False,
        "formal_training_allowed": False,
        "dev_accessed": False,
        "test_accessed": False,
        "training_used": False,
        "row_level_artifacts_public": False,
        "human_rationale_text_public": False,
    }
    if any(
        summary["row_events"] != EXPECTED_TRAIN_ROWS * FORMAL_EPOCHS
        for summary in seed_summaries
    ):
        raise AssertionError("formal event count mismatch")

    audit_path = (
        args.output_dir / "audit" / "reference_schedule_candidate_report.json"
    )
    lock_path = (
        args.output_dir / "protocol" / "reference_schedule_candidate_lock.json"
    )
    write_json(audit_path, report)
    candidate_lock = {
        "status": report["status"],
        "schedule_schema_version": SCHEDULE_SCHEMA_VERSION,
        "reference_source_sha256": actual_reference_hash,
        "source_hashes": source_hashes,
        "schedule_file_hashes": schedule_file_hashes,
        "ordered_three_epoch_row_vector_sha256": cross_seed[
            "ordered_three_epoch_row_vector_sha256"
        ],
        "manifest_freeze_allowed": False,
        "formal_training_allowed": False,
        "dev_accessed": False,
        "test_accessed": False,
        "training_used": False,
    }
    write_json(lock_path, candidate_lock)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
