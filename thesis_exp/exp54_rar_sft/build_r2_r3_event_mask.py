"""Build byte-identical candidate R2/R3 rationale-active event masks."""

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
from thesis_exp.exp54_rar_sft.build_r2_event_donor_map import validate_event_map
from thesis_exp.exp54_rar_sft.reference_schedule import FORMAL_EPOCHS, FORMAL_SEEDS


DEFAULT_OUTPUT = REPO_ROOT / "thesis_exp/outputs/exp54_rar_sft/rar_v2"
DEFAULT_DONOR_LOCK = (
    DEFAULT_OUTPUT / "protocol/r2_event_donor_map_candidate_lock.json"
)


def build_event_mask_rows(
    schedule_events: list[dict[str, Any]],
    donor_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    donor_by_event = {
        str(row["recipient_event_id"]): row for row in donor_rows
    }
    if len(donor_by_event) != len(donor_rows):
        raise ValueError("duplicate recipient event in donor map")
    schedule_ids = {str(event["event_id"]) for event in schedule_events}
    if set(donor_by_event) != schedule_ids:
        raise ValueError("event donor map and base schedule sets differ")

    output = []
    for event in schedule_events:
        event_id = str(event["event_id"])
        donor = donor_by_event[event_id]
        active = bool(donor["active"])
        output.append(
            {
                "base_event_id": event_id,
                "seed": int(event["seed"]),
                "epoch_index": int(event["epoch_index"]),
                "epoch_number": int(event["epoch_number"]),
                "row_position": int(event["row_position"]),
                "record_id": str(event["record_id"]),
                "base_selected_reference_id": event["selected_reference_id"],
                "rationale_active": active,
                "inactive_reason": "" if active else str(donor["inactive_reason"]),
            }
        )
    return output


def _vector_hash(values: list[Any]) -> str:
    payload = json.dumps(
        values,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def summarize_mask(mask_rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_epoch = []
    cumulative_vector: list[bool] = []
    for epoch_index in range(FORMAL_EPOCHS):
        epoch_rows = [
            row for row in mask_rows if int(row["epoch_index"]) == epoch_index
        ]
        vector = [bool(row["rationale_active"]) for row in epoch_rows]
        cumulative_vector.extend(vector)
        per_epoch.append(
            {
                "epoch_number": epoch_index + 1,
                "row_events": len(epoch_rows),
                "active_rationale_events": sum(vector),
                "inactive_rationale_events": len(vector) - sum(vector),
                "active_vector_sha256": _vector_hash(vector),
                "cumulative_active_vector_sha256": _vector_hash(cumulative_vector),
            }
        )
    active = sum(bool(row["rationale_active"]) for row in mask_rows)
    return {
        "row_events": len(mask_rows),
        "active_rationale_events": active,
        "inactive_rationale_events": len(mask_rows) - active,
        "full_active_vector_sha256": _vector_hash(
            [bool(row["rationale_active"]) for row in mask_rows]
        ),
        "per_epoch": per_epoch,
    }


def _schedule_path(output_dir: Path, seed: int) -> Path:
    return output_dir / "data" / f"base_event_schedule_seed{seed}.jsonl"


def _donor_map_path(output_dir: Path, seed: int) -> Path:
    return output_dir / "data" / f"r2_event_donor_map_seed{seed}.jsonl"


def _donor_report_path(output_dir: Path, seed: int) -> Path:
    return output_dir / "audit" / f"r2_event_donor_match_report_seed{seed}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--donor-lock", type=Path, default=DEFAULT_DONOR_LOCK)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reject_eval_path(args.donor_lock)
    if not args.donor_lock.exists():
        raise FileNotFoundError(args.donor_lock)
    donor_lock = json.loads(args.donor_lock.read_text(encoding="utf-8"))
    if donor_lock.get("status") != "R2_EVENT_DONOR_MAP_CANDIDATE_READY":
        raise ValueError("event mask requires candidate event donor maps")
    if donor_lock.get("manifest_freeze_allowed"):
        raise ValueError("candidate donor lock must not authorize manifest freeze")

    seed_reports = []
    mask_hashes: dict[str, dict[str, str]] = {}
    for seed in FORMAL_SEEDS:
        schedule_path = _schedule_path(args.out_dir, seed)
        donor_map_path = _donor_map_path(args.out_dir, seed)
        donor_report_path = _donor_report_path(args.out_dir, seed)
        for path in (schedule_path, donor_map_path, donor_report_path):
            reject_eval_path(path)
            if not path.exists():
                raise FileNotFoundError(path)
        expected_map_hash = donor_lock["event_donor_map_sha256_by_seed"][
            f"seed{seed}"
        ]
        if file_sha256(donor_map_path) != expected_map_hash:
            raise ValueError(f"seed {seed}: event donor map differs from lock")

        schedule_events = read_jsonl(schedule_path, protect_split=True)
        donor_rows = read_jsonl(donor_map_path, protect_split=True)
        donor_report = json.loads(donor_report_path.read_text(encoding="utf-8"))
        validation = validate_event_map(schedule_events, donor_rows, seed=seed)
        if not all(validation["checks"].values()):
            raise AssertionError(f"seed {seed}: event donor map validation failed")
        if donor_report["event_donor_map_sha256"] != expected_map_hash:
            raise ValueError(f"seed {seed}: donor report hash differs from lock")

        mask_rows = build_event_mask_rows(schedule_events, donor_rows)
        summary = summarize_mask(mask_rows)
        if (
            summary["active_rationale_events"]
            != donor_report["counts"]["active_rationale_events"]
        ):
            raise ValueError(f"seed {seed}: active mask count differs from donor report")

        r2_path = args.out_dir / "data" / f"r2_event_active_mask_seed{seed}.jsonl"
        r3_path = args.out_dir / "data" / f"r3_event_active_mask_seed{seed}.jsonl"
        write_jsonl(r2_path, mask_rows)
        write_jsonl(r3_path, mask_rows)
        r2_hash = file_sha256(r2_path)
        r3_hash = file_sha256(r3_path)
        checks = {
            "donor_map_checks_pass": all(validation["checks"].values()),
            "r2_r3_mask_bytes_identical": r2_path.read_bytes() == r3_path.read_bytes(),
            "r2_r3_mask_sha256_identical": r2_hash == r3_hash,
            "mask_covers_all_base_events": len(mask_rows) == len(schedule_events),
            "mask_does_not_encode_score_deactivation": all(
                "score_active" not in row
                and "score_mask" not in row
                and "score_loss_active" not in row
                for row in mask_rows
            ),
        }
        if not all(checks.values()):
            failed = [name for name, passed in checks.items() if not passed]
            raise AssertionError(f"seed {seed}: event mask checks failed: {failed}")
        seed_reports.append(
            {
                "seed": seed,
                "checks": checks,
                "counts": summary,
                "base_schedule_sha256": file_sha256(schedule_path),
                "event_donor_map_sha256": expected_map_hash,
                "r2_event_mask_sha256": r2_hash,
                "r3_event_mask_sha256": r3_hash,
            }
        )
        mask_hashes[f"seed{seed}"] = {"r2": r2_hash, "r3": r3_hash}

    report = {
        "status": "R2_R3_EVENT_MASK_CANDIDATE_READY",
        "formal_seeds": list(FORMAL_SEEDS),
        "epochs": FORMAL_EPOCHS,
        "seed_reports": seed_reports,
        "donor_lock_sha256": file_sha256(args.donor_lock),
        "candidate_manifest_build_allowed": False,
        "manifest_freeze_allowed": False,
        "smoke_training_allowed": False,
        "formal_training_allowed": False,
        "dev_accessed": False,
        "test_accessed": False,
        "training_used": False,
    }
    report_path = args.out_dir / "audit/r2_r3_event_mask_report.json"
    lock_path = args.out_dir / "protocol/r2_r3_event_mask_candidate_lock.json"
    write_json(report_path, report)
    write_json(
        lock_path,
        {
            "status": report["status"],
            "event_mask_sha256_by_seed": mask_hashes,
            "donor_lock_sha256": file_sha256(args.donor_lock),
            "event_mask_builder_source_sha256": file_sha256(Path(__file__)),
            "candidate_manifest_build_allowed": False,
            "manifest_freeze_allowed": False,
            "formal_training_allowed": False,
            "dev_accessed": False,
            "test_accessed": False,
            "training_used": False,
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
