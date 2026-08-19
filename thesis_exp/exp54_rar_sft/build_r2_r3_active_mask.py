"""Freeze the identical R2/R3 rationale-active mask from the formal donor map."""

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
    write_jsonl,
)
from thesis_exp.exp54_rar_sft.build_r2_donor_map import flatten_references, validate_map


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
DEFAULT_DONOR_REPORT = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/audit/"
    "r2_donor_match_report.json"
)
DEFAULT_DONOR_LOCK = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/protocol/"
    "r2_donor_map_lock.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "thesis_exp/outputs/exp54_rar_sft/rar_v2"


def build_mask_rows(
    reference_rows: list[dict[str, Any]],
    donor_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    items = flatten_references(reference_rows)
    donor_by_recipient = {
        str(row["recipient_reference_id"]): row for row in donor_rows
    }
    if len(donor_by_recipient) != len(donor_rows):
        raise ValueError("duplicate recipient_reference_id in formal donor map")
    if set(donor_by_recipient) != {item["reference_id"] for item in items}:
        raise ValueError("formal donor map and reference inventory do not cover the same references")

    output = []
    for item in sorted(items, key=lambda value: value["reference_id"]):
        donor = donor_by_recipient[item["reference_id"]]
        active = bool(donor["active"])
        output.append(
            {
                "record_id": item["record_id"],
                "reference_id": item["reference_id"],
                "label_5": item["label_5"],
                "metric_id": item["metric_id"],
                "language": item["language"],
                "rationale_active": active,
                "inactive_reason": "" if active else str(donor["inactive_reason"]),
            }
        )
    return output


def summarize_mask(mask_rows: list[dict[str, Any]]) -> dict[str, Any]:
    active = [row for row in mask_rows if row["rationale_active"]]
    inactive = [row for row in mask_rows if not row["rationale_active"]]
    source_samples_by_label: dict[int, set[str]] = defaultdict(set)
    active_samples_by_label: dict[int, set[str]] = defaultdict(set)
    active_references_by_label = Counter(int(row["label_5"]) for row in active)
    inactive_references_by_label = Counter(int(row["label_5"]) for row in inactive)
    for row in mask_rows:
        source_samples_by_label[int(row["label_5"])].add(str(row["record_id"]))
    for row in active:
        active_samples_by_label[int(row["label_5"])].add(str(row["record_id"]))
    coverage = []
    for label in sorted(source_samples_by_label):
        source_samples = len(source_samples_by_label[label])
        active_samples = len(active_samples_by_label[label])
        coverage.append(
            {
                "label_5": label,
                "source_reason_samples": source_samples,
                "active_reason_samples": active_samples,
                "fully_inactive_samples": source_samples - active_samples,
                "active_sample_coverage": (
                    active_samples / source_samples if source_samples else 0.0
                ),
                "active_references": active_references_by_label[label],
                "inactive_references": inactive_references_by_label[label],
            }
        )
    return {
        "source_references": len(mask_rows),
        "active_references": len(active),
        "inactive_references": len(inactive),
        "active_coverage": len(active) / len(mask_rows) if mask_rows else 0.0,
        "coverage_by_label": coverage,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--donor-map", type=Path, default=DEFAULT_DONOR_MAP)
    parser.add_argument("--donor-report", type=Path, default=DEFAULT_DONOR_REPORT)
    parser.add_argument("--donor-lock", type=Path, default=DEFAULT_DONOR_LOCK)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (
        args.references,
        args.donor_map,
        args.donor_report,
        args.donor_lock,
    ):
        reject_eval_path(path)
        if not path.exists():
            raise FileNotFoundError(path)

    reference_rows = read_jsonl(args.references, protect_split=True)
    donor_rows = read_jsonl(args.donor_map, protect_split=True)
    donor_report = json.loads(args.donor_report.read_text(encoding="utf-8"))
    donor_lock = json.loads(args.donor_lock.read_text(encoding="utf-8"))

    if donor_report.get("status") != "R2_TOKENIZED_DONOR_MAP_READY":
        raise ValueError("active mask requires R2_TOKENIZED_DONOR_MAP_READY")
    if donor_report.get("tokenizer_lock", {}).get("status") != "QWEN_TOKENIZER_REVISION_LOCKED":
        raise ValueError("active mask requires QWEN_TOKENIZER_REVISION_LOCKED")
    if donor_lock.get("donor_map_sha256") != file_sha256(args.donor_map):
        raise ValueError("formal donor map hash does not match its lock")
    if donor_lock.get("reference_source_sha256") != file_sha256(args.references):
        raise ValueError("reference inventory hash does not match the donor lock")

    validation = validate_map(reference_rows, donor_rows)
    if not all(validation["checks"].values()):
        raise ValueError("formal donor map failed strict validation")

    mask_rows = build_mask_rows(reference_rows, donor_rows)
    summary = summarize_mask(mask_rows)
    out_dir: Path = args.out_dir
    r2_path = out_dir / "data/r2_rationale_active_mask.jsonl"
    r3_path = out_dir / "data/r3_rationale_active_mask.jsonl"
    write_jsonl(r2_path, mask_rows)
    write_jsonl(r3_path, mask_rows)
    r2_sha256 = file_sha256(r2_path)
    r3_sha256 = file_sha256(r3_path)
    checks = {
        "donor_map_strict_checks_pass": all(validation["checks"].values()),
        "r2_r3_rows_identical": r2_path.read_bytes() == r3_path.read_bytes(),
        "r2_r3_sha256_identical": r2_sha256 == r3_sha256,
        "active_count_matches_donor_report": summary["active_references"]
        == donor_report["counts"]["active_references"],
        "inactive_count_matches_donor_report": summary["inactive_references"]
        == donor_report["counts"]["inactive_references"],
        "mask_does_not_encode_score_deactivation": all(
            "score_active" not in row and "score_mask" not in row
            for row in mask_rows
        ),
    }
    status = (
        "R2_STRICT_DONOR_MAP_READY"
        if all(checks.values())
        else "R2_R3_ACTIVE_MASK_NO_GO"
    )
    report = {
        "status": status,
        "checks": checks,
        "counts": summary,
        "r2_active_mask_sha256": r2_sha256,
        "r3_active_mask_sha256": r3_sha256,
        "donor_map_sha256": file_sha256(args.donor_map),
        "donor_report_sha256": file_sha256(args.donor_report),
        "tokenizer_lock_sha256": donor_report["tokenizer_lock"]["tokenizer_lock_sha256"],
        "training_manifest_build_allowed": status == "R2_STRICT_DONOR_MAP_READY",
        "formal_training_allowed": False,
        "next_required_gate": "S0_R1_R2_R3_TRAINING_MANIFESTS_FROZEN",
        "dev_accessed": False,
        "test_accessed": False,
        "training_used": False,
    }
    write_json(out_dir / "audit/r2_r3_active_mask_report.json", report)
    write_json(
        out_dir / "protocol/r2_r3_active_mask_lock.json",
        {
            "status": status,
            "r2_active_mask_sha256": r2_sha256,
            "r3_active_mask_sha256": r3_sha256,
            "donor_map_sha256": file_sha256(args.donor_map),
            "donor_map_lock_sha256": file_sha256(args.donor_lock),
            "tokenizer_lock_sha256": donor_report["tokenizer_lock"]["tokenizer_lock_sha256"],
            "training_manifest_build_allowed": status == "R2_STRICT_DONOR_MAP_READY",
            "formal_training_allowed": False,
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if status != "R2_STRICT_DONOR_MAP_READY":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
