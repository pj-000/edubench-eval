"""Audit whether Exp36A V7 changed payloads, not merely donor row IDs."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp37_failure_evidence_qualification.common import (  # noqa: E402
    OOF_PATH, ROOT, TRAIN_PATH, V5_PATH, V7_PATH, read_jsonl, sample_id, write_csv, write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, default=TRAIN_PATH)
    parser.add_argument("--v5", type=Path, default=V5_PATH)
    parser.add_argument("--v7", type=Path, default=V7_PATH)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    return parser.parse_args()


def unequal(left: Any, right: Any) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        try:
            return abs(float(left) - float(right)) > 1e-9
        except (TypeError, ValueError):
            pass
    return left != right


def main() -> None:
    args = parse_args()
    train = {sample_id(row): row for row in read_jsonl(args.train_jsonl)}
    if not args.v5.exists() or not args.v7.exists():
        status = {
            "status": "MISSING_PRIVATE_EXP36_DATA",
            "v5_exists": args.v5.exists(), "v7_exists": args.v7.exists(),
            "moved_donor_rows": None, "actual_teacher_target_changed_rows": None,
            "actual_failure_target_changed_rows": None, "actual_lambda_changed_rows": None,
            "all_payload_identical_rows": None, "donor_field": None, "test_access_count": 0,
        }
        write_csv(args.out_dir / "tables/exp37a_r1_v7_payload_audit_status.csv", [status])
        write_json(args.out_dir / "decision/exp37a_r1_v7_payload_audit.json", status)
        print(status)
        return
    v5 = {sample_id(row): row for row in read_jsonl(args.v5)}
    v7 = {sample_id(row): row for row in read_jsonl(args.v7)}
    if set(v5) != set(v7):
        raise ValueError("V5/V7 sample IDs are not aligned")
    fields = {
        "teacher_target_changed": "teacher_target_5",
        "teacher_score_changed": "teacher_score",
        "teacher_lambda_changed": "teacher_lambda",
        "failure_target_changed": "failure_target_6",
        "failure_mask_changed": "failure_mask",
    }
    donor_fields = ("donor_sample_id", "source_sample_id", "teacher_donor_sample_id", "shuffled_from_sample_id")
    donor_field = next((field for field in donor_fields if any(field in row for row in v7.values())), None)
    rows = []
    for sid in sorted(v5):
        row = {"sample_id": sid, "human_label": train.get(sid, {}).get("label_5"), "language": train.get(sid, {}).get("language"), "metric_group": train.get(sid, {}).get("metric_group")}
        for name, field in fields.items():
            row[name] = int(unequal(v5[sid].get(field), v7[sid].get(field)))
        row["all_teacher_payload_identical"] = int(not any(row[name] for name in fields))
        row["donor_row_moved"] = int(donor_field is not None and str(v7[sid].get(donor_field, sid)) != sid)
        rows.append(row)
    write_csv(args.out_dir / "private_reference/exp37a_r1_v7_effective_shuffle_rows.csv", rows)
    aggregate = [{
        "group": "all", "group_value": "all", "rows": len(rows),
        **{name + "_rows": sum(row[name] for row in rows) for name in fields},
        "all_payload_identical_rows": sum(row["all_teacher_payload_identical"] for row in rows),
        "moved_donor_rows": sum(row["donor_row_moved"] for row in rows) if donor_field else "not_available_from_payload_only",
        "donor_field": donor_field,
        "test_access_count": 0,
    }]
    for group in ("human_label", "language", "metric_group"):
        for value in sorted({str(row[group]) for row in rows}):
            subset = [row for row in rows if str(row[group]) == value]
            aggregate.append({
                "group": group, "group_value": value, "rows": len(subset),
                **{name + "_rows": sum(row[name] for row in subset) for name in fields},
                "all_payload_identical_rows": sum(row["all_teacher_payload_identical"] for row in subset),
                "moved_donor_rows": sum(row["donor_row_moved"] for row in subset) if donor_field else "not_available_from_payload_only", "donor_field": donor_field, "test_access_count": 0,
            })
    write_csv(args.out_dir / "tables/exp37a_r1_v7_payload_audit_status.csv", aggregate)
    status = {"status": "PASS", "rows": len(rows), "donor_field": donor_field, "moved_donor_rows": sum(row["donor_row_moved"] for row in rows) if donor_field else "not_available_from_payload_only", "test_access_count": 0, "note": "Actual payload field changes are counted directly; donor movement is counted when a donor ID field is present."}
    write_json(args.out_dir / "decision/exp37a_r1_v7_payload_audit.json", status)
    print(status)


if __name__ == "__main__":
    main()
