"""Prepare the frozen train-only Exp37A-R1 multi-session review protocol."""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp37_failure_evidence_qualification.common import (  # noqa: E402
    EVIDENCE_SUFFICIENCY,
    EVIDENCE_TYPES,
    EXPERIMENT_NAME,
    FAILURE_CLASSES,
    MAJOR_FAILURE_PRESENCE,
    R0_ROOT,
    REFERENCE_TYPE,
    ROOT,
    TRAIN_PATH,
    packet_hash,
    question_key,
    read_jsonl,
    sample_id,
    sha256_text,
    stratum,
    subject,
    write_csv,
    write_json,
    write_jsonl,
)

EXPECTED_VIEW_COUNTS = {"low_tail_all": 76, "boundary_view": 60, "high_control_view": 60}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, default=TRAIN_PATH)
    parser.add_argument("--out-dir", "--output-dir", dest="out_dir", type=Path, default=ROOT)
    parser.add_argument("--r0-out-dir", type=Path, default=R0_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_packet(row: dict[str, Any]) -> dict[str, Any]:
    packet = {
        "sample_id": sample_id(row),
        "anonymized_question_key_hash": sha256_text(question_key(row)),
        "question_context": row.get("question", ""),
        "evaluator_output": row.get("answer", ""),
        "metric": row.get("metric_canonical") or row.get("metric_raw") or row.get("metric_id", ""),
        "rubric": row.get("rubric", ""),
        "non_label_metadata": {
            "language": row.get("language"),
            "subject": row.get("subject_canonical") or row.get("subject_raw"),
            "education_level": row.get("education_level_canonical") or row.get("education_level_raw"),
            "scenario": row.get("scenario_canonical") or row.get("scenario_raw"),
            "metric_group": row.get("metric_group"),
        },
    }
    packet["review_text"] = (
        "<CONTEXT_ONLY_ORIGINAL_TASK>\n"
        + str(packet["question_context"])
        + "\nmetric: " + str(packet["metric"])
        + "\nrubric: " + str(packet["rubric"])
        + "\n</CONTEXT_ONLY_ORIGINAL_TASK>\n"
        "<EVALUATOR_OUTPUT_TO_SCORE>\n"
        + str(packet["evaluator_output"])
        + "\n</EVALUATOR_OUTPUT_TO_SCORE>"
    )
    packet["packet_hash"] = packet_hash(packet)
    return packet


def load_frozen_views(r0_out_dir: Path) -> tuple[dict[str, list[dict[str, str]]], dict[str, str]]:
    views: dict[str, list[dict[str, str]]] = {}
    id_to_view: dict[str, str] = {}
    template_dir = r0_out_dir / "annotation_templates"
    for view, expected in EXPECTED_VIEW_COUNTS.items():
        path = template_dir / f"exp37a_{view}_reviewer_a_template.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"R0 frozen template missing: {path}")
        rows = read_jsonl(path)
        if len(rows) != expected:
            raise ValueError(f"R0 {view} expected {expected} rows, found {len(rows)}")
        views[view] = rows
        for row in rows:
            sid = str(row["sample_id"])
            if sid in id_to_view:
                raise ValueError(f"R0 sample appears in multiple views: {sid}")
            id_to_view[sid] = view
    if len(id_to_view) != 196:
        raise ValueError(f"R0 freeze must contain 196 unique sample IDs, found {len(id_to_view)}")
    return views, id_to_view


def qkey_overlap_rows(view_ids: dict[str, list[str]], train: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    qkeys = {view: {question_key(train[sid]) for sid in ids} for view, ids in view_ids.items()}
    rows: list[dict[str, Any]] = []
    for view in sorted(qkeys):
        rows.append({"scope": "within_view", "left_view": view, "right_view": view, "unique_question_keys": len(qkeys[view]), "overlap_count": len(qkeys[view])})
    names = sorted(qkeys)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            rows.append({"scope": "pairwise", "left_view": left, "right_view": right, "unique_question_keys": "", "overlap_count": len(qkeys[left] & qkeys[right])})
    rows.append({"scope": "three_way", "left_view": "all_three", "right_view": "all_three", "unique_question_keys": "", "overlap_count": len(set.intersection(*(qkeys[name] for name in names)))})
    rows.append({"scope": "union", "left_view": "all_views", "right_view": "all_views", "unique_question_keys": len(set.union(*(qkeys[name] for name in names))), "overlap_count": ""})
    return rows


def main() -> None:
    args = parse_args()
    train_rows = read_jsonl(args.train_jsonl)
    if len(train_rows) != 2654:
        raise ValueError(f"Expected 2654 paper-like train rows, found {len(train_rows)}")
    train = {sample_id(row): row for row in train_rows}
    if len(train) != len(train_rows):
        raise ValueError("Duplicate train sample IDs")

    frozen, id_to_view = load_frozen_views(args.r0_out_dir)
    missing = set(id_to_view) - set(train)
    if missing:
        raise ValueError(f"Frozen R0 IDs missing from train: {len(missing)}")

    packet_by_id = {sid: build_packet(train[sid]) for sid in sorted(id_to_view)}
    hash_mismatches = []
    for view, rows in frozen.items():
        expected_hash = {str(row["sample_id"]): str(row["packet_hash"]) for row in rows}
        for sid in expected_hash:
            if packet_by_id[sid]["packet_hash"] != expected_hash[sid]:
                hash_mismatches.append(sid)
    if hash_mismatches:
        raise ValueError(f"R1 packet input differs from frozen R0 for {len(hash_mismatches)} samples")

    view_ids = {view: sorted(str(row["sample_id"]) for row in rows) for view, rows in frozen.items()}
    all_ids = sorted(id_to_view)
    reviewer_seeds = {"reviewer_a": args.seed, "reviewer_b": args.seed + 1}
    for reviewer, seed in reviewer_seeds.items():
        order = list(all_ids)
        random.Random(seed).shuffle(order)
        write_jsonl(args.out_dir / "private_packets" / f"exp37a_r1_{reviewer}_packets.jsonl", [packet_by_id[sid] for sid in order])
        write_jsonl(args.out_dir / "annotation_templates" / f"exp37a_r1_{reviewer}_template.jsonl", [
            {"sample_id": sid, "packet_hash": packet_by_id[sid]["packet_hash"], "review_status": "pending"}
            for sid in order
        ])

    distribution = Counter()
    for sid, view in id_to_view.items():
        row = train[sid]
        distribution[(view, stratum(row)[0], stratum(row)[1], subject(row), int(round(float(row["label_5"]))))] += 1
    write_csv(args.out_dir / "tables/exp37a_r1_sampling_distribution.csv", [
        {"view": key[0], "language": key[1], "metric_group": key[2], "subject": key[3], "rounded_human_label": key[4], "count": value}
        for key, value in sorted(distribution.items())
    ])
    overlap = qkey_overlap_rows(view_ids, train)
    write_csv(args.out_dir / "tables/exp37a_r1_question_key_overlap_matrix.csv", overlap)

    sample_freeze_payload = [
        {"sample_id": sid, "view": id_to_view[sid], "packet_hash": packet_by_id[sid]["packet_hash"]}
        for sid in all_ids
    ]
    write_json(args.out_dir / "configs/exp37a_r1_sampling_hashes.json", {
        "r0_output": str(args.r0_out_dir),
        "r1_sample_count": len(all_ids),
        "sample_ids_sha256": sha256_text("\n".join(all_ids)),
        "view_assignment_sha256": sha256_text(sample_freeze_payload),
        "packet_input_sha256": sha256_text([packet_by_id[sid] for sid in all_ids]),
        "r0_r1_sample_ids_equal": True,
        "r0_r1_view_assignment_equal": True,
        "r0_r1_packet_hashes_equal": True,
    })
    write_json(args.out_dir / "configs/exp37a_r1_protocol_lock.json", {
        "experiment_name": EXPERIMENT_NAME,
        "reference_type": REFERENCE_TYPE,
        "train_path": str(args.train_jsonl),
        "sample_count": 196,
        "view_counts": EXPECTED_VIEW_COUNTS,
        "reviewer_packet_seeds": reviewer_seeds,
        "failure_classes": FAILURE_CLASSES,
        "major_failure_presence": MAJOR_FAILURE_PRESENCE,
        "evidence_types": EVIDENCE_TYPES,
        "evidence_sufficiency": EVIDENCE_SUFFICIENCY,
        "permutation_count": 1000,
        "question_key_bootstrap_resamples": 2000,
        "student_training": False,
        "student_inference": False,
        "api_calls": False,
        "dev_access_count": 0,
        "test_access_count": 0,
    })

    report = [
        "# Exp37A-R1 prepare report", "",
        f"Experiment: {EXPERIMENT_NAME}", "",
        "- R0/R1 sample IDs equal: `true`",
        "- R0/R1 view assignments equal: `true`",
        "- R0/R1 packet input hashes equal: `true`",
        "- low_tail_all: 76 rows",
        "- boundary_view: 60 rows",
        "- high_control_view: 60 rows",
        f"- Unique question keys: {next(row['unique_question_keys'] for row in overlap if row['scope'] == 'union')}",
        "- Reviewer A/B packet orders are independently randomized with frozen seeds 42/43.",
        "- Human reasons are not supplied to A, B, or C.",
        "- No API, GPU, student training, student inference, dev, or test access occurred.",
    ]
    (args.out_dir / "reports/exp37a_r1_prepare_report.md").parent.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "reports/exp37a_r1_prepare_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print({"status": "PREPARED", "rows": 196, "views": EXPECTED_VIEW_COUNTS, "sample_ids_preserved": True, "test_access_count": 0})


if __name__ == "__main__":
    main()
