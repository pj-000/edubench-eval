"""Prepare the train-only Exp37A blind failure/evidence qualification set."""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp37_failure_evidence_qualification.common import (  # noqa: E402
    ROOT, TRAIN_PATH, largest_remainder_quota, packet_hash, question_key, read_jsonl, sample_id, sha256_text,
    stratified_sample, stratum, subject, write_csv, write_json, write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, default=TRAIN_PATH)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def label(row: dict[str, Any]) -> int:
    return int(round(float(row.get("label_5", row.get("human_mean_5", 0)))))


def choose_high_controls(low: list[dict[str, Any]], high: list[dict[str, Any]], total: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Match label-5 controls to low-tail language/metric strata deterministically."""
    targets = Counter(stratum(row) for row in low)
    available: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in high:
        available[stratum(row)].append(row)
    selected: list[dict[str, Any]] = []
    match_status: dict[str, str] = {}
    # A stable order avoids dependence on the source JSONL order.
    for key in sorted(available):
        available[key].sort(key=sample_id)
    quotas = largest_remainder_quota(dict(targets), total)
    for key in sorted(targets):
        quota = min(quotas.get(key, 0), len(available.get(key, [])))
        selected.extend(available[key][:quota])
        for row in available[key][:quota]:
            match_status[sample_id(row)] = "exact_language_metric"
    remaining = [row for key in sorted(available) for row in available[key] if row not in selected]
    remaining.sort(key=lambda row: (
        min((0 if stratum(row) == target else (1 if stratum(row)[0] == target[0] else 2)) for target in targets),
        stratum(row), sample_id(row),
    ))
    for row in remaining:
        if len(selected) >= total:
            break
        selected.append(row)
        same_language = any(stratum(row)[0] == key[0] for key in targets)
        match_status[sample_id(row)] = "nearest_language_metric" if same_language else "nearest_available_stratum"
    if len(selected) < total:
        raise ValueError(f"Only {len(selected)} high controls available; need {total}")
    return sorted(selected[:total], key=sample_id), match_status


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


def distribution(rows: list[dict[str, Any]], view: str) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, str, str]] = Counter()
    for row in rows:
        counts[(view, stratum(row)[0], stratum(row)[1], subject(row))] += 1
    return [{"view": key[0], "language": key[1], "metric_group": key[2], "subject": key[3], "count": value} for key, value in sorted(counts.items())]


def main() -> None:
    args = parse_args()
    train = read_jsonl(args.train_jsonl)
    if len(train) != 2654:
        raise ValueError(f"Expected paper-like train to contain 2654 rows, found {len(train)}")
    ids = [sample_id(row) for row in train]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate train sample IDs")
    low = [row for row in train if label(row) <= 2]
    boundary3 = [row for row in train if label(row) == 3]
    boundary4 = [row for row in train if label(row) == 4]
    high = [row for row in train if label(row) == 5]
    if len(low) != 76:
        raise ValueError(f"Expected 76 low-tail rows, found {len(low)}")
    boundary = stratified_sample(boundary3, 30, args.seed) + stratified_sample(boundary4, 30, args.seed + 1)
    controls, control_status = choose_high_controls(low, high, 60, args.seed)
    views = {"low_tail_all": sorted(low, key=sample_id), "boundary_view": sorted(boundary, key=sample_id), "high_control_view": controls}
    view_ids = {name: {sample_id(row) for row in rows} for name, rows in views.items()}
    union = set().union(*view_ids.values())
    overlap_rows = []
    names = list(views)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            overlap_rows.append({"left_view": left, "right_view": right, "overlap_count": len(view_ids[left] & view_ids[right])})
    qkey_by_id = {sample_id(row): question_key(row) for row in train}
    qkey_rows = []
    for name, rows in views.items():
        qkeys = [qkey_by_id[sample_id(row)] for row in rows]
        qkey_rows.append({"view": name, "sample_count": len(rows), "unique_question_keys": len(set(qkeys)), "question_key_repeated_rows": len(qkeys) - len(set(qkeys))})
    all_packets = []
    packet_paths = {}
    for view, rows in views.items():
        packets = [build_packet(row) for row in rows]
        all_packets.extend({**packet, "view": view} for packet in packets)
        private_path = args.out_dir / "private_packets" / f"exp37a_{view}_blind_packet.jsonl"
        write_jsonl(private_path, packets)
        packet_paths[view] = str(private_path)
        write_jsonl(args.out_dir / "annotation_templates" / f"exp37a_{view}_reviewer_a_template.jsonl", [
            {"sample_id": packet["sample_id"], "packet_hash": packet["packet_hash"], "review_status": "pending"} for packet in packets
        ])
        write_jsonl(args.out_dir / "annotation_templates" / f"exp37a_{view}_reviewer_b_template.jsonl", [
            {"sample_id": packet["sample_id"], "packet_hash": packet["packet_hash"], "review_status": "pending"} for packet in packets
        ])
    write_jsonl(args.out_dir / "private_packets" / "exp37a_adjudication_template.jsonl", [
        {"sample_id": packet["sample_id"], "packet_hash": packet["packet_hash"], "review_status": "pending"} for packet in all_packets
    ])
    write_csv(args.out_dir / "tables/exp37a_sampling_distribution.csv", distribution(low, "low_tail_all") + distribution(boundary, "boundary_view") + distribution(controls, "high_control_view"))
    write_csv(args.out_dir / "tables/exp37a_question_key_distribution.csv", qkey_rows)
    write_csv(args.out_dir / "tables/exp37a_view_overlap_audit.csv", overlap_rows)
    write_csv(args.out_dir / "tables/exp37a_control_matching.csv", [{"sample_id": sample_id(row), "match_status": control_status.get(sample_id(row), "") , "language": row.get("language"), "metric_group": row.get("metric_group")} for row in controls])
    report = {
        "experiment": "Exp37A Human-Rationale-Grounded Failure-Evidence Qualification",
        "train_path": str(args.train_jsonl), "train_rows": len(train),
        "view_counts": {name: len(rows) for name, rows in views.items()},
        "unique_sample_ids": len(union), "expected_unique_sample_ids": 196,
        "unique_question_keys": len({qkey_by_id[sid] for sid in union}),
        "question_key_overlap_allowed_and_recorded": True,
        "packet_paths_private": packet_paths,
        "api_called": False, "gpu_used": False, "student_training": False,
        "student_inference": False, "dev_access_count": 0, "test_access_count": 0,
        "human_review_results_present": False,
    }
    write_json(args.out_dir / "decision/exp37a_failure_evidence_qualification_decision.json", {
        "reference_complete": False,
        "recommend_new_reason_evidence_training": False,
        "recommend_full_train_score_range_annotation": False,
        "recommend_student_training": False,
        "status": "WAITING_FOR_EXTERNAL_REVIEWS",
        "reason": "Blind Reviewer A/B and adjudication outputs are not yet present; no semantic metrics are fabricated.",
        "test_access_count": 0,
    })
    lines = [
        "# Exp37A prepare report", "", "This is a train-only qualification packet. No student model was trained or inferred.", "",
        f"- Train rows: {len(train)}", f"- low_tail_all: {len(low)}", f"- boundary_view: {len(boundary)}", f"- high_control_view: {len(controls)}",
        f"- Unique sample IDs: {len(union)}", f"- Unique question keys: {report['unique_question_keys']}",
        "- Blind reviewer packets contain no human labels, human reasons, teacher outputs, OOF predictions, or Exp36 results.",
        "- Reviewers A/B must fill the private review templates externally; Reviewer C adjudicates only the flagged disagreements.",
        "- Until those files exist, all semantic decisions remain false/unknown.", "",
        "## Boundary", "Dev and test were not read. API calls, GPU use, student training, and student inference were not performed.",
    ]
    (args.out_dir / "reports/exp37a_prepare_report.md").parent.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "reports/exp37a_prepare_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"status": "PASS", "views": {name: len(rows) for name, rows in views.items()}, "unique_sample_ids": len(union), "test_access_count": 0})


if __name__ == "__main__":
    main()
