"""Prepare the train-only Exp28 teacher-annotation inventory and protocol pilot.

The output packets contain paper-like train rows only. Original benchmark
labels are retained in a private reference packet and excluded from blind
teacher packets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SPLIT_DIR = Path("thesis_exp/data/splits/paper_like_triple_seed42")
DEFAULT_LEGACY_AUDIT = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/"
    "exp27i_target_aware_teacher_audit_361_seed42/data/"
    "exp27i_teacher_audited_361_calibrated_train.jsonl"
)
DEFAULT_OUT_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/"
    "exp28b_paper_train_annotation_inventory_seed42"
)
SEED = 42
SUBSET_LABEL_TARGETS = {
    "protocol_demo_reference": {1: 3, 2: 3, 3: 3, 4: 3, 5: 3},
    "protocol_development": {1: 5, 2: 8, 3: 12, 4: 17, 5: 18},
    "sealed_qualification": {1: 10, 2: 18, 3: 24, 4: 32, 5: 36},
}
SUBSET_TARGETS = {name: sum(targets.values()) for name, targets in SUBSET_LABEL_TARGETS.items()}
SUBSET_MIN_QUESTION_KEYS = {
    "protocol_demo_reference": 5,
    "protocol_development": 20,
    "sealed_qualification": 40,
}
BLIND_INPUT_FIELDS = ("question", "answer", "metric", "rubric", "metadata")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sample_id(row: dict[str, Any]) -> str:
    value = row.get("record_id") or row.get("sample_id")
    if not value:
        raise ValueError("Missing record_id/sample_id")
    return str(value)


def question_key(row: dict[str, Any]) -> str:
    value = row.get("question_key") or row.get("question_id")
    if not value:
        raise ValueError(f"Missing question key for {sample_id(row)}")
    return str(value)


def label(row: dict[str, Any]) -> int:
    value = int(float(row["label_5"]))
    if value not in range(1, 6):
        raise ValueError(f"Invalid label for {sample_id(row)}: {value}")
    return value


def human_scores(row: dict[str, Any]) -> list[int]:
    values = []
    for key in ("human_1_5", "human_2_5", "human_3_5", "human_1", "human_2", "human_3"):
        if key.startswith("human_") and key.endswith("_5"):
            value = row.get(key)
            if value is not None:
                values.append(int(float(value)))
    if not values:
        for key in ("human_1", "human_2", "human_3"):
            if row.get(key) is not None:
                values.append(int(float(row[key])))
    return values[:3]


def stable_rank(value: str) -> str:
    return hashlib.sha256(f"{SEED}|{value}".encode("utf-8")).hexdigest()


def normalized_input(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    if metadata is None:
        metadata = {
            "scenario": row.get("scenario_canonical"),
            "subject": row.get("subject_canonical"),
            "education_level": row.get("education_level_canonical"),
            "language": row.get("language"),
            "generator_model": row.get("generator_model"),
        }
    return {
        "question": row.get("question", ""),
        "answer": row.get("answer", ""),
        "metric": row.get("metric_canonical") or row.get("metric") or row.get("metric_raw") or "",
        "rubric": row.get("rubric", []),
        "metadata": metadata,
    }


def allocate_question_groups(rows: list[dict[str, Any]]) -> dict[str, str]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[question_key(row)].append(row)
    available = set(groups)
    assignment: dict[str, str] = {}
    for subset, targets in SUBSET_LABEL_TARGETS.items():
        pool_counts = Counter()
        assigned_count = 0
        while (
            any(pool_counts[score] < target for score, target in targets.items())
            or assigned_count < SUBSET_MIN_QUESTION_KEYS[subset]
        ):
            deficits = {score: max(0, target - pool_counts[score]) for score, target in targets.items()}
            candidates = []
            for key in available:
                counts = Counter(label(row) for row in groups[key])
                coverage = sum(min(counts[score], deficits[score]) / max(1, targets[score]) for score in targets)
                rare_coverage = 2.0 * min(counts[1], deficits[1]) + min(counts[2], deficits[2])
                candidates.append((coverage, rare_coverage, stable_rank(key), key, counts))
            if not candidates:
                raise RuntimeError(f"Insufficient question groups for {subset}: {deficits}")
            _, _, _, key, counts = max(candidates, key=lambda item: (item[0], item[1], item[2]))
            assignment[key] = subset
            available.remove(key)
            pool_counts.update(counts)
            assigned_count += 1
    return assignment


def stratified_take(
    rows: list[dict[str, Any]], targets: dict[int, int], min_question_keys: int
) -> list[dict[str, Any]]:
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_question[question_key(row)].append(row)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_counts = Counter()
    for key in sorted(by_question, key=stable_rank):
        if len({question_key(row) for row in selected}) >= min_question_keys:
            break
        candidates = [row for row in by_question[key] if selected_counts[label(row)] < targets[label(row)]]
        if not candidates:
            continue
        chosen = max(
            candidates,
            key=lambda row: (
                (targets[label(row)] - selected_counts[label(row)]) / max(1, targets[label(row)]),
                stable_rank(sample_id(row)),
            ),
        )
        selected.append(chosen)
        selected_ids.add(sample_id(chosen))
        selected_counts[label(chosen)] += 1
    if len({question_key(row) for row in selected}) < min_question_keys:
        raise RuntimeError(f"Could not cover {min_question_keys} question keys")

    buckets: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if sample_id(row) in selected_ids:
            continue
        buckets[(label(row), str(row.get("language") or "unknown"))].append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda row: stable_rank(sample_id(row)))
    for score, target in targets.items():
        language_keys = sorted(key for key in buckets if key[0] == score)
        score_selected = selected_counts[score]
        while score_selected < target:
            progressed = False
            for key in language_keys:
                if buckets[key] and score_selected < target:
                    selected.append(buckets[key].pop(0))
                    score_selected += 1
                    progressed = True
            if not progressed:
                raise RuntimeError(f"Could not select label={score} target={target}")
    return selected


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    train_path = args.split_dir / "train.jsonl"
    dev_path = args.split_dir / "dev.jsonl"
    test_path = args.split_dir / "test.jsonl"
    for path in (train_path, dev_path, test_path, args.legacy_audit):
        if not path.exists():
            raise FileNotFoundError(path)

    train = read_jsonl(train_path)
    if len(train) != 2654:
        raise ValueError(f"Expected 2654 train rows, found {len(train)}")
    train_ids = {sample_id(row) for row in train}
    if len(train_ids) != len(train):
        raise ValueError("Duplicate paper-train sample IDs")

    # Held-out rows are reduced to identity guards; labels are never accessed.
    heldout_ids: set[str] = set()
    heldout_questions: set[str] = set()
    for path in (dev_path, test_path):
        for row in read_jsonl(path):
            heldout_ids.add(sample_id(row))
            heldout_questions.add(question_key(row))
    if train_ids & heldout_ids:
        raise ValueError("Sample overlap between train and held-out splits")

    legacy_by_id = {sample_id(row): row for row in read_jsonl(args.legacy_audit)}
    reusable_legacy_ids = train_ids & set(legacy_by_id)
    assignment = allocate_question_groups(train)
    subset_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train:
        subset = assignment.get(question_key(row))
        if subset:
            subset_candidates[subset].append(row)
    selected_by_subset = {
        subset: stratified_take(subset_candidates[subset], targets, SUBSET_MIN_QUESTION_KEYS[subset])
        for subset, targets in SUBSET_LABEL_TARGETS.items()
    }

    selected_ids = {
        sample_id(row): subset
        for subset, rows in selected_by_subset.items()
        for row in rows
    }
    selected_question_sets = {
        subset: {question_key(row) for row in rows}
        for subset, rows in selected_by_subset.items()
    }
    subsets = list(SUBSET_TARGETS)
    for index, left in enumerate(subsets):
        for right in subsets[index + 1 :]:
            if selected_question_sets[left] & selected_question_sets[right]:
                raise ValueError(f"Pilot question overlap: {left} vs {right}")

    blind_packets = []
    reference_packets = []
    manifest_rows = []
    for row in train:
        sid = sample_id(row)
        scores = human_scores(row)
        agreement = "missing"
        if len(scores) == 3:
            gap = max(scores) - min(scores)
            agreement = "unanimous" if gap == 0 else "adjacent" if gap == 1 else "wide_disagreement"
        subset = selected_ids.get(sid, "full_annotation_pool")
        blind = {
            "sample_id": sid,
            "question_key": question_key(row),
            "protocol_subset": subset,
            "teacher_input": normalized_input(row),
        }
        if set(blind["teacher_input"]) != set(BLIND_INPUT_FIELDS):
            raise ValueError(f"Blind packet schema mismatch for {sid}")
        blind_packets.append(blind)
        reference_packets.append(
            {
                "sample_id": sid,
                "question_key": question_key(row),
                "protocol_subset": subset,
                "original_label": label(row),
                "human_scores": scores,
                "human_agreement": agreement,
                "legacy_exp27_teacher_annotation_available": sid in reusable_legacy_ids,
            }
        )
        manifest_rows.append(
            {
                "sample_id": sid,
                "question_key": question_key(row),
                "protocol_subset": subset,
                "original_label": label(row),
                "language": row.get("language", ""),
                "metric": row.get("metric_canonical", ""),
                "subject": row.get("subject_canonical", ""),
                "human_agreement": agreement,
                "legacy_exp27_train_annotation_available": sid in reusable_legacy_ids,
                "primary_teacher_required": True,
                "secondary_teacher_route_initial": (
                    label(row) <= 2 or agreement == "wide_disagreement" or stable_rank(sid).startswith(("0", "1"))
                ),
            }
        )

    out = args.out_dir
    write_jsonl(out / "private" / "exp28b_blind_teacher_packets_2654.jsonl", blind_packets)
    write_jsonl(out / "private" / "exp28b_benchmark_reference_2654.jsonl", reference_packets)
    write_csv(
        out / "tables" / "exp28b_annotation_inventory_light.csv",
        manifest_rows,
        [
            "sample_id", "question_key", "protocol_subset", "original_label", "language", "metric", "subject",
            "human_agreement", "legacy_exp27_train_annotation_available", "primary_teacher_required",
            "secondary_teacher_route_initial",
        ],
    )

    distribution_rows = []
    for subset in [*SUBSET_TARGETS, "full_annotation_pool", "all_train"]:
        source = train if subset == "all_train" else [row for row in train if selected_ids.get(sample_id(row), "full_annotation_pool") == subset]
        counts = Counter(label(row) for row in source)
        distribution_rows.append(
            {
                "subset": subset,
                "rows": len(source),
                **{f"label_{score}": counts[score] for score in range(1, 6)},
                "unique_question_keys": len({question_key(row) for row in source}),
            }
        )
    write_csv(
        out / "tables" / "exp28b_protocol_subset_distribution.csv",
        distribution_rows,
        ["subset", "rows", "label_1", "label_2", "label_3", "label_4", "label_5", "unique_question_keys"],
    )

    leakage_rows = [
        {"check": "train_sample_overlap_with_heldout", "count": len(train_ids & heldout_ids), "status": "PASS"},
        {
            "check": "paper_protocol_question_overlap_with_heldout",
            "count": len({question_key(row) for row in train} & heldout_questions),
            "status": "EXPECTED_WARNING",
        },
        {"check": "blind_packets_contain_original_label", "count": sum("original_label" in row for row in blind_packets), "status": "PASS"},
        {
            "check": "legacy_exp27_annotations_quarantined_outside_paper_train",
            "count": len(set(legacy_by_id) - reusable_legacy_ids),
            "status": "PASS",
        },
    ]
    write_csv(out / "tables" / "exp28b_leakage_and_blinding_audit.csv", leakage_rows, ["check", "count", "status"])

    decision = {
        "status": "READY_FOR_PROTOCOL_DRY_RUN",
        "train_rows": len(train),
        "blind_packet_rows": len(blind_packets),
        "reference_rows": len(reference_packets),
        "legacy_paper_train_rows_available": len(reusable_legacy_ids),
        "legacy_annotations_used_as_new_targets": 0,
        "protocol_subsets": SUBSET_TARGETS,
        "protocol_subset_label_targets": SUBSET_LABEL_TARGETS,
        "protocol_subset_min_question_keys": SUBSET_MIN_QUESTION_KEYS,
        "primary_teacher_scope": "all_2654_paper_train_rows_after_protocol_lock",
        "secondary_teacher_scope": "risk_routed_after_primary_output_plus_locked_high_controls",
        "api_called": False,
        "test_labels_read": False,
        "next_action": "Dry-run P0/P1/P2 message construction, then annotate development and sealed qualification subsets before full 2654 expansion.",
    }
    write_json(out / "decision" / "exp28b_annotation_inventory_decision.json", decision)
    report = f"""# Exp28B Paper-Train Annotation Inventory

## Status

**READY FOR PROTOCOL DRY RUN**

- paper-train rows: {len(train)}
- blind teacher packets: {len(blind_packets)}
- benchmark reference rows: {len(reference_packets)}
- legacy Exp27 rows that map to paper train: {len(reusable_legacy_ids)}
- legacy rows used as new targets: 0

The blind packet contains only question, answer, metric, rubric, and metadata. Original labels
and individual human scores live in a separate private reference packet.

## Protocol Subsets

- protocol demonstration references: 15 (benchmark labels; not expert-reviewed)
- protocol development: 60
- sealed qualification: 120
- full annotation pool: all 2,654 paper-train rows

The three pilot subsets use disjoint question keys. They are all drawn from paper train. Dev and
test identities are used only for leakage guards, and held-out labels are not read.

## Teacher Routing

1. The primary teacher will annotate all 2,654 rows after protocol qualification.
2. A secondary teacher is routed to low-score, high-disagreement, primary/original-conflict,
   low-confidence, and locked high-control cases.
3. Remaining high teacher conflicts receive independent model adjudication recorded as
   `model_review_silver`.
4. No model output is described as human review.
"""
    report_path = out / "reports" / "exp28b_annotation_inventory_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--legacy-audit", type=Path, default=DEFAULT_LEGACY_AUDIT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(prepare(parse_args()), ensure_ascii=False, sort_keys=True))
