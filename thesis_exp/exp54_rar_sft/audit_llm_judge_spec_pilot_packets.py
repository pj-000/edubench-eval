"""Audit the private packet boundary for the LLM-judge specification pilot."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft.build_llm_judge_spec_pilot_packets import (
    BOUNDARIES,
    DEFAULT_BANK,
    DEFAULT_PRIVATE,
    DEFAULT_PUBLIC_REPORT,
    DEFAULT_TRAIN,
    FAILURE_BANK_SHA256,
    SCHEMA_VERSION,
    TRAIN_SHA256,
    canonical_json,
    clear_anchor,
    crossing_count,
    read_jsonl,
    rubric_key,
    sha256_file,
)


FORBIDDEN_VISIBLE_KEYS = {
    "record_id",
    "question_key",
    "gold_label",
    "label_5",
    "human_scores",
    "human_1_5",
    "human_2_5",
    "human_3_5",
    "generated_score",
    "generated_scores",
    "error_class",
    "boundary_crossing_seed_count",
    "selection_role",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _walk_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        keys.update(str(key) for key in value)
        for child in value.values():
            keys.update(_walk_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_walk_keys(child))
    return keys


def audit_packets(
    *,
    train_path: Path,
    bank_path: Path,
    private_dir: Path,
    public_report_path: Path,
) -> dict[str, Any]:
    if sha256_file(train_path) != TRAIN_SHA256:
        raise ValueError("train hash differs")
    if sha256_file(bank_path) != FAILURE_BANK_SHA256:
        raise ValueError("failure-bank hash differs")

    proposer_path = private_dir / "clarification_proposer_packet.json"
    evaluation_path = private_dir / "evaluation_source_packet.json"
    answer_key_path = private_dir / "private_answer_key.json"
    proposer = _read_json(proposer_path)
    evaluation = _read_json(evaluation_path)
    answer_key = _read_json(answer_key_path)
    report = _read_json(public_report_path)

    for value in (proposer, evaluation, answer_key):
        if value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("packet schema differs")
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("report schema differs")
    expected_hashes = {
        proposer_path.name: sha256_file(proposer_path),
        evaluation_path.name: sha256_file(evaluation_path),
        answer_key_path.name: sha256_file(answer_key_path),
    }
    if report.get("private_artifact_hashes") != expected_hashes:
        raise ValueError("private packet hash chain differs")

    for name, visible in (("proposer", proposer), ("evaluation", evaluation)):
        leaked = sorted(_walk_keys(visible) & FORBIDDEN_VISIBLE_KEYS)
        if leaked:
            raise ValueError(f"{name} packet leaks hidden keys: {leaked}")

    proposer_groups = proposer.get("groups", [])
    evaluation_groups = evaluation.get("groups", [])
    if len(proposer_groups) != 4 or len(evaluation_groups) != 4:
        raise ValueError("expected four groups in each visible packet")
    proposer_by_group = {group["group_id"]: group for group in proposer_groups}
    evaluation_by_group = {group["group_id"]: group for group in evaluation_groups}
    if len(proposer_by_group) != 4 or set(proposer_by_group) != set(evaluation_by_group):
        raise ValueError("group identity mismatch")

    visible_items: dict[str, dict[str, Any]] = {}
    development_ids: set[str] = set()
    evaluation_ids: set[str] = set()
    rubric_hashes: set[str] = set()
    for group_id in sorted(proposer_by_group):
        proposed = proposer_by_group[group_id]
        evaluated = evaluation_by_group[group_id]
        if proposed["target_adjacent_boundary"] != evaluated["target_adjacent_boundary"]:
            raise ValueError("boundary differs across packets")
        if canonical_json(proposed["original_rubric"]) != canonical_json(evaluated["original_rubric"]):
            raise ValueError("rubric differs across packets")
        rubric_hashes.add(rubric_key({"rubric": proposed["original_rubric"]}))
        development = proposed.get("clarification_development_items", [])
        formal = evaluated.get("evaluation_items", [])
        if len(development) != 2 or len(formal) != 4:
            raise ValueError("group item count differs")
        for item in development:
            presentation_id = item["presentation_id"]
            if presentation_id in visible_items:
                raise ValueError("duplicate presentation ID")
            visible_items[presentation_id] = item
            development_ids.add(presentation_id)
        for item in formal:
            presentation_id = item["presentation_id"]
            if presentation_id in visible_items:
                raise ValueError("duplicate presentation ID")
            visible_items[presentation_id] = item
            evaluation_ids.add(presentation_id)
    if len(rubric_hashes) != 4 or len(development_ids) != 8 or len(evaluation_ids) != 16:
        raise ValueError("rubric or visible item totals differ")

    train_rows = read_jsonl(train_path)
    bank_rows = read_jsonl(bank_path)
    train_by_id = {str(row["record_id"]): row for row in train_rows}
    events_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in bank_rows:
        events_by_id[str(event["record_id"])].append(event)
    key_rows = answer_key.get("rows", [])
    if len(key_rows) != 24:
        raise ValueError("private answer-key total differs")
    if len({row["presentation_id"] for row in key_rows}) != 24:
        raise ValueError("duplicate answer-key presentation ID")
    if len({row["record_id"] for row in key_rows}) != 24:
        raise ValueError("development/evaluation records overlap")
    if {row["presentation_id"] for row in key_rows} != set(visible_items):
        raise ValueError("visible and private item sets differ")

    crossing_eval = 0
    anchors = 0
    for key_row in key_rows:
        record_id = str(key_row["record_id"])
        train_row = train_by_id[record_id]
        events = events_by_id[record_id]
        if len(events) != 3:
            raise ValueError("selected record lacks three frozen seed events")
        lower = int(key_row["group_id"].split("-")[-1][0])
        expected = {
            "gold_label": int(train_row["label_5"]),
            "human_scores": [
                float(train_row[field])
                for field in ("human_1_5", "human_2_5", "human_3_5")
            ],
            "generated_scores": [int(event["generated_score"]) for event in events],
            "boundary_crossing_seed_count": crossing_count(events, lower),
        }
        for field, value in expected.items():
            if key_row[field] != value:
                raise ValueError(f"answer-key {field} differs")
        visible = visible_items[key_row["presentation_id"]]
        expected_visible = {
            "presentation_id": key_row["presentation_id"],
            "language": train_row["language"],
            "metric": train_row["metric_canonical"],
            "question": train_row["question"],
            "answer": train_row["answer"],
        }
        if visible != expected_visible:
            raise ValueError("visible item differs from locked train row")
        group = proposer_by_group[key_row["group_id"]]
        if rubric_key(train_row) != rubric_key({"rubric": group["original_rubric"]}):
            raise ValueError("selected row does not belong to group rubric")
        if key_row["selection_role"] == "boundary_crossing":
            crossing_eval += 1
            if expected["boundary_crossing_seed_count"] < 2:
                raise ValueError("evaluation crossing is not recurrent")
        elif key_row["selection_role"] == "clear_anchor":
            anchors += 1
            if not clear_anchor(train_row, events):
                raise ValueError("evaluation anchor is not clear")
        elif key_row["selection_role"] == "clarification_development":
            if expected["boundary_crossing_seed_count"] < 2:
                raise ValueError("development crossing is not recurrent")
        else:
            raise ValueError("unknown selection role")
    if crossing_eval != 12 or anchors != 4:
        raise ValueError("evaluation role totals differ")

    if [group["target_adjacent_boundary"][0] for group in proposer_groups] != list(BOUNDARIES):
        raise ValueError("adjacent boundary order differs")
    return {
        "status": "LLM_JUDGE_SPEC_PILOT_PACKET_AUDIT_PASS",
        "rubric_groups": 4,
        "development_items": 8,
        "evaluation_items": 16,
        "hidden_key_leakage": False,
        "record_overlap": False,
        "dev_accessed": False,
        "test_accessed": False,
        "judge_calls_started": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--failure-bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--private-dir", type=Path, default=DEFAULT_PRIVATE)
    parser.add_argument("--public-report", type=Path, default=DEFAULT_PUBLIC_REPORT)
    args = parser.parse_args()
    result = audit_packets(
        train_path=args.train,
        bank_path=args.failure_bank,
        private_dir=args.private_dir,
        public_report_path=args.public_report,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
