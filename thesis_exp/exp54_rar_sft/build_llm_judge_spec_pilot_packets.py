"""Build deterministic private packets for the LLM-judge specification pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT


SCHEMA_VERSION = "exp54-llm-judge-spec-pilot-packets-v1"
TRAIN_SHA256 = "0a1733b9209984c5c4291d205d1ac6057bed341717903b9de075d07de44a878e"
FAILURE_BANK_SHA256 = "d5d452017d8f6bb8b19e89c278a3e714c488d2211e266cde4e72dd5cf1c26757"
BOUNDARIES = (1, 2, 3, 4)
DEFAULT_ROOT = REPO_ROOT / "thesis_exp/outputs/exp54_rar_sft/rar_v2"
DEFAULT_TRAIN = REPO_ROOT / "thesis_exp/data/splits/paper_like_triple_seed42/train.jsonl"
DEFAULT_BANK = DEFAULT_ROOT / "actual_failure_bank/private/actual_failure_bank.jsonl"
DEFAULT_PRIVATE = DEFAULT_ROOT / "llm_judge_spec_pilot/private"
DEFAULT_PUBLIC_REPORT = DEFAULT_ROOT / "llm_judge_spec_pilot/packet_build_report.json"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def rubric_key(row: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(row["rubric"]))


def deterministic_order(namespace: str, record_id: str) -> str:
    return sha256_bytes(f"{namespace}|{record_id}".encode("utf-8"))


def crossing_count(events: list[dict[str, Any]], lower: int) -> int:
    gold = int(events[0]["gold_label"])
    return sum(
        (gold <= lower < int(event["generated_score"]))
        or (int(event["generated_score"]) <= lower < gold)
        for event in events
    )


def clear_anchor(train_row: dict[str, Any], events: list[dict[str, Any]]) -> bool:
    gold = int(train_row["label_5"])
    human = [
        float(train_row[field])
        for field in ("human_1_5", "human_2_5", "human_3_5")
    ]
    return max(human) == min(human) and all(
        int(event["generated_score"]) == gold for event in events
    )


def select_groups(
    train_rows: list[dict[str, Any]], bank_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    train_by_id = {str(row["record_id"]): row for row in train_rows}
    events_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in bank_rows:
        events_by_id[str(event["record_id"])].append(event)
    if set(train_by_id) != set(events_by_id):
        raise ValueError("train and failure-bank record sets differ")
    if any(len(events) != 3 for events in events_by_id.values()):
        raise ValueError("each train record must have exactly three seed events")

    used_rubrics: set[str] = set()
    selected = []
    for lower in BOUNDARIES:
        candidates: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"recurrent": [], "crossing": [], "anchors": [], "rows": []}
        )
        for record_id, train_row in train_by_id.items():
            key = rubric_key(train_row)
            events = events_by_id[record_id]
            candidates[key]["rows"].append(record_id)
            crosses = crossing_count(events, lower)
            if crosses:
                candidates[key]["crossing"].append(record_id)
            if crosses >= 2:
                candidates[key]["recurrent"].append(record_id)
            if clear_anchor(train_row, events):
                candidates[key]["anchors"].append(record_id)

        eligible = [
            (key, value)
            for key, value in candidates.items()
            if key not in used_rubrics
            and len(value["recurrent"]) >= 5
            and len(value["anchors"]) >= 1
        ]
        if not eligible:
            raise ValueError(f"boundary {lower}-{lower + 1} has no eligible rubric")
        eligible.sort(
            key=lambda item: (
                -len(item[1]["recurrent"]),
                -len(item[1]["crossing"]),
                -len(item[1]["anchors"]),
                item[0],
            )
        )
        key, pool = eligible[0]
        used_rubrics.add(key)
        recurrent = sorted(
            pool["recurrent"],
            key=lambda record_id: deterministic_order(
                f"{SCHEMA_VERSION}|boundary-{lower}|recurrent", record_id
            ),
        )
        anchors = sorted(
            [record_id for record_id in pool["anchors"] if record_id not in recurrent[:5]],
            key=lambda record_id: deterministic_order(
                f"{SCHEMA_VERSION}|boundary-{lower}|anchor", record_id
            ),
        )
        if not anchors:
            raise ValueError(f"boundary {lower}-{lower + 1} has no disjoint anchor")
        selected.append(
            {
                "lower": lower,
                "rubric_key": key,
                "development_ids": recurrent[:2],
                "crossing_evaluation_ids": recurrent[2:5],
                "anchor_evaluation_id": anchors[0],
                "pool_counts": {
                    "all_records": len(pool["rows"]),
                    "crossing_records": len(pool["crossing"]),
                    "recurrent_crossing_records": len(pool["recurrent"]),
                    "clear_anchor_records": len(pool["anchors"]),
                },
            }
        )
    return selected


def _visible_item(row: dict[str, Any], presentation_id: str) -> dict[str, Any]:
    return {
        "presentation_id": presentation_id,
        "language": row["language"],
        "metric": row["metric_canonical"],
        "question": row["question"],
        "answer": row["answer"],
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_packets(
    *,
    train_path: Path,
    bank_path: Path,
    private_dir: Path,
    public_report_path: Path,
) -> dict[str, Any]:
    expected = {train_path: TRAIN_SHA256, bank_path: FAILURE_BANK_SHA256}
    for path, digest in expected.items():
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        if sha256_file(path) != digest:
            raise ValueError(f"locked input hash differs: {path}")
    train_rows = read_jsonl(train_path)
    bank_rows = read_jsonl(bank_path)
    if len(train_rows) != 2654 or len(bank_rows) != 7962:
        raise ValueError("locked input row count differs")
    train_by_id = {str(row["record_id"]): row for row in train_rows}
    events_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in bank_rows:
        events_by_id[str(event["record_id"])].append(event)
    groups = select_groups(train_rows, bank_rows)

    proposer_groups = []
    evaluation_groups = []
    answer_key = []
    for group_index, group in enumerate(groups, 1):
        lower = int(group["lower"])
        group_id = f"RB{group_index}-{lower}{lower + 1}"
        reference_row = train_by_id[group["development_ids"][0]]
        development = []
        for item_index, record_id in enumerate(group["development_ids"], 1):
            presentation_id = f"{group_id}-D{item_index}"
            development.append(_visible_item(train_by_id[record_id], presentation_id))
            answer_key.append(
                _private_key_row(
                    group_id, presentation_id, "clarification_development", record_id,
                    train_by_id[record_id], events_by_id[record_id], lower
                )
            )
        evaluation = []
        evaluation_ids = [
            *(('boundary_crossing', record_id) for record_id in group["crossing_evaluation_ids"]),
            ("clear_anchor", group["anchor_evaluation_id"]),
        ]
        evaluation_ids.sort(
            key=lambda item: deterministic_order(
                f"{SCHEMA_VERSION}|{group_id}|evaluation-order", item[1]
            )
        )
        for item_index, (selection_role, record_id) in enumerate(evaluation_ids, 1):
            presentation_id = f"{group_id}-E{item_index}"
            evaluation.append(_visible_item(train_by_id[record_id], presentation_id))
            answer_key.append(
                _private_key_row(
                    group_id, presentation_id, selection_role, record_id,
                    train_by_id[record_id], events_by_id[record_id], lower
                )
            )
        proposer_groups.append(
            {
                "group_id": group_id,
                "target_adjacent_boundary": [lower, lower + 1],
                "original_rubric": reference_row["rubric"],
                "clarification_development_items": development,
            }
        )
        evaluation_groups.append(
            {
                "group_id": group_id,
                "target_adjacent_boundary": [lower, lower + 1],
                "original_rubric": reference_row["rubric"],
                "evaluation_items": evaluation,
            }
        )

    proposer_path = private_dir / "clarification_proposer_packet.json"
    evaluation_path = private_dir / "evaluation_source_packet.json"
    answer_key_path = private_dir / "private_answer_key.json"
    _write_json(
        proposer_path,
        {"schema_version": SCHEMA_VERSION, "groups": proposer_groups},
    )
    _write_json(
        evaluation_path,
        {"schema_version": SCHEMA_VERSION, "groups": evaluation_groups},
    )
    _write_json(
        answer_key_path,
        {"schema_version": SCHEMA_VERSION, "rows": answer_key},
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "TRAIN_ONLY_PACKETS_BUILT_NO_JUDGE_CALLS",
        "selected_groups": [
            {
                "boundary": f"{group['lower']}-{group['lower'] + 1}",
                "development_items": 2,
                "crossing_evaluation_items": 3,
                "clear_anchor_evaluation_items": 1,
                "pool_counts": group["pool_counts"],
            }
            for group in groups
        ],
        "rubric_groups": len(groups),
        "unique_rubrics": len({group["rubric_key"] for group in groups}),
        "development_items": 8,
        "evaluation_items": 16,
        "private_answer_key_rows": len(answer_key),
        "private_artifact_hashes": {
            "clarification_proposer_packet.json": sha256_file(proposer_path),
            "evaluation_source_packet.json": sha256_file(evaluation_path),
            "private_answer_key.json": sha256_file(answer_key_path),
        },
        "visible_proposer_contains_evaluation_items": False,
        "visible_packets_contain_historical_labels_or_predictions": False,
        "private_row_content_published": False,
        "judge_calls_started": False,
        "training_started": False,
        "gpu_used": False,
        "dev_accessed": False,
        "test_accessed": False,
        "train_sha256": TRAIN_SHA256,
        "failure_bank_sha256": FAILURE_BANK_SHA256,
        "failure_bank_error_class_counts": dict(
            sorted(Counter(str(row["error_class"]) for row in bank_rows).items())
        ),
    }
    _write_json(public_report_path, report)
    return report


def _private_key_row(
    group_id: str,
    presentation_id: str,
    selection_role: str,
    record_id: str,
    train_row: dict[str, Any],
    events: list[dict[str, Any]],
    lower: int,
) -> dict[str, Any]:
    return {
        "group_id": group_id,
        "presentation_id": presentation_id,
        "selection_role": selection_role,
        "record_id": record_id,
        "question_key": train_row["question_key"],
        "metric_id": train_row["metric_id"],
        "language": train_row["language"],
        "gold_label": int(train_row["label_5"]),
        "human_scores": [
            float(train_row[field])
            for field in ("human_1_5", "human_2_5", "human_3_5")
        ],
        "generated_scores": [int(event["generated_score"]) for event in events],
        "boundary_crossing_seed_count": crossing_count(events, lower),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--failure-bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--private-dir", type=Path, default=DEFAULT_PRIVATE)
    parser.add_argument("--public-report", type=Path, default=DEFAULT_PUBLIC_REPORT)
    args = parser.parse_args()
    build_packets(
        train_path=args.train,
        bank_path=args.failure_bank,
        private_dir=args.private_dir,
        public_report_path=args.public_report,
    )
    print("LLM_JUDGE_SPEC_PILOT_PACKETS_BUILT")


if __name__ == "__main__":
    main()
