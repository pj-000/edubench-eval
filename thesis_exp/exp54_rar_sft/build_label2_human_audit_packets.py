"""Build deterministic, model-blind packets for the Label-2 human audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT


DEV_SHA256 = "a18d6a27b9a524d4592a359658ae70c9348fe88e43c962971ba95f62d2b6cdf0"
REVIEWERS = ("A", "B")
RESPONSE_FIELDS = (
    "score_2_uniquely_defensible",
    "score_3_also_defensible",
    "decisive_criterion_absent",
    "present_criterion_too_vague",
    "boundary_evidence_span",
    "reviewer_notes",
)
ALLOWED_CATEGORICAL_RESPONSES = ("YES", "NO", "UNCERTAIN")
DEFAULT_DEV = REPO_ROOT / "thesis_exp/data/splits/paper_like_triple_seed42/dev.jsonl"
DEFAULT_PRIVATE_DIR = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/label2_identification_audit/private/human_audit_v1"
)
DEFAULT_PUBLIC_REPORT = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/label2_identification_audit/human_packet_build_report.json"
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _digest(namespace: str, record_id: str) -> str:
    return sha256_bytes(f"{namespace}|{record_id}".encode("utf-8"))


def _packet_rows(
    source_rows: list[dict[str, Any]], reviewer: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(
        source_rows,
        key=lambda row: _digest(f"exp54-label2-reviewer-{reviewer}-order", row["record_id"]),
    )
    packet_rows = []
    answer_rows = []
    for case_number, row in enumerate(ordered, 1):
        presentation_id = (
            f"L2-{reviewer}-"
            + _digest(f"exp54-label2-reviewer-{reviewer}-id", row["record_id"])[
                :16
            ]
        )
        packet = {
            "case_number": case_number,
            "presentation_id": presentation_id,
            "language": row["language"],
            "metric": row["metric_canonical"],
            "question": row["question"],
            "answer": row["answer"],
            "rubric": row["rubric"],
            **{field: "" for field in RESPONSE_FIELDS},
        }
        packet_rows.append(packet)
        rater_scores = [
            float(row[field]) for field in ("human_1_5", "human_2_5", "human_3_5")
        ]
        answer_rows.append(
            {
                "reviewer": reviewer,
                "case_number": case_number,
                "presentation_id": presentation_id,
                "record_id": row["record_id"],
                "question_key": row["question_key"],
                "metric_id": row["metric_id"],
                "language": row["language"],
                "label_5": int(row["label_5"]),
                "human_1_5": rater_scores[0],
                "human_2_5": rater_scores[1],
                "human_3_5": rater_scores[2],
                "automatic_measurement_ambiguous": bool(
                    max(rater_scores) - min(rater_scores) >= 2.0
                    or {2.0, 3.0}.issubset(set(rater_scores))
                ),
            }
        )
    return packet_rows, answer_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty review packet")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    )
    path.write_text(payload, encoding="utf-8")


def build_packets(
    *, dev_path: Path, private_dir: Path, public_report_path: Path
) -> dict[str, Any]:
    if not dev_path.is_file() or dev_path.is_symlink():
        raise FileNotFoundError(dev_path)
    if sha256_file(dev_path) != DEV_SHA256:
        raise ValueError("locked dev hash differs")
    dev_rows = read_jsonl(dev_path)
    if len(dev_rows) != 664:
        raise ValueError("locked dev row count differs")
    selected = [row for row in dev_rows if int(row["label_5"]) == 2]
    if len(selected) != 14 or len({row["record_id"] for row in selected}) != 14:
        raise ValueError("expected 14 unique Label-2 dev records")

    report_packets: dict[str, Any] = {}
    all_answer_rows = []
    for reviewer in REVIEWERS:
        packet_rows, answer_rows = _packet_rows(selected, reviewer)
        packet_path = private_dir / f"reviewer_{reviewer.lower()}_packet.csv"
        _write_csv(packet_path, packet_rows)
        all_answer_rows.extend(answer_rows)
        report_packets[reviewer] = {
            "rows": len(packet_rows),
            "sha256": sha256_file(packet_path),
            "packet_fields": list(packet_rows[0]),
            "labels_raters_model_outputs_excluded": True,
        }

    answer_key_path = private_dir / "private_answer_key.jsonl"
    _write_jsonl(answer_key_path, all_answer_rows)
    report = {
        "schema_version": "exp54-label2-human-packet-build-v1",
        "status": "PRIMARY_PACKETS_BUILT_AWAITING_TWO_HUMAN_REVIEWS",
        "population": "all locked dev rows with observed-consensus label_5 == 2",
        "unique_records": len(selected),
        "question_groups": len({row["question_key"] for row in selected}),
        "reviewers_required": 2,
        "independent_first_pass_required": True,
        "adjudication_required_for_disagreement": True,
        "allowed_categorical_responses": list(ALLOWED_CATEGORICAL_RESPONSES),
        "response_fields": list(RESPONSE_FIELDS),
        "packets": report_packets,
        "reviewer_orders_differ": [
            row["record_id"] for row in _packet_rows(selected, "A")[1]
        ]
        != [row["record_id"] for row in _packet_rows(selected, "B")[1]],
        "private_answer_key_rows": len(all_answer_rows),
        "private_answer_key_sha256": sha256_file(answer_key_path),
        "private_row_content_published": False,
        "human_reviews_completed": False,
        "model_or_llm_used_as_human_reviewer": False,
        "training_started": False,
        "test_accessed": False,
        "dev_sha256": DEV_SHA256,
    }
    public_report_path.parent.mkdir(parents=True, exist_ok=True)
    public_report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--private-dir", type=Path, default=DEFAULT_PRIVATE_DIR)
    parser.add_argument("--public-report", type=Path, default=DEFAULT_PUBLIC_REPORT)
    args = parser.parse_args()
    build_packets(
        dev_path=args.dev,
        private_dir=args.private_dir,
        public_report_path=args.public_report,
    )
    print("LABEL2_PRIMARY_HUMAN_PACKETS_BUILT")


if __name__ == "__main__":
    main()
