"""Validate and summarize the independent model-agent Label-2 blind audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.build_label2_human_audit_packets import RESPONSE_FIELDS


CATEGORICAL_FIELDS = RESPONSE_FIELDS[:4]
DEFAULT_ROOT = REPO_ROOT / "thesis_exp/outputs/exp54_rar_sft/rar_v2"
DEFAULT_CONFIG = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/label2_model_agent_audit_v1.json"
)
DEFAULT_AUTOMATED = (
    DEFAULT_ROOT / "label2_identification_audit/automated_mechanism_report.json"
)
DEFAULT_PACKET_REPORT = (
    DEFAULT_ROOT / "label2_identification_audit/human_packet_build_report.json"
)
DEFAULT_PRIVATE = (
    DEFAULT_ROOT / "label2_identification_audit/private/human_audit_v1"
)
DEFAULT_OUTPUT = (
    DEFAULT_ROOT / "label2_identification_audit/model_agent_blind_audit_report.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_completed_packet(
    source: list[dict[str, str]],
    completed: list[dict[str, str]],
    *,
    allowed: set[str],
) -> None:
    if len(source) != 14 or len(completed) != 14:
        raise ValueError("each packet must contain 14 rows")
    if not source or list(source[0]) != list(completed[0]):
        raise ValueError("completed packet columns differ")
    editable = set(RESPONSE_FIELDS)
    fixed = [field for field in source[0] if field not in editable]
    for before, after in zip(source, completed, strict=True):
        if any(before[field] != after[field] for field in fixed):
            raise ValueError("completed packet changed a blinded source field")
        if any(after[field] not in allowed for field in CATEGORICAL_FIELDS):
            raise ValueError("completed packet contains an invalid categorical response")
        if not after["boundary_evidence_span"].strip():
            raise ValueError("completed packet contains an empty evidence span")
    ids = [row["presentation_id"] for row in completed]
    if len(set(ids)) != 14:
        raise ValueError("completed packet presentation IDs are not unique")


def cohen_kappa(first: list[str], second: list[str]) -> float | None:
    if len(first) != len(second) or not first:
        raise ValueError("kappa inputs differ in length or are empty")
    categories = sorted(set(first) | set(second))
    observed = sum(a == b for a, b in zip(first, second, strict=True)) / len(first)
    first_counts = Counter(first)
    second_counts = Counter(second)
    expected = sum(
        first_counts[value] * second_counts[value] for value in categories
    ) / (len(first) ** 2)
    if expected == 1.0:
        return None
    return float((observed - expected) / (1.0 - expected))


def summarize(
    *,
    config_path: Path,
    automated_path: Path,
    packet_report_path: Path,
    private_dir: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_hashes = config["expected_hashes"]
    paths = {
        "automated_mechanism_report.json": automated_path,
        "human_packet_build_report.json": packet_report_path,
        **{
            name: private_dir / name
            for name in (
                "reviewer_a_packet.csv",
                "reviewer_b_packet.csv",
                "reviewer_a_completed.csv",
                "reviewer_b_completed.csv",
                "private_answer_key.jsonl",
            )
        },
    }
    for name, path in paths.items():
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        if sha256_file(path) != expected_hashes[name]:
            raise ValueError(f"locked hash differs: {name}")

    automated = json.loads(automated_path.read_text(encoding="utf-8"))
    packet_report = json.loads(packet_report_path.read_text(encoding="utf-8"))
    if automated["test_accessed"] or automated["training_started"]:
        raise ValueError("automated audit exceeded its execution boundary")
    if automated["final_exclusive_attribution_allowed"]:
        raise ValueError("automated report unexpectedly claims final attribution")
    if packet_report["human_reviews_completed"]:
        raise ValueError("packet report unexpectedly claims completed human review")
    if any(
        gate["passes_two_seed_gate"]
        for gate in automated["p1_provisional_dominance_gate"].values()
    ):
        raise ValueError("an automated mechanism unexpectedly passes dominance")

    allowed = set(config["allowed_categorical_responses"])
    reviews_by_record: dict[str, dict[str, tuple[dict[str, str], dict[str, Any]]]] = {}
    key_by_presentation = {}
    for row in read_jsonl(paths["private_answer_key.jsonl"]):
        key = (str(row["reviewer"]), str(row["presentation_id"]))
        if key in key_by_presentation:
            raise ValueError("duplicate private answer-key presentation ID")
        key_by_presentation[key] = row
    if len(key_by_presentation) != 28:
        raise ValueError("private answer key must contain 28 reviewer rows")

    packet_validation = {}
    for reviewer in ("A", "B"):
        source = read_csv(paths[f"reviewer_{reviewer.lower()}_packet.csv"])
        completed = read_csv(paths[f"reviewer_{reviewer.lower()}_completed.csv"])
        validate_completed_packet(source, completed, allowed=allowed)
        packet_validation[reviewer] = {
            "rows": 14,
            "format_valid": True,
            "source_columns_unchanged": True,
            "categorical_values_valid": True,
            "evidence_spans_nonempty": True,
            "completed_sha256": expected_hashes[
                f"reviewer_{reviewer.lower()}_completed.csv"
            ],
        }
        for response in completed:
            key = key_by_presentation[(reviewer, response["presentation_id"])]
            reviews_by_record.setdefault(str(key["record_id"]), {})[reviewer] = (
                response,
                key,
            )
    if len(reviews_by_record) != 14 or any(
        set(reviewers) != {"A", "B"} for reviewers in reviews_by_record.values()
    ):
        raise ValueError("reviewer packets do not close over the same 14 records")

    agreement = {}
    for field in CATEGORICAL_FIELDS:
        first = [value["A"][0][field] for value in reviews_by_record.values()]
        second = [value["B"][0][field] for value in reviews_by_record.values()]
        equal = sum(a == b for a, b in zip(first, second, strict=True))
        agreement[field] = {
            "agreement_count": equal,
            "agreement_rate": equal / 14,
            "cohen_kappa": cohen_kappa(first, second),
            "reviewer_A_distribution": dict(sorted(Counter(first).items())),
            "reviewer_B_distribution": dict(sorted(Counter(second).items())),
            "both_yes": sum(
                a == b == "YES" for a, b in zip(first, second, strict=True)
            ),
        }

    rubric_consensus = 0
    boundary_consensus = 0
    score2_unique_consensus = 0
    automatic_ambiguous = 0
    boundary_among_automatic = 0
    boundary_among_nonautomatic = 0
    for reviewers in reviews_by_record.values():
        first, key = reviewers["A"]
        second, _ = reviewers["B"]
        first_rubric = (
            first["decisive_criterion_absent"] == "YES"
            or first["present_criterion_too_vague"] == "YES"
        )
        second_rubric = (
            second["decisive_criterion_absent"] == "YES"
            or second["present_criterion_too_vague"] == "YES"
        )
        first_boundary = (
            first["score_3_also_defensible"] == "YES"
            or first["score_2_uniquely_defensible"] == "NO"
        )
        second_boundary = (
            second["score_3_also_defensible"] == "YES"
            or second["score_2_uniquely_defensible"] == "NO"
        )
        consensus_boundary = first_boundary and second_boundary
        rubric_consensus += first_rubric and second_rubric
        boundary_consensus += consensus_boundary
        score2_unique_consensus += (
            first["score_2_uniquely_defensible"] == "YES"
            and second["score_2_uniquely_defensible"] == "YES"
        )
        if key["automatic_measurement_ambiguous"]:
            automatic_ambiguous += 1
            boundary_among_automatic += consensus_boundary
        else:
            boundary_among_nonautomatic += consensus_boundary

    return {
        "schema_version": "exp54-label2-model-agent-blind-audit-report-v1",
        "status": "MODEL_AGENT_AUDIT_COMPLETE_HUMAN_PROTOCOL_NOT_SATISFIED",
        "reviewer_type": config["reviewer_type"],
        "reviewer_model_identity": config["reviewer_model_identity"],
        "records": 14,
        "packet_validation": packet_validation,
        "field_agreement": agreement,
        "consensus": {
            "both_agents_flag_rubric_missing_or_vague": rubric_consensus,
            "both_agents_flag_2_3_boundary_ambiguous": boundary_consensus,
            "both_agents_say_score_2_unique": score2_unique_consensus,
            "automatic_rater_ambiguous_rows": automatic_ambiguous,
            "consensus_boundary_ambiguous_within_automatic": boundary_among_automatic,
            "nonautomatic_rows": 14 - automatic_ambiguous,
            "consensus_boundary_ambiguous_within_nonautomatic": boundary_among_nonautomatic,
        },
        "automated_mechanism_dominance_gate_passed": False,
        "credible_causal_dominant_mechanism_identified": False,
        "strongest_current_hypothesis": "observed-consensus 2/3 boundary instability and rubric executability",
        "interpretation": "model-agent evidence supports a target-boundary hypothesis but cannot be represented as human validity evidence or decisive causal attribution",
        "new_training_method_authorized": False,
        "human_audit_satisfied": False,
        "adjudication_performed": False,
        "private_row_content_published": False,
        "training_started": False,
        "gpu_used": False,
        "test_accessed": False,
        "config_sha256": sha256_file(config_path),
        "automated_report_sha256": expected_hashes[
            "automated_mechanism_report.json"
        ],
        "packet_report_sha256": expected_hashes["human_packet_build_report.json"],
        "private_answer_key_sha256": expected_hashes["private_answer_key.jsonl"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--automated-report", type=Path, default=DEFAULT_AUTOMATED)
    parser.add_argument("--packet-report", type=Path, default=DEFAULT_PACKET_REPORT)
    parser.add_argument("--private-dir", type=Path, default=DEFAULT_PRIVATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = summarize(
        config_path=args.config,
        automated_path=args.automated_report,
        packet_report_path=args.packet_report,
        private_dir=args.private_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(report["status"])


if __name__ == "__main__":
    main()
