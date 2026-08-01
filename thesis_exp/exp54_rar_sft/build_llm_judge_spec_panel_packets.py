"""Materialize blinded, order-locked judge-panel packets for the pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "exp54-llm-judge-spec-panel-packet-v1"
CONDITIONS = ("original", "clarified")
JUDGES_PER_CONDITION = 5


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_panel_packets(
    *,
    evaluation_path: Path,
    clarification_path: Path,
    audit_a_path: Path,
    audit_b_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    evaluation = _read(evaluation_path)
    clarification = _read(clarification_path)
    audits = [_read(audit_a_path), _read(audit_b_path)]
    if len(evaluation.get("groups", [])) != 4:
        raise ValueError("evaluation packet must contain four groups")
    clarification_by_group = {
        row["group_id"]: row for row in clarification.get("groups", [])
    }
    if len(clarification_by_group) != 4:
        raise ValueError("clarification file must contain four unique groups")
    for audit in audits:
        decisions = {
            row["group_id"]: row["decision"] for row in audit.get("groups", [])
        }
        if set(decisions) != set(clarification_by_group) or any(
            decision != "PASS" for decision in decisions.values()
        ):
            raise ValueError("all groups must pass each independent fidelity audit")

    base_items = []
    for group in evaluation["groups"]:
        group_id = group["group_id"]
        proposed = clarification_by_group[group_id]
        if _canonical(group["original_rubric"]) != _canonical(proposed["original_rubric"]):
            raise ValueError("clarification original rubric differs")
        if group["target_adjacent_boundary"] != proposed["target_adjacent_boundary"]:
            raise ValueError("clarification boundary differs")
        for item in group["evaluation_items"]:
            base_items.append(
                {
                    "presentation_id": item["presentation_id"],
                    "group_id": group_id,
                    "language": item["language"],
                    "metric": item["metric"],
                    "question": item["question"],
                    "answer": item["answer"],
                    "original_rubric": group["original_rubric"],
                    "clarification": proposed["clarification"],
                }
            )
    if len(base_items) != 16 or len({item["presentation_id"] for item in base_items}) != 16:
        raise ValueError("formal evaluation set must contain 16 unique items")

    packet_hashes: dict[str, str] = {}
    order_hashes: dict[str, str] = {}
    for condition in CONDITIONS:
        for judge_number in range(1, JUDGES_PER_CONDITION + 1):
            panel_id = f"{condition}-J{judge_number}"
            ordered = sorted(
                base_items,
                key=lambda item: _sha(
                    f"{SCHEMA_VERSION}|{panel_id}|{item['presentation_id']}".encode("utf-8")
                ),
            )
            visible = []
            for item in ordered:
                specification = {"original_rubric": item["original_rubric"]}
                if condition == "clarified":
                    specification["policy_preserving_clarification"] = item["clarification"]
                visible.append(
                    {
                        "presentation_id": item["presentation_id"],
                        "language": item["language"],
                        "metric": item["metric"],
                        "scoring_specification": specification,
                        "question": item["question"],
                        "answer": item["answer"],
                    }
                )
            packet = {
                "schema_version": SCHEMA_VERSION,
                "panel_instance_id": panel_id,
                "task": (
                    "Act as an independent educational-quality judge. Apply only the "
                    "provided scoring specification. For every item, output the score "
                    "first (integer 1-5), then a concise evidence-grounded reason. "
                    "Do not infer hidden target labels or consult other judgments."
                ),
                "required_output": {
                    "schema_version": "exp54-llm-judge-spec-panel-response-v1",
                    "panel_instance_id": panel_id,
                    "results": [
                        {
                            "presentation_id": "copy exactly",
                            "score": "integer 1-5",
                            "reason": "concise evidence-grounded string",
                        }
                    ],
                },
                "items": visible,
            }
            path = output_dir / f"judge_{panel_id}.json"
            _write(path, packet)
            packet_hashes[path.name] = _sha(path.read_bytes())
            order_hashes[panel_id] = _sha(
                _canonical([item["presentation_id"] for item in visible])
            )
    if len(set(order_hashes.values())) != 10:
        raise ValueError("judge item orders must be independently locked")
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "PANEL_PACKETS_BUILT_NO_JUDGE_CALLS",
        "conditions": list(CONDITIONS),
        "judge_instances_per_condition": JUDGES_PER_CONDITION,
        "evaluation_items": 16,
        "formal_decisions_expected": 160,
        "all_groups_dual_fidelity_pass": True,
        "packet_hashes": dict(sorted(packet_hashes.items())),
        "order_hashes": dict(sorted(order_hashes.items())),
        "unique_order_vectors": len(set(order_hashes.values())),
        "historical_labels_or_predictions_visible": False,
        "cross_condition_results_visible": False,
        "dev_accessed": False,
        "test_accessed": False,
        "gpu_used": False,
    }
    _write(output_dir / "panel_packet_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--clarifications", required=True, type=Path)
    parser.add_argument("--audit-a", required=True, type=Path)
    parser.add_argument("--audit-b", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    build_panel_packets(
        evaluation_path=args.evaluation,
        clarification_path=args.clarifications,
        audit_a_path=args.audit_a,
        audit_b_path=args.audit_b,
        output_dir=args.output_dir,
    )
    print("LLM_JUDGE_SPEC_PANEL_PACKETS_BUILT")


if __name__ == "__main__":
    main()
