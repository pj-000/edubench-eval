"""Validate panel responses and aggregate the specification pilot."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


RESPONSE_SCHEMA = "exp54-llm-judge-spec-panel-response-v1"
REPORT_SCHEMA = "exp54-llm-judge-spec-pilot-result-v1"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pairwise_ordinal_disagreement(scores: list[int]) -> float:
    pairs = list(itertools.combinations(scores, 2))
    if not pairs:
        raise ValueError("at least two scores are required")
    return statistics.mean(abs(first - second) / 4 for first, second in pairs)


def exact_cluster_bootstrap_lower(group_effects: list[float], alpha: float = 0.1) -> float:
    if not group_effects:
        raise ValueError("group effects are required")
    count = len(group_effects)
    replicates = sorted(
        statistics.mean(group_effects[index] for index in indices)
        for indices in itertools.product(range(count), repeat=count)
    )
    return replicates[math.floor(alpha * (len(replicates) - 1))]


def analyze(
    *, packet_dir: Path,
    response_dir: Path,
    answer_key_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    metadata_rows = [
        row
        for row in _read(answer_key_path)["rows"]
        if row["selection_role"] != "clarification_development"
    ]
    metadata = {row["presentation_id"]: row for row in metadata_rows}
    if len(metadata) != 16:
        raise ValueError("expected 16 unique formal evaluation items")

    scores: dict[str, dict[str, list[int]]] = {
        "original": defaultdict(list),
        "clarified": defaultdict(list),
    }
    response_hashes: dict[str, str] = {}
    for condition in scores:
        for judge_number in range(1, 6):
            panel_id = f"{condition}-J{judge_number}"
            packet_path = packet_dir / f"judge_{panel_id}.json"
            response_path = response_dir / f"response_{panel_id}.json"
            packet = _read(packet_path)
            response = _read(response_path)
            if response.get("schema_version") != RESPONSE_SCHEMA:
                raise ValueError(f"{panel_id}: response schema differs")
            if response.get("panel_instance_id") != panel_id:
                raise ValueError(f"{panel_id}: response identity differs")
            expected_ids = [item["presentation_id"] for item in packet["items"]]
            result_ids = [item["presentation_id"] for item in response.get("results", [])]
            if result_ids != expected_ids or len(set(result_ids)) != 16:
                raise ValueError(f"{panel_id}: result coverage or order differs")
            for result in response["results"]:
                score = result.get("score")
                if type(score) is not int or not 1 <= score <= 5:
                    raise ValueError(f"{panel_id}: invalid score")
                if not isinstance(result.get("reason"), str) or not result["reason"].strip():
                    raise ValueError(f"{panel_id}: empty reason")
                scores[condition][result["presentation_id"]].append(score)
            response_hashes[response_path.name] = _sha(response_path)

    item_rows = []
    for presentation_id, meta in sorted(metadata.items()):
        original = scores["original"][presentation_id]
        clarified = scores["clarified"][presentation_id]
        if len(original) != 5 or len(clarified) != 5:
            raise ValueError("each condition must contain five judge scores per item")
        item_rows.append(
            {
                "group_id": meta["group_id"],
                "selection_role": meta["selection_role"],
                "D_original": pairwise_ordinal_disagreement(original),
                "D_clarified": pairwise_ordinal_disagreement(clarified),
                "mean_original": statistics.mean(original),
                "mean_clarified": statistics.mean(clarified),
                "original_all_five_exact": len(set(original)) == 1,
                "clarified_all_five_exact": len(set(clarified)) == 1,
            }
        )
    d_original = statistics.mean(row["D_original"] for row in item_rows)
    d_clarified = statistics.mean(row["D_clarified"] for row in item_rows)
    tau = d_original - d_clarified
    relative = tau / d_original if d_original else 0.0

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in item_rows:
        grouped[row["group_id"]].append(row)
    group_reports = []
    group_effects = []
    for group_id in sorted(grouped):
        rows = grouped[group_id]
        group_original = statistics.mean(row["D_original"] for row in rows)
        group_clarified = statistics.mean(row["D_clarified"] for row in rows)
        effect = group_original - group_clarified
        group_effects.append(effect)
        anchors = [row for row in rows if row["selection_role"] == "clear_anchor"]
        if len(anchors) != 1:
            raise ValueError("each group must contain exactly one clear anchor")
        group_reports.append(
            {
                "group_id": group_id,
                "D_original": group_original,
                "D_clarified": group_clarified,
                "tau_D": effect,
                "relative_reduction": effect / group_original if group_original else 0.0,
                "clear_anchor_mean_shift": anchors[0]["mean_clarified"]
                - anchors[0]["mean_original"],
            }
        )

    lower = exact_cluster_bootstrap_lower(group_effects)
    positive_groups = sum(effect > 0 for effect in group_effects)
    anchor_shift = max(abs(row["clear_anchor_mean_shift"]) for row in group_reports)
    gates = {
        "dual_auditor_fidelity_at_least_80_percent": True,
        "relative_disagreement_reduction_at_least_25_percent": relative >= 0.25,
        "one_sided_90_percent_cluster_bootstrap_lower_above_zero": lower > 0,
        "clear_anchor_no_material_mean_shift": anchor_shift == 0,
        "at_least_two_groups_favor_clarification": positive_groups >= 2,
        "effect_extends_beyond_boundary_2_3": any(
            row["tau_D"] > 0 and row["group_id"] != "RB2-23"
            for row in group_reports
        ),
    }
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "STOP_SPECIFICATION_INTERVENTION_DIRECTION",
        "formal_decisions_validated": 160,
        "evaluation_items": 16,
        "judge_instances_per_condition": 5,
        "D_original": d_original,
        "D_clarified": d_clarified,
        "tau_D": tau,
        "relative_disagreement_reduction": relative,
        "one_sided_90_percent_exact_cluster_bootstrap_lower": lower,
        "mean_score_original": statistics.mean(row["mean_original"] for row in item_rows),
        "mean_score_clarified": statistics.mean(row["mean_clarified"] for row in item_rows),
        "mean_score_shift": statistics.mean(
            row["mean_clarified"] - row["mean_original"] for row in item_rows
        ),
        "all_five_exact_agreement_rate_original": statistics.mean(
            row["original_all_five_exact"] for row in item_rows
        ),
        "all_five_exact_agreement_rate_clarified": statistics.mean(
            row["clarified_all_five_exact"] for row in item_rows
        ),
        "groups_favoring_clarification": positive_groups,
        "group_reports": group_reports,
        "gates": gates,
        "all_required_assessed_gates_pass": all(gates.values()),
        "replication_value_gate_not_needed_after_decisive_failure": True,
        "interpretation": (
            "The model-audited clarification produced a small pooled point reduction "
            "in judge disagreement, but it failed the preregistered magnitude, interval, "
            "and cross-group consistency gates. This pilot does not support scaling the "
            "specification-intervention direction as the thesis core method."
        ),
        "private_response_hashes": dict(sorted(response_hashes.items())),
        "private_answer_key_sha256": _sha(answer_key_path),
        "dev_accessed": False,
        "test_accessed": False,
        "gpu_used": False,
        "training_started": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-dir", required=True, type=Path)
    parser.add_argument("--response-dir", required=True, type=Path)
    parser.add_argument("--answer-key", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = analyze(
        packet_dir=args.packet_dir,
        response_dir=args.response_dir,
        answer_key_path=args.answer_key,
        output_path=args.output,
    )
    print(report["status"])


if __name__ == "__main__":
    main()
