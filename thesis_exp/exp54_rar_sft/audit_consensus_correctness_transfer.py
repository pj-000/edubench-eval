"""Audit GPT-panel correctness and R3-to-GPT error transfer on frozen train items."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "exp54-train-only-consensus-correctness-transfer-audit-v1"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signed(value: int) -> int:
    return (value > 0) - (value < 0)


def human_class(scores: list[float]) -> str:
    unique = sorted(set(scores))
    if len(unique) == 1:
        return "H0"
    if len(unique) == 2 and unique[1] - unique[0] == 1:
        return "H1"
    if max(scores) - min(scores) >= 2:
        return "H2"
    raise ValueError("human scores do not match frozen classes")


def panel_summary(scores: list[int], label: int) -> dict[str, Any]:
    if len(scores) != 5 or any(type(score) is not int or not 1 <= score <= 5 for score in scores):
        raise ValueError("panel must contain five integer scores in 1..5")
    counts = Counter(scores)
    top = max(counts.values())
    modes = sorted(score for score, count in counts.items() if count == top)
    mode = modes[0] if len(modes) == 1 else None
    share = top / 5
    consensus_error = share >= 0.8 and mode is not None and mode != label
    error = None if mode is None else mode - label
    return {
        "scores": scores,
        "mode": mode,
        "mode_share": share,
        "agreement_class": "P5" if share == 1 else "P4" if share >= 0.8 else "Pweak",
        "correct": mode == label,
        "consensus_error": consensus_error,
        "severe_consensus_error": consensus_error and abs(error) >= 2,
        "severe_direction": "L2H" if error is not None and error >= 2 else "H2L" if error is not None and error <= -2 else None,
    }


def crosses_boundary(label: int, prediction: int, lower: int) -> bool:
    return (label <= lower < prediction) or (prediction <= lower < label)


def audit(*, answer_key: Path, response_dir: Path, private_output: Path, public_output: Path) -> dict[str, Any]:
    key_rows = _read(answer_key)["rows"]
    formal = [row for row in key_rows if row["selection_role"] != "clarification_development"]
    if len(formal) != 16 or len({row["presentation_id"] for row in formal}) != 16:
        raise ValueError("expected 16 unique formal items")
    responses: dict[str, dict[str, list[int]]] = {"original": defaultdict(list), "clarified": defaultdict(list)}
    response_hashes = {}
    for condition in responses:
        for judge in range(1, 6):
            path = response_dir / f"response_{condition}-J{judge}.json"
            payload = _read(path)
            if payload["panel_instance_id"] != f"{condition}-J{judge}":
                raise ValueError("panel identity differs")
            for result in payload["results"]:
                responses[condition][result["presentation_id"]].append(result["score"])
            response_hashes[path.name] = _sha(path)

    rows = []
    for source in sorted(formal, key=lambda row: row["presentation_id"]):
        item_id = source["presentation_id"]
        label = int(source["gold_label"])
        original = panel_summary(responses["original"][item_id], label)
        clarified = panel_summary(responses["clarified"][item_id], label)
        lower = int(source["group_id"].split("-")[-1][0])
        r3 = [int(score) for score in source["generated_scores"]]
        mode = original["mode"]
        seed_transfer = [
            mode is not None
            and score != label
            and mode != label
            and signed(score - label) == signed(mode - label)
            for score in r3
        ]
        boundary_transfer = [
            transfer
            and crosses_boundary(label, score, lower)
            and crosses_boundary(label, mode, lower)
            for score, transfer in zip(r3, seed_transfer)
        ]
        human_scores = [float(score) for score in source["human_scores"]]
        rows.append({
            "item_id": item_id,
            "rubric_boundary": f"{lower}-{lower + 1}",
            "selection_role": source["selection_role"],
            "human_rater_scores": human_scores,
            "human_ambiguity_class": human_class(human_scores),
            "label_5": label,
            "R3_seed42_43_44": r3,
            "P1_seed42_43_44": None,
            "P1_available": False,
            "original": original,
            "clarified": clarified,
            "R3_to_GPT_direction_transfer_by_seed": seed_transfer,
            "R3_to_GPT_boundary_transfer_by_seed": boundary_transfer,
            "clarification_fix": original["mode"] != label and clarified["mode"] == label,
            "clarification_harm": original["mode"] == label and clarified["mode"] != label,
            "GPT_original_mode_in_human_score_set": original["mode"] in set(human_scores) if original["mode"] is not None else False,
        })

    recurrent = [row for row in rows if row["selection_role"] == "boundary_crossing"]
    anchors = [row for row in rows if row["selection_role"] == "clear_anchor"]
    original_errors = [row for row in recurrent if row["original"]["consensus_error"]]
    persistent = [row for row in original_errors if row["clarified"]["consensus_error"]]
    common_mode_gate = (
        len(original_errors) >= 4
        and len(persistent) >= 3
        and len({row["rubric_boundary"] for row in original_errors}) >= 2
        and (sum(row["human_ambiguity_class"] == "H0" for row in original_errors) >= 3 or not any(row["human_ambiguity_class"] == "H2" for row in original_errors))
        and sum(row["original"]["severe_consensus_error"] for row in original_errors) >= 2
    )
    fixed_by_gpt = sum(row["original"]["correct"] for row in recurrent)
    anchors_correct = all(row["original"]["correct"] for row in anchors)
    stop_general = fixed_by_gpt >= 10 and len(original_errors) <= 1 and anchors_correct
    random_instability = sum(row["original"]["agreement_class"] == "Pweak" for row in recurrent) >= 4
    h2_count = sum(row["human_ambiguity_class"] == "H2" for row in recurrent)
    in_human_set = sum(row["GPT_original_mode_in_human_score_set"] for row in recurrent)
    incorrect = [row for row in recurrent if not row["original"]["correct"]]
    incorrect_in_human_set = sum(row["GPT_original_mode_in_human_score_set"] for row in incorrect)
    human_nonunique = h2_count >= 6 or in_human_set >= 7
    fired = [name for name, value in {
        "AUTHORIZE_COMMON_MODE_ERROR_VERIFICATION_PILOT": common_mode_gate,
        "STOP_GENERAL_LLM_JUDGE_METHOD": stop_general,
        "RANDOM_INSTABILITY_ENGINEERING_ONLY": random_instability,
        "HUMAN_RATING_NONUNIQUENESS_DOMINANT": human_nonunique,
    }.items() if value]
    preregistered_decision = fired[0] if len(fired) == 1 else "MULTIPLE_BRANCHES_REQUIRE_INTERPRETATION" if fired else "NO_DIRECTION_SELECTED_MORE_EVIDENCE_NEEDED"
    # The frozen human-set rule counts correct and incorrect modes together. A
    # correct mode normally belongs to the human-score set and therefore cannot
    # establish that an observed error is rating-indeterminate. Preserve the
    # preregistered branch result, but do not treat it as actionable unless the
    # error-only diagnostic also supports the interpretation.
    human_nonunique_error_support = h2_count >= 6 or (
        bool(incorrect) and incorrect_in_human_set * 2 >= len(incorrect)
    )
    decision = (
        "NO_DIRECTION_SELECTED_MORE_EVIDENCE_NEEDED"
        if preregistered_decision == "HUMAN_RATING_NONUNIQUENESS_DOMINANT"
        and not human_nonunique_error_support
        else preregistered_decision
    )
    private = {"schema_version": SCHEMA_VERSION, "rows": rows}
    private_output.parent.mkdir(parents=True, exist_ok=True)
    private_output.write_text(json.dumps(private, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    by_boundary = {}
    for boundary in sorted({row["rubric_boundary"] for row in recurrent}):
        subset = [row for row in recurrent if row["rubric_boundary"] == boundary]
        by_boundary[boundary] = {
            "n": len(subset),
            "original_correct": sum(row["original"]["correct"] for row in subset),
            "original_consensus_errors": sum(row["original"]["consensus_error"] for row in subset),
            "original_Pweak": sum(row["original"]["agreement_class"] == "Pweak" for row in subset),
        }
    leave_one_boundary_out = {
        boundary: {
            "n": len([row for row in recurrent if row["rubric_boundary"] != boundary]),
            "original_consensus_errors": sum(row["original"]["consensus_error"] for row in recurrent if row["rubric_boundary"] != boundary),
            "original_correct": sum(row["original"]["correct"] for row in recurrent if row["rubric_boundary"] != boundary),
        }
        for boundary in by_boundary
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "preregistered_decision": preregistered_decision,
        "fired_branches": fired,
        "recurrent_failures": 12,
        "clear_anchors": 4,
        "original_panel_correct_recurrent": fixed_by_gpt,
        "original_panel_consensus_errors_recurrent": len(original_errors),
        "clarified_panel_persistent_consensus_errors_recurrent": len(persistent),
        "original_panel_severe_consensus_errors_recurrent": sum(row["original"]["severe_consensus_error"] for row in original_errors),
        "original_panel_Pweak_recurrent": sum(row["original"]["agreement_class"] == "Pweak" for row in recurrent),
        "human_classes_recurrent": dict(sorted(Counter(row["human_ambiguity_class"] for row in recurrent).items())),
        "original_mode_in_human_score_set_recurrent": in_human_set,
        "original_panel_incorrect_recurrent": len(incorrect),
        "incorrect_mode_in_human_score_set_recurrent": incorrect_in_human_set,
        "human_nonuniqueness_error_only_support": human_nonunique_error_support,
        "preregistered_human_set_gate_has_correctness_denominator_confound": True,
        "clear_anchors_original_correct": sum(row["original"]["correct"] for row in anchors),
        "clarification_fixes_recurrent": sum(row["clarification_fix"] for row in recurrent),
        "clarification_harms_recurrent": sum(row["clarification_harm"] for row in recurrent),
        "R3_to_GPT_direction_transfer_seed_events": sum(sum(row["R3_to_GPT_direction_transfer_by_seed"]) for row in recurrent),
        "R3_to_GPT_boundary_transfer_seed_events": sum(sum(row["R3_to_GPT_boundary_transfer_by_seed"]) for row in recurrent),
        "by_boundary": by_boundary,
        "leave_one_boundary_out": leave_one_boundary_out,
        "private_row_table_sha256": _sha(private_output),
        "private_answer_key_sha256": _sha(answer_key),
        "private_response_hashes": dict(sorted(response_hashes.items())),
        "P1_predictions_available": False,
        "new_model_calls": 0,
        "gpu_used": False,
        "dev_accessed": False,
        "test_accessed": False,
        "training_started": False,
    }
    public_output.parent.mkdir(parents=True, exist_ok=True)
    public_output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answer-key", required=True, type=Path)
    parser.add_argument("--response-dir", required=True, type=Path)
    parser.add_argument("--private-output", required=True, type=Path)
    parser.add_argument("--public-output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(answer_key=args.answer_key, response_dir=args.response_dir, private_output=args.private_output, public_output=args.public_output)
    print(result["decision"])


if __name__ == "__main__":
    main()
