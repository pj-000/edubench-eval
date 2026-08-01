"""Collect the two train-only score-blind evaluator families for P3."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.actual_failure_bank import (
    read_jsonl,
    sha256_file,
)
from thesis_exp.exp54_rar_sft.audit_sorc_rationale_qualification import (
    validate_result_identity,
)
from thesis_exp.exp54_rar_sft.train_sorc_dpo_smoke import (
    FINAL_QUALIFICATION_SCHEMA_VERSION,
    FINAL_QUALIFICATION_STATUS,
)


ROOT = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/"
    "preference_pairs/rationale_qualification"
)
PRIVATE = ROOT / "private"
CANDIDATE_LOCK = ROOT / "candidate_lock.json"
ANSWER_KEY = PRIVATE / "answer_key.jsonl"
TASKS = PRIVATE / "score_blind_tasks.jsonl"
SCHEMA = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/schemas/"
    "rationale_audit_score_blind_v2.schema.json"
)
API_RUNNER = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/"
    "run_sorc_rationale_qualification_api.py"
)
DEFAULT_RESULTS = {
    "qwen": (
        PRIVATE
        / "evaluator_results/qwen_qwen3.7-max_score_blind.jsonl"
    ),
    "deepseek": (
        PRIVATE
        / "evaluator_results/deepseek_deepseek-v4-pro_score_blind.jsonl"
    ),
}
FINAL_REPORT = ROOT / "final_report.json"
FINAL_LOCK = ROOT / "final_lock.json"
EXPECTED_PAIRS = 120
EXPECTED_PRESENTATIONS = 240


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _validate_judgment(
    *,
    result: dict[str, Any],
    task: dict[str, Any],
    key: dict[str, Any],
    family: str,
    model: str,
    schema: dict[str, Any],
) -> str:
    validate_result_identity(result=result, task=task, answer_key=key)
    if (
        result.get("schema_version")
        != "exp54-sorc-rationale-qualification-evaluator-result-v1"
        or result.get("evaluator_family") != family
        or result.get("model") != model
    ):
        raise ValueError(f"{family}: evaluator result contract differs")
    parsed = result["parsed_judgment"]
    required = set(schema["required"])
    if set(parsed) != required | {"stage"}:
        raise ValueError(f"{family}: parsed judgment fields differ")
    if parsed.get("stage") != "score_blind":
        raise ValueError(f"{family}: parsed stage differs")
    for name, field in schema["properties"].items():
        value = parsed[name]
        if field.get("type") == "string" and not isinstance(value, str):
            raise ValueError(f"{family}: {name} type differs")
        if "enum" in field and value not in field["enum"]:
            raise ValueError(f"{family}: {name} enum differs")
        if "maxLength" in field and len(value) > int(field["maxLength"]):
            raise ValueError(f"{family}: {name} is too long")
        if "minLength" in field and len(value) < int(field["minLength"]):
            raise ValueError(f"{family}: {name} is too short")
    preference = str(parsed["overall_preference"])
    if preference == "tie":
        return "TIE"
    source = key["a_source"] if preference == "A" else key["b_source"]
    if source not in {"R3_ALIGNED", "R2_SHUFFLED"}:
        raise ValueError(f"{family}: answer-key source differs")
    return str(source)


def _family_decision(
    *,
    family: str,
    model: str,
    result_path: Path,
    tasks: dict[str, dict[str, Any]],
    keys: dict[str, dict[str, Any]],
    schema: dict[str, Any],
) -> dict[str, Any]:
    rows = read_jsonl(result_path)
    by_id = {str(row["presentation_id"]): row for row in rows}
    if (
        len(rows) != EXPECTED_PRESENTATIONS
        or len(by_id) != EXPECTED_PRESENTATIONS
        or set(by_id) != set(tasks)
    ):
        raise ValueError(f"{family}: evaluator result inventory differs")
    pair_votes: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for presentation_id, task in tasks.items():
        key = keys[presentation_id]
        vote = _validate_judgment(
            result=by_id[presentation_id],
            task=task,
            key=key,
            family=family,
            model=model,
            schema=schema,
        )
        pair_votes[str(key["pair_id"])].append(
            (int(key["orientation_index"]), vote)
        )
    if len(pair_votes) != EXPECTED_PAIRS:
        raise ValueError(f"{family}: pair inventory differs")
    outcomes = Counter()
    for pair_id, votes in pair_votes.items():
        if {orientation for orientation, _vote in votes} != {0, 1}:
            raise ValueError(f"{family}: pair orientations differ: {pair_id}")
        ordered_votes = [vote for _orientation, vote in sorted(votes)]
        if ordered_votes == ["R3_ALIGNED", "R3_ALIGNED"]:
            outcomes["R3_WIN"] += 1
        elif ordered_votes == ["R2_SHUFFLED", "R2_SHUFFLED"]:
            outcomes["R2_WIN"] += 1
        else:
            outcomes["TIE"] += 1
    if sum(outcomes.values()) != EXPECTED_PAIRS:
        raise ValueError(f"{family}: outcome count differs")
    return {
        "qualification_completed": True,
        "model": model,
        "presentations": EXPECTED_PRESENTATIONS,
        "pairs": EXPECTED_PAIRS,
        "r3_wins": outcomes["R3_WIN"],
        "r2_wins": outcomes["R2_WIN"],
        "ties": outcomes["TIE"],
        "p3_family_pass": outcomes["R3_WIN"] > outcomes["R2_WIN"],
        "result_sha256": sha256_file(result_path),
    }


def collect(
    *,
    family_a: str,
    family_a_results: Path,
    family_a_model: str,
    family_b: str,
    family_b_results: Path,
    family_b_model: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not family_a or not family_b or family_a == family_b:
        raise ValueError("qualification requires two distinct evaluator families")
    candidate = json.loads(CANDIDATE_LOCK.read_text(encoding="utf-8"))
    if (
        candidate.get("status")
        != "QUALIFICATION_PACKAGE_NOT_RUN_TRAINING_NOT_ALLOWED"
        or candidate.get("rationale_blind_qualification_completed") is not False
        or candidate.get("p3_preference_training_allowed") is not False
    ):
        raise PermissionError("qualification candidate boundary differs")
    if (
        sha256_file(TASKS)
        != str(
            candidate["private_output_hashes"]["score_blind_tasks"]["sha256"]
        )
        or sha256_file(ANSWER_KEY)
        != str(candidate["private_output_hashes"]["answer_key"]["sha256"])
        or sha256_file(SCHEMA)
        != str(candidate["source_hashes"]["score_blind_schema"])
    ):
        raise ValueError("qualification input hash differs")
    task_rows = read_jsonl(TASKS)
    key_rows = [
        row
        for row in read_jsonl(ANSWER_KEY)
        if row["stage"] == "score_blind"
    ]
    tasks = {str(row["presentation_id"]): row for row in task_rows}
    keys = {str(row["presentation_id"]): row for row in key_rows}
    if (
        len(tasks) != EXPECTED_PRESENTATIONS
        or len(keys) != EXPECTED_PRESENTATIONS
        or set(tasks) != set(keys)
    ):
        raise ValueError("qualification task/key closure differs")
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    families = {
        family_a: _family_decision(
            family=family_a,
            model=family_a_model,
            result_path=family_a_results,
            tasks=tasks,
            keys=keys,
            schema=schema,
        ),
        family_b: _family_decision(
            family=family_b,
            model=family_b_model,
            result_path=family_b_results,
            tasks=tasks,
            keys=keys,
            schema=schema,
        ),
    }
    passed = all(
        value["p3_family_pass"] is True for value in families.values()
    )
    public_families = {
        family: {
            field: value[field]
            for field in (
                "qualification_completed",
                "model",
                "presentations",
                "pairs",
                "r3_wins",
                "r2_wins",
                "ties",
                "p3_family_pass",
            )
        }
        for family, value in families.items()
    }
    evidence_hashes = {
        "candidate_lock": sha256_file(CANDIDATE_LOCK),
        "answer_key": sha256_file(ANSWER_KEY),
        "score_blind_tasks": sha256_file(TASKS),
        "score_blind_schema": sha256_file(SCHEMA),
        "api_runner": sha256_file(API_RUNNER),
        "collector": sha256_file(Path(__file__)),
        f"{family_a}_results": families[family_a]["result_sha256"],
        f"{family_b}_results": families[family_b]["result_sha256"],
    }
    report = {
        "schema_version": (
            "exp54-sorc-rationale-qualification-v1-final-report"
        ),
        "status": (
            "SORC_RATIONALE_QUALIFICATION_COMPLETE_P3_ALLOWED"
            if passed
            else "SORC_RATIONALE_QUALIFICATION_COMPLETE_P3_NOT_ALLOWED"
        ),
        "primary_stage": "score_blind",
        "score_visible_secondary_executed": False,
        "evaluator_family_results": public_families,
        "formal_rationale_block_passed": passed,
        "rationale_blind_qualification_completed": True,
        "p3_preference_training_allowed": passed,
        "preference_training_allowed": False,
        "source_hashes": candidate["source_hashes"],
        "qualification_evidence_hashes": evidence_hashes,
        "dev_accessed": False,
        "test_accessed": False,
    }
    lock = {
        "schema_version": FINAL_QUALIFICATION_SCHEMA_VERSION,
        "status": (
            FINAL_QUALIFICATION_STATUS
            if passed
            else "SORC_RATIONALE_QUALIFICATION_COMPLETE_P3_NOT_ALLOWED"
        ),
        "rationale_blind_qualification_completed": True,
        "p3_preference_training_allowed": passed,
        "evaluator_family_qualification_completed": True,
        "evaluator_family_count": 2,
        "evaluator_family_results": public_families,
        "source_hashes": candidate["source_hashes"],
        "qualification_evidence_hashes": evidence_hashes,
        "final_report_sha256": "",
        "preference_training_allowed": False,
        "dev_accessed": False,
        "test_accessed": False,
    }
    return report, lock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family-a-results",
        type=Path,
        default=DEFAULT_RESULTS["qwen"],
    )
    parser.add_argument("--family-a", default="qwen")
    parser.add_argument("--family-a-model", default="qwen3.7-max")
    parser.add_argument(
        "--family-b-results",
        type=Path,
        default=DEFAULT_RESULTS["deepseek"],
    )
    parser.add_argument("--family-b", default="deepseek")
    parser.add_argument("--family-b-model", default="deepseek-v4-pro")
    parser.add_argument("--final-report", type=Path, default=FINAL_REPORT)
    parser.add_argument("--final-lock", type=Path, default=FINAL_LOCK)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report, lock = collect(
        family_a=args.family_a,
        family_a_results=args.family_a_results,
        family_a_model=args.family_a_model,
        family_b=args.family_b,
        family_b_results=args.family_b_results,
        family_b_model=args.family_b_model,
    )
    write_json_exclusive(args.final_report, report)
    lock["final_report_sha256"] = sha256_file(args.final_report)
    write_json_exclusive(args.final_lock, lock)
    print(
        json.dumps(
            {
                "status": lock["status"],
                "rationale_blind_qualification_completed": True,
                "p3_preference_training_allowed": (
                    lock["p3_preference_training_allowed"]
                ),
                "evaluator_family_results": (
                    lock["evaluator_family_results"]
                ),
                "final_report_sha256": lock["final_report_sha256"],
                "final_lock_sha256": sha256_file(args.final_lock),
                "dev_accessed": False,
                "test_accessed": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
