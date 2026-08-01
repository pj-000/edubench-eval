"""Collect aggregate P2-vs-P3 rationale blind-audit results.

Row-level answer keys, model outputs, free-text judge reasons, and judgments
remain private.  The public report contains only aggregate preferences,
record-cluster bootstrap intervals, agreement statistics, and source hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from thesis_exp.exp54_rar_sft import REPO_ROOT


TARGET_ARM = "P3_JOINT_SORC"
COMPARATOR_ARM = "P2_SORC_SCORE"
SEEDS = (42, 43, 44)
VERDICTS = ("win", "tie", "loss")
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_730
STAGE_DIMENSIONS = {
    "score_blind": (
        "rubric_alignment",
        "answer_grounding",
        "specificity",
        "unsupported_claims_control",
        "completeness",
        "repetition_control",
        "overall_preference",
    ),
    "score_visible": (
        "score_rationale_consistency",
        "overall_scoring_justification_usefulness",
        "overall_preference",
    ),
}
DEFAULT_AUDIT_ROOT = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/"
    "preference_lr5e6_followup/rationale_blind_audit"
)


def _read_jsonl(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    payload = path.read_bytes()
    rows = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path}: empty JSONL")
    return payload, rows


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(
        ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
    )


def preference_metrics(verdicts: Sequence[str]) -> dict[str, Any]:
    counts = Counter(verdicts)
    total = len(verdicts)
    if total == 0:
        return {
            "pairs": 0,
            "wins": 0,
            "ties": 0,
            "losses": 0,
            "tie_adjusted_preference": None,
            "net_preference": None,
        }
    return {
        "pairs": total,
        "wins": counts["win"],
        "ties": counts["tie"],
        "losses": counts["loss"],
        "tie_adjusted_preference": (
            counts["win"] + 0.5 * counts["tie"]
        )
        / total,
        "net_preference": (counts["win"] - counts["loss"]) / total,
    }


def _map_ab(value: str, key: dict[str, Any]) -> str:
    if value == "tie":
        return "tie"
    if value not in {"A", "B"}:
        raise ValueError(f"invalid judgment: {value!r}")
    selected = (
        str(key["candidate_a_arm"])
        if value == "A"
        else str(key["candidate_b_arm"])
    )
    if selected == TARGET_ARM:
        return "win"
    if selected == COMPARATOR_ARM:
        return "loss"
    raise ValueError("answer key contains an unexpected arm")


def _forced_stratum(target_forced: bool, comparator_forced: bool) -> str:
    if target_forced and comparator_forced:
        return "both_forced"
    if target_forced:
        return "P3_only_forced"
    if comparator_forced:
        return "P2_only_forced"
    return "neither_forced"


def collapse_orientations(
    answer_key: Sequence[dict[str, Any]],
    judgments: Sequence[dict[str, Any]],
    *,
    stage: str,
    enforce_full_shape: bool = True,
) -> list[dict[str, Any]]:
    if stage not in STAGE_DIMENSIONS:
        raise ValueError(f"unknown stage: {stage}")
    expected_fields = {
        "presentation_id",
        "brief_reason",
        *STAGE_DIMENSIONS[stage],
    }
    by_id = {}
    for row in judgments:
        if set(row) != expected_fields:
            raise ValueError(f"{stage}: judgment fields differ")
        presentation_id = str(row["presentation_id"])
        if presentation_id in by_id:
            raise ValueError(f"{stage}: duplicate judgment ID")
        if any(
            str(row[field]) not in {"A", "B", "tie"}
            for field in STAGE_DIMENSIONS[stage]
        ):
            raise ValueError(f"{stage}: judgment enum differs")
        if not str(row["brief_reason"]).strip():
            raise ValueError(f"{stage}: empty judge reason")
        by_id[presentation_id] = row
    key_ids = [str(row["presentation_id"]) for row in answer_key]
    if len(key_ids) != len(set(key_ids)):
        raise ValueError("answer key contains duplicate presentation IDs")
    if set(by_id) != set(key_ids):
        raise ValueError("judgment and answer-key IDs differ")

    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = (
        defaultdict(list)
    )
    for key in answer_key:
        arms = {
            str(key["candidate_a_arm"]),
            str(key["candidate_b_arm"]),
        }
        if arms != {TARGET_ARM, COMPARATOR_ARM}:
            raise ValueError("answer key arm set differs")
        grouped[str(key["pair_id"])].append(
            (key, by_id[str(key["presentation_id"])])
        )

    collapsed = []
    for pair_id, presentations in sorted(grouped.items()):
        presentations.sort(key=lambda item: int(item[0]["orientation"]))
        if (
            len(presentations) != 2
            or [int(item[0]["orientation"]) for item in presentations]
            != [0, 1]
        ):
            raise ValueError(f"{pair_id}: orientation closure differs")
        first = presentations[0][0]
        for key, _judgment in presentations[1:]:
            for field in ("record_id", "seed", "label_5"):
                if key[field] != first[field]:
                    raise ValueError(f"{pair_id}: pair metadata differs")

        forced = {}
        mapped: dict[str, list[str]] = defaultdict(list)
        for key, judgment in presentations:
            for arm_field, forced_field in (
                ("candidate_a_arm", "candidate_a_forced"),
                ("candidate_b_arm", "candidate_b_forced"),
            ):
                arm = str(key[arm_field])
                current = bool(key[forced_field])
                if arm in forced and forced[arm] != current:
                    raise ValueError(f"{pair_id}: forced flag differs")
                forced[arm] = current
            for dimension in STAGE_DIMENSIONS[stage]:
                mapped[dimension].append(
                    _map_ab(str(judgment[dimension]), key)
                )
        collapsed.append(
            {
                "record_id": str(first["record_id"]),
                "seed": int(first["seed"]),
                "label_5": int(first["label_5"]),
                "forced_stratum": _forced_stratum(
                    forced[TARGET_ARM], forced[COMPARATOR_ARM]
                ),
                "orientation_conflict": any(
                    values[0] != values[1] for values in mapped.values()
                ),
                "verdicts": {
                    dimension: (
                        values[0] if values[0] == values[1] else "tie"
                    )
                    for dimension, values in mapped.items()
                },
            }
        )
    if enforce_full_shape:
        if len(collapsed) != 120:
            raise ValueError("collapsed audit must contain 120 pairs")
        records: dict[str, set[int]] = defaultdict(set)
        for row in collapsed:
            records[str(row["record_id"])].add(int(row["seed"]))
        if len(records) != 40 or any(
            seeds != set(SEEDS) for seeds in records.values()
        ):
            raise ValueError("audit must contain 40 records x three seeds")
    return collapsed


def _bootstrap(
    rows: Sequence[dict[str, Any]],
    *,
    dimension: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_record[str(row["record_id"])].append(row)
    record_ids = sorted(by_record)
    rng = random.Random(seed)
    adjusted = []
    net = []
    for _ in range(replicates):
        verdicts = []
        for _cluster in record_ids:
            sampled = record_ids[rng.randrange(len(record_ids))]
            verdicts.extend(
                str(row["verdicts"][dimension])
                for row in by_record[sampled]
            )
        metrics = preference_metrics(verdicts)
        adjusted.append(float(metrics["tie_adjusted_preference"]))
        net.append(float(metrics["net_preference"]))
    return {
        "tie_adjusted_preference": [
            _percentile(adjusted, 0.025),
            _percentile(adjusted, 0.975),
        ],
        "net_preference": [
            _percentile(net, 0.025),
            _percentile(net, 0.975),
        ],
        "replicates": replicates,
        "seed": seed,
    }


def _aggregate(
    rows: Sequence[dict[str, Any]],
    *,
    evaluator: str,
    stage: str,
    replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    output = []
    strata = (
        "all",
        "neither_forced",
        "P3_only_forced",
        "P2_only_forced",
        "both_forced",
    )
    label_strata = {
        "all": lambda row: True,
        "label_1": lambda row: int(row["label_5"]) == 1,
        "label_2": lambda row: int(row["label_5"]) == 2,
        "low_1_2": lambda row: int(row["label_5"]) <= 2,
        "non_low_3_5": lambda row: int(row["label_5"]) >= 3,
    }
    stratum_specs = [
        ("all", "all"),
        *(("all", stratum) for stratum in strata if stratum != "all"),
        *(
            (label_stratum, "all")
            for label_stratum in label_strata
            if label_stratum != "all"
        ),
    ]
    for dimension in STAGE_DIMENSIONS[stage]:
        for label_stratum, stratum in stratum_specs:
            label_predicate = label_strata[label_stratum]
            selected = [
                row
                for row in rows
                if label_predicate(row)
                and (
                    stratum == "all"
                    or row["forced_stratum"] == stratum
                )
            ]
            metrics = preference_metrics(
                [str(row["verdicts"][dimension]) for row in selected]
            )
            bootstrap = (
                _bootstrap(
                    selected,
                    dimension=dimension,
                    replicates=replicates,
                    seed=seed,
                )
                if selected
                else None
            )
            output.append(
                {
                    "evaluator": evaluator,
                    "comparison": "P3_JOINT_SORC_vs_P2_SORC_SCORE",
                    "stage": stage,
                    "dimension": dimension,
                    "label_stratum": label_stratum,
                    "forced_completion_stratum": stratum,
                    **metrics,
                    "bootstrap_95ci": bootstrap,
                }
            )
    return output


def _cohen_kappa(left: Sequence[str], right: Sequence[str]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("kappa inputs differ")
    total = len(left)
    observed = sum(a == b for a, b in zip(left, right)) / total
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum(
        left_counts[value] * right_counts[value] / (total * total)
        for value in VERDICTS
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else math.nan
    return (observed - expected) / (1.0 - expected)


def _agreement(
    evaluator_rows: Mapping[str, Mapping[str, Sequence[dict[str, Any]]]]
) -> list[dict[str, Any]]:
    evaluators = sorted(evaluator_rows)
    if len(evaluators) != 2:
        raise ValueError("agreement requires two evaluators")
    first_name, second_name = evaluators
    output = []
    for stage in STAGE_DIMENSIONS:
        def keyed(name: str) -> dict[tuple[str, int], dict[str, Any]]:
            return {
                (str(row["record_id"]), int(row["seed"])): row
                for row in evaluator_rows[name][stage]
            }

        first = keyed(first_name)
        second = keyed(second_name)
        if set(first) != set(second):
            raise ValueError("evaluator pair sets differ")
        keys = sorted(first)
        for dimension in STAGE_DIMENSIONS[stage]:
            left = [
                str(first[key]["verdicts"][dimension]) for key in keys
            ]
            right = [
                str(second[key]["verdicts"][dimension]) for key in keys
            ]
            output.append(
                {
                    "evaluators": [first_name, second_name],
                    "stage": stage,
                    "dimension": dimension,
                    "pairs": len(keys),
                    "exact_agreement_rate": statistics_fmean(
                        a == b for a, b in zip(left, right)
                    ),
                    "cohen_kappa": _cohen_kappa(left, right),
                }
            )
    return output


def statistics_fmean(values: Sequence[float] | Any) -> float:
    materialized = [float(value) for value in values]
    if not materialized:
        raise ValueError("mean requires values")
    return sum(materialized) / len(materialized)


def collect(
    *,
    answer_key_path: Path,
    judgment_paths: Mapping[tuple[str, str], Path],
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    evaluators = sorted({name for name, _stage in judgment_paths})
    expected = {
        (evaluator, stage)
        for evaluator in evaluators
        for stage in STAGE_DIMENSIONS
    }
    if len(evaluators) != 2 or set(judgment_paths) != expected:
        raise ValueError("two evaluators must each provide both stages")
    answer_payload, answer_key = _read_jsonl(answer_key_path)
    rows_by_evaluator: dict[str, dict[str, list[dict[str, Any]]]] = {
        evaluator: {} for evaluator in evaluators
    }
    judgment_hashes = {}
    aggregates = []
    private_reasons = set()
    for evaluator in evaluators:
        for stage in STAGE_DIMENSIONS:
            payload, judgments = _read_jsonl(
                judgment_paths[(evaluator, stage)]
            )
            judgment_hashes[f"{evaluator}|{stage}"] = _sha256_bytes(payload)
            private_reasons.update(str(row["brief_reason"]) for row in judgments)
            collapsed = collapse_orientations(
                answer_key, judgments, stage=stage
            )
            rows_by_evaluator[evaluator][stage] = collapsed
            aggregates.extend(
                _aggregate(
                    collapsed,
                    evaluator=evaluator,
                    stage=stage,
                    replicates=replicates,
                    seed=seed,
                )
            )
    report = {
        "schema_version": "exp54-sorc-dpo-rationale-audit-results-v1",
        "status": "SORC_DPO_RATIONALE_BLIND_AUDIT_COMPLETE",
        "comparison": "P3_JOINT_SORC_vs_P2_SORC_SCORE",
        "claim_scope": (
            "exploratory Codex-agent model-based visible rationale preference"
        ),
        "evaluators": evaluators,
        "evaluator_family_independence_satisfied": False,
        "human_correctness_claim_allowed": False,
        "aggregates": aggregates,
        "agreement": _agreement(rows_by_evaluator),
        "orientation_conflict_rates": {
            f"{evaluator}|{stage}": statistics_fmean(
                row["orientation_conflict"]
                for row in rows_by_evaluator[evaluator][stage]
            )
            for evaluator in evaluators
            for stage in STAGE_DIMENSIONS
        },
        "source_hashes": {
            "answer_key": _sha256_bytes(answer_payload),
            **judgment_hashes,
        },
        "row_level_judgments_public": False,
        "free_text_reasons_public": False,
        "forced_completion_included_in_primary": True,
        "dev_regenerated": False,
        "test_accessed": False,
    }
    public_text = json.dumps(report, ensure_ascii=False, sort_keys=True)
    forbidden = {
        str(row["record_id"]) for row in answer_key
    } | {
        str(row["presentation_id"]) for row in answer_key
    } | private_reasons
    if any(value and value in public_text for value in forbidden):
        raise ValueError("public report leaks a private row-level value")
    return report


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# P3 versus P2 rationale blind audit",
        "",
        "This is an exploratory Codex-agent model preference audit, not human "
        "or expert correctness. Both evaluators are from the Codex GPT-5.6 "
        "family, so evaluator-family independence is not satisfied.",
        "",
        "| Evaluator | Stage | Dimension | P3 win / tie / loss | "
        "Tie-adjusted preference (95% CI) |",
        "|---|---|---|---:|---:|",
    ]
    for row in report["aggregates"]:
        if (
            row["forced_completion_stratum"] != "all"
            or row["label_stratum"] != "all"
        ):
            continue
        ci = row["bootstrap_95ci"]["tie_adjusted_preference"]
        lines.append(
            f"| {row['evaluator']} | {row['stage']} | {row['dimension']} | "
            f"{row['wins']} / {row['ties']} / {row['losses']} | "
            f"{row['tie_adjusted_preference']:.3f} "
            f"[{ci[0]:.3f}, {ci[1]:.3f}] |"
        )
    lines.extend(
        [
            "",
            "All forced-completed outputs remain in the primary aggregate. "
            "Forced-completion strata are diagnostic only.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--answer-key",
        type=Path,
        default=DEFAULT_AUDIT_ROOT / "private/answer_key.jsonl",
    )
    parser.add_argument("--sol-score-blind", type=Path, required=True)
    parser.add_argument("--sol-score-visible", type=Path, required=True)
    parser.add_argument("--terra-score-blind", type=Path, required=True)
    parser.add_argument("--terra-score-visible", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_AUDIT_ROOT / "public_results",
    )
    args = parser.parse_args()
    report = collect(
        answer_key_path=args.answer_key,
        judgment_paths={
            ("codex_sol", "score_blind"): args.sol_score_blind,
            ("codex_sol", "score_visible"): args.sol_score_visible,
            ("codex_terra", "score_blind"): args.terra_score_blind,
            ("codex_terra", "score_visible"): args.terra_score_visible,
        },
    )
    report_path = args.output / "results.json"
    markdown_path = args.output / "report.md"
    _atomic_write(
        report_path,
        (
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8"),
    )
    _atomic_write(markdown_path, _markdown(report).encode("utf-8"))
    print(
        json.dumps(
            {
                "status": report["status"],
                "report_sha256": _sha256_bytes(report_path.read_bytes()),
                "evaluator_family_independence_satisfied": False,
                "dev_regenerated": False,
                "test_accessed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
