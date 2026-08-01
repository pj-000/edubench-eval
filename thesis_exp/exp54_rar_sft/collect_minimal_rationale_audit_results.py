"""Collect aggregate results for the Exp54 minimal rationale blind audit.

The collector reads the private orientation answer key and four judgment
files: two evaluator families crossed with the score-blind and score-visible
stages. It writes only aggregate public artifacts. Record identities,
presentation identities, free-text judge reasons, and row-level verdicts
remain private and are never serialized.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from thesis_exp.exp54_rar_sft import REPO_ROOT


STAGE_DIMENSIONS = {
    "score_blind": (
        "metric_alignment",
        "rubric_relevance",
        "answer_grounding",
        "specificity",
        "unsupported_claims",
        "completeness",
        "overall_preference",
    ),
    "score_visible": (
        "score_rationale_consistency",
        "overall_scoring_justification_usefulness",
        "overall_preference",
    ),
}
COMPARISONS = {
    "primary": "R2",
    "secondary": "R1",
}
VERDICTS = ("win", "tie", "loss")
FORCED_STRATA = (
    "all",
    "neither_candidate_forced",
    "R3_only_forced",
    "comparator_only_forced",
    "both_candidates_forced",
)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_728
EXPECTED_RECORDS = 40
EXPECTED_SEEDS = (42, 43, 44)
EXPECTED_PRESENTATIONS = 480


def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_jsonl_bytes(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    payload = path.read_bytes()
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        rows.append(value)
    return payload, rows


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot calculate percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(
        ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
    )


def preference_metrics(
    verdicts: Sequence[str],
) -> dict[str, float | int | None]:
    counts = Counter(verdicts)
    total = len(verdicts)
    if total == 0:
        return {
            "pairs": 0,
            "wins": 0,
            "ties": 0,
            "losses": 0,
            "win_rate_ties_retained": None,
            "tie_adjusted_preference": None,
            "net_preference": None,
        }
    return {
        "pairs": total,
        "wins": counts["win"],
        "ties": counts["tie"],
        "losses": counts["loss"],
        "win_rate_ties_retained": counts["win"] / total,
        "tie_adjusted_preference": (
            counts["win"] + 0.5 * counts["tie"]
        )
        / total,
        "net_preference": (counts["win"] - counts["loss"]) / total,
    }


def forced_stratum(r3_forced: bool, comparator_forced: bool) -> str:
    if r3_forced and comparator_forced:
        return "both_candidates_forced"
    if r3_forced:
        return "R3_only_forced"
    if comparator_forced:
        return "comparator_only_forced"
    return "neither_candidate_forced"


def map_ab_to_r3(
    judgment: str,
    *,
    candidate_a_arm: str,
    candidate_b_arm: str,
    comparator_arm: str,
) -> str:
    if judgment == "tie":
        return "tie"
    if judgment not in {"A", "B"}:
        raise ValueError(f"unexpected A/B judgment: {judgment!r}")
    chosen_arm = (
        candidate_a_arm if judgment == "A" else candidate_b_arm
    )
    if chosen_arm == "R3":
        return "win"
    if chosen_arm == comparator_arm:
        return "loss"
    raise ValueError("presentation contains an unexpected candidate arm")


def _validate_answer_key(
    answer_key: Sequence[dict[str, Any]],
    *,
    enforce_formal_shape: bool,
) -> None:
    if enforce_formal_shape and len(answer_key) != EXPECTED_PRESENTATIONS:
        raise ValueError("answer key must contain exactly 480 presentations")
    presentation_ids = [str(row["presentation_id"]) for row in answer_key]
    if len(presentation_ids) != len(set(presentation_ids)):
        raise ValueError("answer key contains duplicate presentation IDs")
    records = {str(row["record_id"]) for row in answer_key}
    if enforce_formal_shape and len(records) != EXPECTED_RECORDS:
        raise ValueError("answer key must contain exactly 40 record clusters")


def pair_verdicts(
    answer_key: Sequence[dict[str, Any]],
    judgments: Sequence[dict[str, Any]],
    *,
    stage: str,
    enforce_formal_shape: bool = True,
) -> list[dict[str, Any]]:
    """Map A/B judgments to R3 and collapse two orientations per pair."""

    if stage not in STAGE_DIMENSIONS:
        raise ValueError(f"unknown audit stage: {stage}")
    _validate_answer_key(answer_key, enforce_formal_shape=enforce_formal_shape)
    judgment_by_id: dict[str, dict[str, Any]] = {}
    expected_judgment_fields = {
        "presentation_id",
        "brief_reason",
        *STAGE_DIMENSIONS[stage],
    }
    for row in judgments:
        if set(row) != expected_judgment_fields:
            raise ValueError("judgment fields differ from the stage contract")
        presentation_id = str(row["presentation_id"])
        if presentation_id in judgment_by_id:
            raise ValueError("judgments contain duplicate presentation IDs")
        judgment_by_id[presentation_id] = row
    expected_ids = {str(row["presentation_id"]) for row in answer_key}
    if set(judgment_by_id) != expected_ids:
        raise ValueError("judgment and answer-key presentation sets differ")

    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = (
        defaultdict(list)
    )
    for key_row in answer_key:
        grouped[str(key_row["pair_id"])].append(
            (
                key_row,
                judgment_by_id[str(key_row["presentation_id"])],
            )
        )

    results: list[dict[str, Any]] = []
    for pair_id, presentations in sorted(grouped.items()):
        if len(presentations) != 2:
            raise ValueError(f"{pair_id}: expected two orientations")
        presentations.sort(key=lambda item: int(item[0]["orientation"]))
        if [int(item[0]["orientation"]) for item in presentations] != [0, 1]:
            raise ValueError(f"{pair_id}: orientations must be 0 and 1")

        first = presentations[0][0]
        comparison = str(first["comparison"])
        if comparison not in COMPARISONS:
            raise ValueError(f"{pair_id}: unknown comparison")
        comparator_arm = COMPARISONS[comparison]
        invariant_fields = ("record_id", "seed", "comparison")
        for key_row, _ in presentations[1:]:
            if any(key_row[name] != first[name] for name in invariant_fields):
                raise ValueError(f"{pair_id}: pair metadata differs")

        forced_by_arm: dict[str, bool] = {}
        mapped_by_dimension: dict[str, list[str]] = defaultdict(list)
        for key_row, judgment_row in presentations:
            arms = {
                str(key_row["candidate_a_arm"]),
                str(key_row["candidate_b_arm"]),
            }
            if arms != {"R3", comparator_arm}:
                raise ValueError(f"{pair_id}: candidate arm set differs")
            for arm_key, forced_key in (
                ("candidate_a_arm", "candidate_a_forced"),
                ("candidate_b_arm", "candidate_b_forced"),
            ):
                arm = str(key_row[arm_key])
                forced = bool(key_row[forced_key])
                if arm in forced_by_arm and forced_by_arm[arm] != forced:
                    raise ValueError(f"{pair_id}: forced flag differs by arm")
                forced_by_arm[arm] = forced
            for dimension in STAGE_DIMENSIONS[stage]:
                mapped_by_dimension[dimension].append(
                    map_ab_to_r3(
                        str(judgment_row[dimension]),
                        candidate_a_arm=str(key_row["candidate_a_arm"]),
                        candidate_b_arm=str(key_row["candidate_b_arm"]),
                        comparator_arm=comparator_arm,
                    )
                )

        dimension_verdicts = {
            dimension: (
                values[0] if values[0] == values[1] else "tie"
            )
            for dimension, values in mapped_by_dimension.items()
        }
        results.append(
            {
                # Private clustering fields remain in memory only.
                "record_id": str(first["record_id"]),
                "seed": int(first["seed"]),
                "comparison": comparison,
                "forced_stratum": forced_stratum(
                    forced_by_arm["R3"],
                    forced_by_arm[comparator_arm],
                ),
                "verdicts": dimension_verdicts,
            }
        )

    if enforce_formal_shape:
        expected_pairs = EXPECTED_RECORDS * len(EXPECTED_SEEDS) * len(
            COMPARISONS
        )
        if len(results) != expected_pairs:
            raise ValueError("collapsed pair count differs from 240")
        for comparison in COMPARISONS:
            selected = [
                row for row in results if row["comparison"] == comparison
            ]
            if len(selected) != EXPECTED_RECORDS * len(EXPECTED_SEEDS):
                raise ValueError(f"{comparison}: pair count differs")
            by_record: dict[str, set[int]] = defaultdict(set)
            for row in selected:
                by_record[str(row["record_id"])].add(int(row["seed"]))
            if any(
                seeds != set(EXPECTED_SEEDS)
                for seeds in by_record.values()
            ):
                raise ValueError(
                    f"{comparison}: each record must carry seeds 42/43/44"
                )
    return results


def bootstrap_interval(
    rows: Sequence[dict[str, Any]],
    *,
    dimension: str,
    stratum: str = "all",
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, list[float] | int | None]:
    """Record-cluster bootstrap carrying all seed pairs per sampled record."""

    by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_record[str(row["record_id"])].append(row)
    record_ids = sorted(by_record)
    if not record_ids:
        raise ValueError("bootstrap received no record clusters")
    rng = random.Random(seed)
    samples: dict[str, list[float]] = {
        "win_rate_ties_retained": [],
        "tie_adjusted_preference": [],
        "net_preference": [],
    }
    skipped = 0
    for _ in range(replicates):
        verdicts: list[str] = []
        for _cluster in record_ids:
            sampled_id = record_ids[rng.randrange(len(record_ids))]
            for row in by_record[sampled_id]:
                if stratum in {"all", str(row["forced_stratum"])}:
                    verdicts.append(str(row["verdicts"][dimension]))
        metrics = preference_metrics(verdicts)
        if not verdicts:
            skipped += 1
            continue
        for name in samples:
            samples[name].append(float(metrics[name]))
    intervals: dict[str, list[float] | int | None] = {
        name: (
            [percentile(values, 0.025), percentile(values, 0.975)]
            if values
            else None
        )
        for name, values in samples.items()
    }
    intervals["valid_replicates"] = replicates - skipped
    intervals["skipped_empty_replicates"] = skipped
    return intervals


def aggregate_rows(
    pair_rows: Sequence[dict[str, Any]],
    *,
    evaluator: str,
    stage: str,
    comparison_roles: Sequence[str] = tuple(COMPARISONS),
    replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    for comparison in comparison_roles:
        if comparison not in COMPARISONS:
            raise ValueError(f"unknown comparison role: {comparison}")
        comparison_rows = [
            row for row in pair_rows if row["comparison"] == comparison
        ]
        for dimension in STAGE_DIMENSIONS[stage]:
            for stratum in FORCED_STRATA:
                selected = [
                    row
                    for row in comparison_rows
                    if stratum in {"all", row["forced_stratum"]}
                ]
                metrics = preference_metrics(
                    [str(row["verdicts"][dimension]) for row in selected]
                )
                bootstrap = bootstrap_interval(
                    comparison_rows,
                    dimension=dimension,
                    stratum=stratum,
                    replicates=replicates,
                    seed=bootstrap_seed,
                )
                aggregates.append(
                    {
                        "evaluator": evaluator,
                        "comparison_role": comparison,
                        "comparison": f"R3_vs_{COMPARISONS[comparison]}",
                        "stage": stage,
                        "dimension": dimension,
                        "forced_completion_stratum": stratum,
                        **metrics,
                        "bootstrap_95ci": bootstrap,
                    }
                )
    return aggregates


def cohen_kappa(left: Sequence[str], right: Sequence[str]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("kappa inputs must have equal nonzero length")
    total = len(left)
    observed = sum(a == b for a, b in zip(left, right)) / total
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum(
        (left_counts[value] / total) * (right_counts[value] / total)
        for value in VERDICTS
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else math.nan
    return (observed - expected) / (1.0 - expected)


def agreement_rows(
    evaluator_rows: Mapping[str, Mapping[str, Sequence[dict[str, Any]]]],
    *,
    stages: Sequence[str] = tuple(STAGE_DIMENSIONS),
    comparison_roles: Sequence[str] = tuple(COMPARISONS),
) -> list[dict[str, Any]]:
    evaluators = sorted(evaluator_rows)
    if len(evaluators) != 2:
        raise ValueError("cross-evaluator agreement requires two evaluators")
    left_name, right_name = evaluators
    output = []
    for stage in stages:
        if stage not in STAGE_DIMENSIONS:
            raise ValueError(f"unknown agreement stage: {stage}")
        if any(stage not in evaluator_rows[name] for name in evaluators):
            raise ValueError("evaluator stage is missing for agreement")

        def keyed(
            rows: Sequence[dict[str, Any]],
        ) -> dict[tuple[str, int, str], dict[str, Any]]:
            return {
                (
                    str(row["record_id"]),
                    int(row["seed"]),
                    str(row["comparison"]),
                ): row
                for row in rows
            }

        left = keyed(evaluator_rows[left_name][stage])
        right = keyed(evaluator_rows[right_name][stage])
        if set(left) != set(right):
            raise ValueError("evaluator pair sets differ")
        for comparison in comparison_roles:
            if comparison not in COMPARISONS:
                raise ValueError(f"unknown comparison role: {comparison}")
            keys = sorted(key for key in left if key[2] == comparison)
            if not keys:
                raise ValueError(
                    f"no evaluator pairs for comparison {comparison}"
                )
            for dimension in STAGE_DIMENSIONS[stage]:
                left_values = [
                    str(left[key]["verdicts"][dimension]) for key in keys
                ]
                right_values = [
                    str(right[key]["verdicts"][dimension]) for key in keys
                ]
                confusion = Counter(zip(left_values, right_values))
                output.append(
                    {
                        "evaluators": [left_name, right_name],
                        "comparison_role": comparison,
                        "comparison": f"R3_vs_{COMPARISONS[comparison]}",
                        "stage": stage,
                        "dimension": dimension,
                        "pairs": len(keys),
                        "exact_agreement_rate": sum(
                            a == b
                            for a, b in zip(left_values, right_values)
                        )
                        / len(keys),
                        "cohen_kappa": cohen_kappa(
                            left_values, right_values
                        ),
                        "confusion_matrix": {
                            left_value: {
                                right_value: confusion[
                                    (left_value, right_value)
                                ]
                                for right_value in VERDICTS
                            }
                            for left_value in VERDICTS
                        },
                    }
                )
    return output


def _all_scalar_values(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _all_scalar_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _all_scalar_values(item)
    else:
        yield value


def assert_public_privacy(
    public_value: Any,
    *,
    answer_key: Sequence[dict[str, Any]],
    judgment_sets: Iterable[Sequence[dict[str, Any]]],
) -> None:
    forbidden_names = {"record_id", "presentation_id", "brief_reason"}
    public_scalars = set(_all_scalar_values(public_value))
    if forbidden_names & public_scalars:
        raise ValueError("public artifact contains a forbidden private field")
    forbidden_values = {
        str(row[name])
        for row in answer_key
        for name in ("record_id", "presentation_id")
    }
    for judgments in judgment_sets:
        forbidden_values.update(
            str(row["brief_reason"])
            for row in judgments
            if "brief_reason" in row
        )
    if public_scalars & forbidden_values:
        raise ValueError("public artifact contains a private row-level value")


def _csv_payload(aggregates: Sequence[dict[str, Any]]) -> str:
    fields = [
        "evaluator",
        "comparison_role",
        "comparison",
        "stage",
        "dimension",
        "forced_completion_stratum",
        "pairs",
        "wins",
        "ties",
        "losses",
        "win_rate_ties_retained",
        "win_rate_ci_low",
        "win_rate_ci_high",
        "tie_adjusted_preference",
        "tie_adjusted_ci_low",
        "tie_adjusted_ci_high",
        "net_preference",
        "net_preference_ci_low",
        "net_preference_ci_high",
    ]
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in aggregates:
        intervals = row["bootstrap_95ci"]
        win_ci = intervals["win_rate_ties_retained"]
        adjusted_ci = intervals["tie_adjusted_preference"]
        net_ci = intervals["net_preference"]
        writer.writerow(
            {
                "evaluator": row["evaluator"],
                "comparison_role": row["comparison_role"],
                "comparison": row["comparison"],
                "stage": row["stage"],
                "dimension": row["dimension"],
                "forced_completion_stratum": row[
                    "forced_completion_stratum"
                ],
                "pairs": row["pairs"],
                "wins": row["wins"],
                "ties": row["ties"],
                "losses": row["losses"],
                "win_rate_ties_retained": row[
                    "win_rate_ties_retained"
                ],
                "win_rate_ci_low": win_ci[0] if win_ci else None,
                "win_rate_ci_high": win_ci[1] if win_ci else None,
                "tie_adjusted_preference": row[
                    "tie_adjusted_preference"
                ],
                "tie_adjusted_ci_low": (
                    adjusted_ci[0] if adjusted_ci else None
                ),
                "tie_adjusted_ci_high": (
                    adjusted_ci[1] if adjusted_ci else None
                ),
                "net_preference": row["net_preference"],
                "net_preference_ci_low": net_ci[0] if net_ci else None,
                "net_preference_ci_high": net_ci[1] if net_ci else None,
            }
        )
    return handle.getvalue()


def _markdown_payload(report: dict[str, Any]) -> str:
    lines = [
        "# Exp54 minimal rationale blind-audit results",
        "",
        "All results are model-based rationale preference/agreement, not "
        "human or expert correctness.",
        "",
        "## Overall score-visible result",
        "",
        "| Evaluator | Comparison | Dimension | Win / tie / loss | "
        "Tie-adjusted preference (95% CI) | Net preference (95% CI) |",
        "|---|---|---|---:|---:|---:|",
    ]
    wanted = {
        "score_rationale_consistency",
        "overall_scoring_justification_usefulness",
        "overall_preference",
    }
    for row in report["aggregates"]:
        if (
            row["stage"] != "score_visible"
            or row["forced_completion_stratum"] != "all"
            or row["dimension"] not in wanted
        ):
            continue
        ci = row["bootstrap_95ci"]
        lines.append(
            "| {evaluator} | {comparison} | {dimension} | "
            "{wins} / {ties} / {losses} | {preference:.3f} "
            "[{preference_low:.3f}, {preference_high:.3f}] | "
            "{net:.3f} [{net_low:.3f}, {net_high:.3f}] |".format(
                **row,
                preference=float(row["tie_adjusted_preference"]),
                preference_low=ci["tie_adjusted_preference"][0],
                preference_high=ci["tie_adjusted_preference"][1],
                net=float(row["net_preference"]),
                net_low=ci["net_preference"][0],
                net_high=ci["net_preference"][1],
            )
        )
    lines.extend(
        [
            "",
            "Primary inference is R3 versus R2 on "
            "`overall_scoring_justification_usefulness`. Other comparisons, "
            "dimensions, evaluator pooling, and forced-completion strata are "
            "secondary or diagnostic.",
            "",
            "Intervals use 10,000 record-cluster bootstrap replicates with "
            "seed 20260728, carrying all three training-seed pairs whenever "
            "a record cluster is sampled.",
            "",
        ]
    )
    return "\n".join(lines)


def _exploratory_markdown_payload(report: dict[str, Any]) -> str:
    lines = [
        "# Exp54 exploratory agent primary rationale preference",
        "",
        "**Claim scope:** Codex-agent exploratory primary score-visible "
        "preference.",
        "",
        "Evaluator-family independence is not satisfied. This artifact does "
        "not complete the formal preregistered two-family blind audit.",
        "",
        "Both orientations were judged within each agent's persistent "
        "context. Therefore the zero orientation-conflict result is not an "
        "independent position-bias diagnostic and must not be interpreted as "
        "one.",
        "",
        "| Agent evaluator | Comparison | Dimension | Win / tie / loss | "
        "Tie-adjusted preference (95% CI) | Net preference (95% CI) |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in report["aggregates"]:
        if row["forced_completion_stratum"] != "all":
            continue
        ci = row["bootstrap_95ci"]
        lines.append(
            "| {evaluator} | {comparison} | {dimension} | "
            "{wins} / {ties} / {losses} | {preference:.3f} "
            "[{preference_low:.3f}, {preference_high:.3f}] | "
            "{net:.3f} [{net_low:.3f}, {net_high:.3f}] |".format(
                **row,
                preference=float(row["tie_adjusted_preference"]),
                preference_low=ci["tie_adjusted_preference"][0],
                preference_high=ci["tie_adjusted_preference"][1],
                net=float(row["net_preference"]),
                net_low=ci["net_preference"][0],
                net_high=ci["net_preference"][1],
            )
        )
    lines.extend(
        [
            "",
            "Intervals use 10,000 record-cluster bootstrap replicates with "
            "seed 20260728, carrying all three training-seed pairs whenever "
            "a record cluster is sampled.",
            "",
            "Forced-completed outputs remain in the primary aggregate; "
            "preplanned forced-completion strata are diagnostic.",
            "",
        ]
    )
    return "\n".join(lines)


def write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def collect_results(
    *,
    answer_key_path: Path,
    judgment_paths: Mapping[tuple[str, str], Path],
    output_dir: Path,
    replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    enforce_formal_shape: bool = True,
) -> dict[str, Any]:
    evaluators = sorted({name for name, _stage in judgment_paths})
    if len(evaluators) != 2:
        raise ValueError("exactly two evaluator families are required")
    expected_keys = {
        (evaluator, stage)
        for evaluator in evaluators
        for stage in STAGE_DIMENSIONS
    }
    if set(judgment_paths) != expected_keys:
        raise ValueError("each evaluator needs both judgment stages")

    answer_payload, answer_key = read_jsonl_bytes(answer_key_path)
    evaluator_rows: dict[str, dict[str, list[dict[str, Any]]]] = {
        evaluator: {} for evaluator in evaluators
    }
    judgment_payloads: dict[tuple[str, str], bytes] = {}
    judgment_inputs: list[list[dict[str, Any]]] = []
    aggregates = []
    for evaluator in evaluators:
        for stage in STAGE_DIMENSIONS:
            payload, judgments = read_jsonl_bytes(
                judgment_paths[(evaluator, stage)]
            )
            judgment_payloads[(evaluator, stage)] = payload
            judgment_inputs.append(judgments)
            pairs = pair_verdicts(
                answer_key,
                judgments,
                stage=stage,
                enforce_formal_shape=enforce_formal_shape,
            )
            evaluator_rows[evaluator][stage] = pairs
            aggregates.extend(
                aggregate_rows(
                    pairs,
                    evaluator=evaluator,
                    stage=stage,
                    replicates=replicates,
                    bootstrap_seed=bootstrap_seed,
                )
            )

    report = {
        "schema_version": "exp54-minimal-rationale-audit-results-v1",
        "status": "MINIMAL_RATIONALE_AUDIT_RESULTS_COLLECTED",
        "analysis_contract": {
            "primary": (
                "R3_vs_R2 score-visible "
                "overall_scoring_justification_usefulness"
            ),
            "secondary_comparison": "R3_vs_R1",
            "orientation_disagreement_rule": "pair verdict becomes tie",
            "bootstrap": {
                "method": "record-cluster hierarchical bootstrap",
                "record_clusters": EXPECTED_RECORDS,
                "training_seeds_carried_per_cluster": list(EXPECTED_SEEDS),
                "replicates": replicates,
                "seed": bootstrap_seed,
                "interval": 0.95,
            },
            "forced_completion_primary_policy": "include all",
            "interpretation": (
                "model-based rationale agreement/preference; not human or "
                "expert correctness"
            ),
        },
        "evaluators": evaluators,
        "input_bindings": {
            "answer_key_sha256": sha256_bytes(answer_payload),
            "judgment_sha256": {
                f"{evaluator}|{stage}": sha256_bytes(
                    judgment_payloads[(evaluator, stage)]
                )
                for evaluator in evaluators
                for stage in STAGE_DIMENSIONS
            },
        },
        "aggregates": aggregates,
        "cross_evaluator_agreement": agreement_rows(evaluator_rows),
        "privacy": {
            "row_level_output_written": False,
            "row_identity_written": False,
            "task_identity_written": False,
            "free_text_judgment_written": False,
            "raw_judgments_written": False,
        },
        "test_accessed": False,
    }
    assert_public_privacy(
        report,
        answer_key=answer_key,
        judgment_sets=judgment_inputs,
    )

    json_payload = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    csv_payload = _csv_payload(aggregates).encode("utf-8")
    markdown_payload = _markdown_payload(report).encode("utf-8")
    for payload in (json_payload, csv_payload, markdown_payload):
        text = payload.decode("utf-8")
        forbidden_values = {
            str(row["record_id"]) for row in answer_key
        } | {
            str(row["presentation_id"]) for row in answer_key
        }
        if any(value in text for value in forbidden_values):
            raise ValueError("serialized public artifact leaks private identity")
    write_atomic(output_dir / "rationale_audit_results.json", json_payload)
    write_atomic(output_dir / "rationale_audit_results.csv", csv_payload)
    write_atomic(output_dir / "RATIONALE_AUDIT_RESULTS.md", markdown_payload)
    return report


def collect_exploratory_agent_primary_only(
    *,
    answer_key_path: Path,
    judgment_paths: Mapping[str, Path],
    output_dir: Path,
    replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Collect an explicitly non-formal two-agent primary-only analysis."""

    evaluators = sorted(judgment_paths)
    if len(evaluators) != 2:
        raise ValueError("exploratory mode requires exactly two agent judges")
    answer_payload, full_answer_key = read_jsonl_bytes(answer_key_path)
    _validate_answer_key(full_answer_key, enforce_formal_shape=True)
    comparison_counts = Counter(
        str(row["comparison"]) for row in full_answer_key
    )
    if comparison_counts != {"primary": 240, "secondary": 240}:
        raise ValueError("full answer key comparison inventory differs")
    primary_answer_key = [
        row
        for row in full_answer_key
        if str(row["comparison"]) == "primary"
    ]

    evaluator_rows: dict[str, dict[str, list[dict[str, Any]]]] = {
        evaluator: {} for evaluator in evaluators
    }
    judgment_payloads: dict[str, bytes] = {}
    judgment_inputs: list[list[dict[str, Any]]] = []
    aggregates = []
    for evaluator in evaluators:
        payload, judgments = read_jsonl_bytes(judgment_paths[evaluator])
        if len(judgments) != 240:
            raise ValueError(
                f"{evaluator}: exploratory judgment count must be 240"
            )
        judgment_payloads[evaluator] = payload
        judgment_inputs.append(judgments)
        pairs = pair_verdicts(
            primary_answer_key,
            judgments,
            stage="score_visible",
            enforce_formal_shape=False,
        )
        if len(pairs) != 120:
            raise ValueError(
                f"{evaluator}: exploratory pair count must be 120"
            )
        by_record: dict[str, set[int]] = defaultdict(set)
        for row in pairs:
            if row["comparison"] != "primary":
                raise ValueError("exploratory mode contains secondary pair")
            by_record[str(row["record_id"])].add(int(row["seed"]))
        if len(by_record) != EXPECTED_RECORDS or any(
            seeds != set(EXPECTED_SEEDS)
            for seeds in by_record.values()
        ):
            raise ValueError(
                "exploratory pairs must be 40 records by three seeds"
            )
        evaluator_rows[evaluator]["score_visible"] = pairs
        aggregates.extend(
            aggregate_rows(
                pairs,
                evaluator=evaluator,
                stage="score_visible",
                comparison_roles=("primary",),
                replicates=replicates,
                bootstrap_seed=bootstrap_seed,
            )
        )

    report = {
        "schema_version": (
            "exp54-exploratory-agent-primary-rationale-results-v1"
        ),
        "status": "EXPLORATORY_AGENT_PRIMARY_ONLY_RESULTS_COLLECTED",
        "mode": "exploratory_agent_primary_only",
        "claim_scope": (
            "Codex-agent exploratory primary score-visible preference"
        ),
        "evaluator_family_independence_satisfied": False,
        "formal_preregistered_two_family_audit_complete": False,
        "orientation_judgments_context_isolated": False,
        "orientation_bias_diagnostic_interpretable": False,
        "analysis_contract": {
            "comparison": "R3_vs_R2",
            "comparison_role": "primary",
            "stage": "score_visible",
            "orientations_per_pair": 2,
            "orientation_disagreement_rule": "pair verdict becomes tie",
            "pair_instances": 120,
            "record_clusters": EXPECTED_RECORDS,
            "training_seeds_per_cluster": list(EXPECTED_SEEDS),
            "bootstrap": {
                "method": "record-cluster hierarchical bootstrap",
                "replicates": replicates,
                "seed": bootstrap_seed,
                "interval": 0.95,
            },
            "forced_completion_primary_policy": "include all",
        },
        "agent_evaluators": evaluators,
        "input_bindings": {
            "answer_key_sha256": sha256_bytes(answer_payload),
            "judgment_sha256": {
                evaluator: sha256_bytes(judgment_payloads[evaluator])
                for evaluator in evaluators
            },
        },
        "aggregates": aggregates,
        "cross_evaluator_agreement": agreement_rows(
            evaluator_rows,
            stages=("score_visible",),
            comparison_roles=("primary",),
        ),
        "privacy": {
            "row_level_output_written": False,
            "row_identity_written": False,
            "task_identity_written": False,
            "free_text_judgment_written": False,
            "raw_judgments_written": False,
        },
        "test_accessed": False,
    }
    assert_public_privacy(
        report,
        answer_key=full_answer_key,
        judgment_sets=judgment_inputs,
    )
    json_payload = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    csv_payload = _csv_payload(aggregates).encode("utf-8")
    markdown_payload = _exploratory_markdown_payload(report).encode("utf-8")
    forbidden_values = {
        str(row["record_id"]) for row in full_answer_key
    } | {
        str(row["presentation_id"]) for row in full_answer_key
    }
    for payload in (json_payload, csv_payload, markdown_payload):
        text = payload.decode("utf-8")
        if any(value in text for value in forbidden_values):
            raise ValueError("serialized public artifact leaks private identity")
    write_atomic(
        output_dir / "exploratory_agent_primary_results.json",
        json_payload,
    )
    write_atomic(
        output_dir / "exploratory_agent_primary_results.csv",
        csv_payload,
    )
    write_atomic(
        output_dir / "EXPLORATORY_AGENT_PRIMARY_RESULTS.md",
        markdown_payload,
    )
    return report


def parse_judgment_spec(value: str) -> tuple[tuple[str, str], Path]:
    try:
        identity, path_text = value.split("=", 1)
        evaluator, stage = identity.split(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected EVALUATOR:STAGE=PATH"
        ) from exc
    if not evaluator or stage not in STAGE_DIMENSIONS or not path_text:
        raise argparse.ArgumentTypeError(
            "expected EVALUATOR:score_blind|score_visible=PATH"
        )
    return (evaluator, stage), Path(path_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = (
        REPO_ROOT
        / "thesis_exp/outputs/exp54_rar_sft/rar_v2/"
        "rationale_blind_audit"
    )
    parser.add_argument(
        "--mode",
        choices=("formal", "exploratory_agent_primary_only"),
        default="formal",
    )
    parser.add_argument(
        "--answer-key",
        type=Path,
        default=default_root / "private/answer_key.jsonl",
    )
    parser.add_argument(
        "--judgment",
        action="append",
        required=True,
        type=parse_judgment_spec,
        help=(
            "repeat four times as EVALUATOR:score_blind|score_visible=PATH"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_root / "public_results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    judgment_paths = dict(args.judgment)
    if len(judgment_paths) != len(args.judgment):
        raise ValueError("duplicate evaluator-stage judgment specification")
    if args.mode == "formal":
        report = collect_results(
            answer_key_path=args.answer_key,
            judgment_paths=judgment_paths,
            output_dir=args.output_dir,
        )
    else:
        if any(stage != "score_visible" for _name, stage in judgment_paths):
            raise ValueError(
                "exploratory mode accepts score_visible judgments only"
            )
        exploratory_paths = {
            name: path
            for (name, _stage), path in judgment_paths.items()
        }
        if len(exploratory_paths) != len(judgment_paths):
            raise ValueError("duplicate exploratory evaluator")
        report = collect_exploratory_agent_primary_only(
            answer_key_path=args.answer_key,
            judgment_paths=exploratory_paths,
            output_dir=args.output_dir,
        )
    print(
        compact_json(
            {
                "status": report["status"],
                "evaluators": report.get(
                    "evaluators", report.get("agent_evaluators")
                ),
                "public_output": args.output_dir.resolve().relative_to(
                    REPO_ROOT.resolve()
                ).as_posix(),
                "test_accessed": report["test_accessed"],
            }
        )
    )


if __name__ == "__main__":
    main()
