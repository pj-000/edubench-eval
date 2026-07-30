"""Collect the frozen Exp54 one-time test only after all 12 runs complete."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.sorc_dpo_test_execution_contract import (
    ARMS,
    EXPECTED_TEST_BLOB_SHA1,
    EXPECTED_XGRAMMAR_SHA256,
    PREREGISTRATION_PATH,
    PREREGISTRATION_SHA256,
    SEEDS,
    load_preregistration,
    read_object,
    validate_runtime_source_closure,
    verify_completion_receipts,
)


BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_731
MINIMUM_VALID_REPLICATES = 9_500
PRIMARY_ENDPOINTS = ("MAE", "L2H_rate")
INFERENTIAL_ENDPOINTS = (
    "MAE",
    "L2H_rate",
    "Exact",
    "Kendall_tau_b",
    "absolute_Signed_Bias",
    "Label_5_Recall",
    "H2L_rate",
    "QWK",
)
LOWER_IS_BETTER = {
    "MAE",
    "L2H_rate",
    "absolute_Signed_Bias",
    "H2L_rate",
}
CONTRASTS = (
    ("H1_FIELD_DPO", "P0_R3_SFT", "P1_FIELD_DPO"),
    ("H2_ORDINAL_OFFSET", "P1_FIELD_DPO", "P2_SORC_SCORE"),
    ("H3_RATIONALE_BLOCK", "P2_SORC_SCORE", "P3_JOINT_SORC"),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    fields: list[str] = []
    for row in values:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)


def _quadratic_weighted_kappa(
    gold: np.ndarray,
    pred: np.ndarray,
) -> float:
    observed = np.zeros((5, 5), dtype=np.float64)
    np.add.at(observed, (gold - 1, pred - 1), 1.0)
    gold_hist = observed.sum(axis=1)
    pred_hist = observed.sum(axis=0)
    expected = np.outer(gold_hist, pred_hist) / float(len(gold))
    coordinates = np.arange(5, dtype=np.float64)
    weights = (
        (coordinates[:, None] - coordinates[None, :]) ** 2 / 16.0
    )
    expected_disagreement = float(np.sum(weights * expected))
    if expected_disagreement == 0.0:
        return float("nan")
    return 1.0 - float(np.sum(weights * observed)) / expected_disagreement


def _kendall_or_nan(gold: np.ndarray, pred: np.ndarray) -> float:
    gold_pairs = np.zeros((5, 5), dtype=np.int64)
    for gold_value, pred_value in zip(gold, pred, strict=True):
        gold_pairs[int(gold_value) - 1, int(pred_value) - 1] += 1
    concordant = 0
    discordant = 0
    for i in range(5):
        for j in range(5):
            count = int(gold_pairs[i, j])
            concordant += count * int(gold_pairs[i + 1 :, j + 1 :].sum())
            discordant += count * int(gold_pairs[i + 1 :, :j].sum())
    tied_gold = sum(
        int(count) * (int(count) - 1) // 2
        for count in gold_pairs.sum(axis=1)
    )
    tied_pred = sum(
        int(count) * (int(count) - 1) // 2
        for count in gold_pairs.sum(axis=0)
    )
    tied_both = sum(
        int(count) * (int(count) - 1) // 2
        for count in gold_pairs.reshape(-1)
    )
    denominator = math.sqrt(
        (concordant + discordant + tied_gold - tied_both)
        * (concordant + discordant + tied_pred - tied_both)
    )
    if denominator == 0.0:
        return float("nan")
    return float((concordant - discordant) / denominator)


def _score_metrics(gold: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    low = gold <= 2
    high = gold >= 4
    values: dict[str, Any] = {
        "n": int(len(gold)),
        "Exact": float(np.mean(pred == gold)),
        "MAE": float(np.mean(np.abs(pred - gold))),
        "Kendall_tau_b": _kendall_or_nan(gold, pred),
        "Signed_Bias": float(np.mean(pred - gold)),
        "L2H_count": int(np.sum(low & (pred >= 4))),
        "L2H_rate": (
            float(np.mean(pred[low] >= 4)) if bool(np.any(low)) else None
        ),
        "low_n": int(np.sum(low)),
        "H2L_count": int(np.sum(high & (pred <= 2))),
        "H2L_rate": (
            float(np.mean(pred[high] <= 2)) if bool(np.any(high)) else None
        ),
        "high_n": int(np.sum(high)),
        "QWK": _quadratic_weighted_kappa(gold, pred),
    }
    values["absolute_Signed_Bias"] = abs(values["Signed_Bias"])
    for label in (1, 2, 3, 4, 5):
        selected = gold == label
        values[f"Label_{label}_n"] = int(np.sum(selected))
        values[f"Label_{label}_correct_count"] = int(
            np.sum(selected & (pred == label))
        )
        values[f"Label_{label}_Recall"] = (
            float(np.mean(pred[selected] == label))
            if bool(np.any(selected))
            else None
        )
    return values


def _sampled_metrics(
    gold: np.ndarray,
    pred: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float | None]:
    values = _score_metrics(gold[indices], pred[indices])
    return {
        endpoint: (
            None
            if values[endpoint] is None
            or not math.isfinite(float(values[endpoint]))
            else float(values[endpoint])
        )
        for endpoint in INFERENTIAL_ENDPOINTS
    }


def _benefit_delta(
    endpoint: str,
    *,
    baseline: float,
    treatment: float,
) -> float:
    if endpoint in LOWER_IS_BETTER:
        return baseline - treatment
    return treatment - baseline


def _load_all_runs(
    output_root: Path,
    preregistration: dict[str, Any] | None = None,
) -> tuple[
    dict[tuple[str, int], dict[str, Any]],
    dict[tuple[str, int], tuple[np.ndarray, np.ndarray]],
    list[str],
]:
    # This gate reads only receipts and file hashes. No prediction is parsed
    # until the complete 4 x 3 grid has passed.
    verify_completion_receipts(output_root)
    if preregistration is None:
        preregistration = load_preregistration()
    runs: dict[tuple[str, int], dict[str, Any]] = {}
    arrays: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    canonical_ids: list[str] | None = None
    canonical_gold: np.ndarray | None = None
    canonical_test_hash: str | None = None
    for arm in ARMS:
        for seed in SEEDS:
            run_dir = output_root / arm.lower() / f"seed_{seed}"
            protocol = read_object(run_dir / "protocol.json")
            predictions = read_jsonl(run_dir / "predictions.jsonl")
            if (
                protocol.get("status")
                != "EXP54_SORC_DPO_ONE_TIME_TEST_INFERENCE_COMPLETE"
                or protocol.get("arm") != arm
                or protocol.get("seed") != seed
                or protocol.get("protocol_id")
                != "RAR_SFT_VLLM_COMPACT_JSON_V1"
                or protocol.get("dev_accessed") is not False
                or protocol.get("test_accessed") is not True
                or protocol.get("scientific_metrics_computed") is not False
                or int(protocol.get("test_rows", -1)) != len(predictions)
                or protocol.get("test_git_blob_sha1")
                != EXPECTED_TEST_BLOB_SHA1
                or protocol.get("preregistration_sha256")
                != PREREGISTRATION_SHA256
                or protocol.get("generation")
                != {
                    "do_sample": False,
                    "temperature": 0.0,
                    "max_new_tokens": 256,
                    "max_model_len": 1796,
                }
                or protocol.get("backend", {}).get("xgrammar_source_sha256")
                != EXPECTED_XGRAMMAR_SHA256
                or protocol.get("checkpoint", {}).get(
                    "adapter_model_sha256"
                )
                != preregistration["arms"][arm][
                    f"seed_{seed}_adapter_model_sha256"
                ]
            ):
                raise ValueError(f"{arm}/seed_{seed}: test protocol differs")
            test_hash = str(protocol["test_sha256"])
            if canonical_test_hash is None:
                canonical_test_hash = test_hash
            elif test_hash != canonical_test_hash:
                raise ValueError("test source SHA-256 differs across runs")
            positions = [int(row["row_position"]) for row in predictions]
            record_ids = [str(row["record_id"]) for row in predictions]
            gold = np.asarray(
                [int(row["label_5"]) for row in predictions],
                dtype=np.int64,
            )
            pred = np.asarray(
                [int(row["prediction"]["score"]) for row in predictions],
                dtype=np.int64,
            )
            if (
                positions != list(range(len(predictions)))
                or len(set(record_ids)) != len(record_ids)
                or any(row.get("parse_success") is not True for row in predictions)
                or np.any(gold < 1)
                or np.any(gold > 5)
                or np.any(pred < 1)
                or np.any(pred > 5)
            ):
                raise ValueError(f"{arm}/seed_{seed}: prediction contract differs")
            if canonical_ids is None:
                canonical_ids = record_ids
                canonical_gold = gold
            elif record_ids != canonical_ids or not np.array_equal(
                gold,
                canonical_gold,
            ):
                raise ValueError("paired test record or gold vector differs")
            forced = np.asarray(
                [bool(row.get("forced_completion", False)) for row in predictions],
                dtype=np.bool_,
            )
            metrics = _score_metrics(gold, pred)
            metrics.update(
                {
                    "strict_parse_rate": 1.0,
                    "forced_completion_count": int(np.sum(forced)),
                    "forced_completion_rate": float(np.mean(forced)),
                }
            )
            runs[(arm, seed)] = {
                "arm": arm,
                "seed": seed,
                "metrics": metrics,
                "forced": forced,
                "test_sha256": test_hash,
                "rows": len(predictions),
            }
            arrays[(arm, seed)] = (gold, pred)
    assert canonical_ids is not None
    return runs, arrays, canonical_ids


def _finite_seed_mean(values: list[float | None]) -> float | None:
    if any(value is None or not math.isfinite(float(value)) for value in values):
        return None
    return float(np.mean([float(value) for value in values]))


def _point_estimates(
    runs: dict[tuple[str, int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float | None]]]:
    per_seed = []
    aggregate: dict[str, dict[str, float | None]] = {}
    reported = (
        "MAE",
        "L2H_rate",
        "Exact",
        "Kendall_tau_b",
        "Signed_Bias",
        "absolute_Signed_Bias",
        "Label_5_Recall",
        "H2L_rate",
        "QWK",
        "Label_1_Recall",
        "Label_2_Recall",
        "strict_parse_rate",
        "forced_completion_rate",
    )
    for arm in ARMS:
        for seed in SEEDS:
            metrics = runs[(arm, seed)]["metrics"]
            per_seed.append({"arm": arm, "seed": seed, **metrics})
        aggregate[arm] = {
            endpoint: _finite_seed_mean(
                [runs[(arm, seed)]["metrics"].get(endpoint) for seed in SEEDS]
            )
            for endpoint in reported
        }
    return per_seed, aggregate


def _bootstrap(
    arrays: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]],
    runs: dict[tuple[str, int], dict[str, Any]],
    record_count: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    randomizer = np.random.default_rng(BOOTSTRAP_SEED)
    deltas: dict[tuple[str, str], list[float]] = defaultdict(list)
    omitted: dict[tuple[str, str], int] = defaultdict(int)
    forced_deltas: dict[str, list[float]] = defaultdict(list)
    for _ in range(BOOTSTRAP_REPLICATES):
        indices = randomizer.integers(0, record_count, size=record_count)
        sampled: dict[tuple[str, int], dict[str, float | None]] = {}
        forced_rates: dict[tuple[str, int], float] = {}
        for arm in ARMS:
            for seed in SEEDS:
                gold, pred = arrays[(arm, seed)]
                sampled[(arm, seed)] = _sampled_metrics(gold, pred, indices)
                forced_rates[(arm, seed)] = float(
                    np.mean(runs[(arm, seed)]["forced"][indices])
                )
        for contrast_id, baseline, treatment in CONTRASTS:
            for endpoint in INFERENTIAL_ENDPOINTS:
                baseline_mean = _finite_seed_mean(
                    [
                        sampled[(baseline, seed)][endpoint]
                        for seed in SEEDS
                    ]
                )
                treatment_mean = _finite_seed_mean(
                    [
                        sampled[(treatment, seed)][endpoint]
                        for seed in SEEDS
                    ]
                )
                key = (contrast_id, endpoint)
                if baseline_mean is None or treatment_mean is None:
                    omitted[key] += 1
                else:
                    deltas[key].append(
                        _benefit_delta(
                            endpoint,
                            baseline=baseline_mean,
                            treatment=treatment_mean,
                        )
                    )
            forced_deltas[contrast_id].append(
                float(
                    np.mean(
                        [
                            forced_rates[(treatment, seed)]
                            - forced_rates[(baseline, seed)]
                            for seed in SEEDS
                        ]
                    )
                )
            )

    results: dict[str, dict[str, dict[str, Any]]] = {}
    for contrast_id, _baseline, _treatment in CONTRASTS:
        results[contrast_id] = {}
        for endpoint in INFERENTIAL_ENDPOINTS:
            values = np.asarray(deltas[(contrast_id, endpoint)])
            valid = int(len(values))
            item: dict[str, Any] = {
                "valid_replicates": valid,
                "omitted_replicates": omitted[(contrast_id, endpoint)],
                "minimum_valid_replicates_met": valid
                >= MINIMUM_VALID_REPLICATES,
            }
            if valid >= MINIMUM_VALID_REPLICATES:
                item.update(
                    {
                        "bootstrap_mean": float(np.mean(values)),
                        "ci95_low": float(np.quantile(values, 0.025)),
                        "ci95_high": float(np.quantile(values, 0.975)),
                        "two_sided_p_unadjusted": min(
                            1.0,
                            2.0
                            * min(
                                (1 + int(np.sum(values <= 0))) / (valid + 1),
                                (1 + int(np.sum(values >= 0))) / (valid + 1),
                            ),
                        ),
                    }
                )
            results[contrast_id][endpoint] = item
        forced = np.asarray(forced_deltas[contrast_id])
        results[contrast_id]["forced_completion_increase"] = {
            "definition": "treatment_minus_baseline_rate",
            "valid_replicates": BOOTSTRAP_REPLICATES,
            "omitted_replicates": 0,
            "bootstrap_mean": float(np.mean(forced)),
            "ci95_low": float(np.quantile(forced, 0.025)),
            "ci95_high": float(np.quantile(forced, 0.975)),
        }
    return results


def _holm_adjust(
    bootstrap: dict[str, dict[str, dict[str, Any]]],
) -> bool:
    family = []
    for contrast_id, _baseline, _treatment in CONTRASTS:
        for endpoint in PRIMARY_ENDPOINTS:
            item = bootstrap[contrast_id][endpoint]
            p_value = item.get("two_sided_p_unadjusted")
            if p_value is None:
                continue
            family.append((float(p_value), contrast_id, endpoint))
    if len(family) != 6:
        for contrast_id, _baseline, _treatment in CONTRASTS:
            for endpoint in PRIMARY_ENDPOINTS:
                bootstrap[contrast_id][endpoint][
                    "holm_family_resolved"
                ] = False
        return False
    family.sort()
    running = 0.0
    family_size = len(family)
    for rank, (p_value, contrast_id, endpoint) in enumerate(family):
        adjusted = min(1.0, (family_size - rank) * p_value)
        running = max(running, adjusted)
        bootstrap[contrast_id][endpoint][
            "holm_adjusted_p"
        ] = min(1.0, running)
        bootstrap[contrast_id][endpoint]["holm_family_resolved"] = True
    return True


def _add_point_deltas(
    *,
    aggregate: dict[str, dict[str, float | None]],
    bootstrap: dict[str, dict[str, dict[str, Any]]],
) -> None:
    for contrast_id, baseline, treatment in CONTRASTS:
        for endpoint in INFERENTIAL_ENDPOINTS:
            baseline_value = aggregate[baseline][endpoint]
            treatment_value = aggregate[treatment][endpoint]
            bootstrap[contrast_id][endpoint]["point_benefit"] = (
                None
                if baseline_value is None or treatment_value is None
                else _benefit_delta(
                    endpoint,
                    baseline=float(baseline_value),
                    treatment=float(treatment_value),
                )
            )
        forced_point = (
            float(aggregate[treatment]["forced_completion_rate"])
            - float(aggregate[baseline]["forced_completion_rate"])
        )
        bootstrap[contrast_id]["forced_completion_increase"][
            "point_treatment_minus_baseline"
        ] = forced_point


def _interpret(
    *,
    preregistration: dict[str, Any],
    aggregate: dict[str, dict[str, float | None]],
    bootstrap: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    sesoi = preregistration["statistical_protocol"][
        "smallest_effect_size_of_interest"
    ]
    outcomes = []
    for contrast_id, baseline, treatment in CONTRASTS:
        result = bootstrap[contrast_id]
        primary_harms = []
        guardrail_harms = []
        required_endpoints = (
            *PRIMARY_ENDPOINTS,
            "Exact",
            "Kendall_tau_b",
            "absolute_Signed_Bias",
            "Label_5_Recall",
            "H2L_rate",
        )
        unresolved_endpoints = [
            endpoint
            for endpoint in required_endpoints
            if not result[endpoint]["minimum_valid_replicates_met"]
            or result[endpoint].get("point_benefit") is None
        ]
        holm_family_resolved = all(
            result[endpoint].get("holm_family_resolved") is True
            for endpoint in PRIMARY_ENDPOINTS
        )
        if not holm_family_resolved:
            unresolved_endpoints.extend(
                endpoint
                for endpoint in PRIMARY_ENDPOINTS
                if endpoint not in unresolved_endpoints
            )
        for endpoint in PRIMARY_ENDPOINTS:
            item = result[endpoint]
            threshold = float(sesoi[endpoint])
            if (
                item.get("point_benefit") is not None
                and float(item["point_benefit"]) <= -threshold
                and item.get("ci95_high") is not None
                and float(item["ci95_high"]) < 0.0
            ):
                primary_harms.append(endpoint)
        for endpoint in (
            "Exact",
            "Kendall_tau_b",
            "absolute_Signed_Bias",
            "Label_5_Recall",
            "H2L_rate",
        ):
            sesoi_key = (
                "Kendall_tau_b"
                if endpoint == "Kendall_tau_b"
                else endpoint
            )
            item = result[endpoint]
            threshold = float(sesoi[sesoi_key])
            if (
                item.get("point_benefit") is not None
                and float(item["point_benefit"]) <= -threshold
                and item.get("ci95_high") is not None
                and float(item["ci95_high"]) < 0.0
            ):
                guardrail_harms.append(endpoint)
        operational_failure = any(
            float(aggregate[arm]["strict_parse_rate"]) < 1.0
            for arm in (baseline, treatment)
        )
        forced = result["forced_completion_increase"]
        forced_harm = (
            float(forced["point_treatment_minus_baseline"]) >= 0.05
            and float(forced["ci95_low"]) > 0.0
        )
        primary_points = {
            endpoint: (
                None
                if result[endpoint]["point_benefit"] is None
                else float(result[endpoint]["point_benefit"])
            )
            for endpoint in PRIMARY_ENDPOINTS
        }
        any_direction_significant = {
            endpoint: (
                result[endpoint].get("holm_adjusted_p") is not None
                and float(result[endpoint]["holm_adjusted_p"]) < 0.05
            )
            for endpoint in PRIMARY_ENDPOINTS
        }
        primary_significant = {
            endpoint: (
                any_direction_significant[endpoint]
                and primary_points[endpoint] is not None
                and float(primary_points[endpoint]) > 0.0
            )
            for endpoint in PRIMARY_ENDPOINTS
        }
        no_disqualifier = not (
            unresolved_endpoints
            or not holm_family_resolved
            or primary_harms
            or guardrail_harms
            or operational_failure
            or forced_harm
        )
        strong = no_disqualifier and any(primary_significant.values())
        directional = (
            not strong
            and no_disqualifier
            and any(
                value is not None and value > 0.0
                for value in primary_points.values()
            )
            and all(
                primary_points[endpoint] is not None
                and float(primary_points[endpoint])
                > -float(sesoi[endpoint])
                for endpoint in PRIMARY_ENDPOINTS
            )
        )
        approximately_zero = (
            no_disqualifier
            and not any(any_direction_significant.values())
            and all(
                primary_points[endpoint] is not None
                and abs(float(primary_points[endpoint]))
                < float(sesoi[endpoint])
                for endpoint in PRIMARY_ENDPOINTS
            )
        )
        if unresolved_endpoints or not holm_family_resolved:
            classification = "UNRESOLVED"
        elif primary_harms or guardrail_harms:
            classification = "MATERIAL_HARM"
        elif strong:
            classification = "STRONG_SUPPORT"
        elif directional:
            classification = "DIRECTIONAL_SUPPORT"
        elif approximately_zero:
            classification = "APPROXIMATELY_ZERO"
        else:
            classification = "UNSUPPORTED"
        outcomes.append(
            {
                "contrast_id": contrast_id,
                "baseline_arm": baseline,
                "treatment_arm": treatment,
                "classification": classification,
                "primary_point_benefits": primary_points,
                "unresolved_endpoints": sorted(set(unresolved_endpoints)),
                "holm_family_resolved": holm_family_resolved,
                "primary_harms": primary_harms,
                "guardrail_harms": guardrail_harms,
                "operational_failure": operational_failure,
                "forced_completion_diagnostic_harm": forced_harm,
                "P3_estimand_is_bundled_not_FLOP_matched": (
                    contrast_id == "H3_RATIONALE_BLOCK"
                ),
            }
        )
    return outcomes


def _format_value(value: Any, spec: str) -> str:
    if value is None:
        return "NA"
    numeric = float(value)
    if not math.isfinite(numeric):
        return "NA"
    return format(numeric, spec)


def _format_benefit_interval(item: dict[str, Any]) -> str:
    if any(
        item.get(field) is None
        for field in ("point_benefit", "ci95_low", "ci95_high")
    ):
        return "NA"
    return (
        f"{float(item['point_benefit']):+.4f} "
        f"[{float(item['ci95_low']):+.4f}, "
        f"{float(item['ci95_high']):+.4f}]"
    )


def collect(args: argparse.Namespace) -> None:
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError(args.output_dir)
    validate_runtime_source_closure(repo_root=REPO_ROOT)
    preregistration = load_preregistration(args.preregistration)
    runs, arrays, record_ids = _load_all_runs(
        args.test_root,
        preregistration,
    )
    # The result directory is created only after the all-12 completion gate.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    per_seed, aggregate = _point_estimates(runs)
    bootstrap = _bootstrap(arrays, runs, len(record_ids))
    _add_point_deltas(aggregate=aggregate, bootstrap=bootstrap)
    holm_family_resolved = _holm_adjust(bootstrap)
    interpretation = _interpret(
        preregistration=preregistration,
        aggregate=aggregate,
        bootstrap=bootstrap,
    )

    aggregate_rows = [
        {"arm": arm, **aggregate[arm]} for arm in ARMS
    ]
    write_csv(args.output_dir / "per_seed_metrics.csv", per_seed)
    write_csv(args.output_dir / "multiseed_summary.csv", aggregate_rows)
    write_json(args.output_dir / "paired_bootstrap.json", bootstrap)
    write_json(
        args.output_dir / "final_results.json",
        {
            "status": "EXP54_SORC_DPO_ONE_TIME_TEST_COMPLETE",
            "test_record_count": len(record_ids),
            "run_count": 12,
            "arms": list(ARMS),
            "seeds": list(SEEDS),
            "multiseed_summary": aggregate,
            "contrasts": bootstrap,
            "interpretation": interpretation,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "holm_family_size": 6,
            "holm_family_resolved": holm_family_resolved,
            "all_metrics_computed_per_seed_then_unweighted_mean": True,
            "P3_not_FLOP_matched": True,
            "rationale_quality_claim_allowed": False,
            "dev_accessed": False,
            "test_accessed": True,
            "test_rerun_allowed": False,
        },
    )

    lines = [
        "# Exp54 SORC-DPO one-time test results",
        "",
        "| Arm | MAE↓ | L2H↓ | Exact↑ | Kendall↑ | Bias | Recall-2↑ | Recall-5↑ | H2L↓ | QWK↑ | Forced close |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = aggregate[arm]
        lines.append(
            f"| {arm} | {_format_value(row['MAE'], '.4f')} "
            f"| {_format_value(row['L2H_rate'], '.2%')} "
            f"| {_format_value(row['Exact'], '.4f')} "
            f"| {_format_value(row['Kendall_tau_b'], '.4f')} "
            f"| {_format_value(row['Signed_Bias'], '+.4f')} "
            f"| {_format_value(row['Label_2_Recall'], '.2%')} "
            f"| {_format_value(row['Label_5_Recall'], '.2%')} "
            f"| {_format_value(row['H2L_rate'], '.2%')} "
            f"| {_format_value(row['QWK'], '.4f')} "
            f"| {_format_value(row['forced_completion_rate'], '.2%')} |"
        )
    lines.extend(
        [
            "",
            "| Contrast | MAE benefit [95% CI] | Holm p | L2H benefit [95% CI] | Holm p | Classification |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    by_id = {row["contrast_id"]: row for row in interpretation}
    for contrast_id, _baseline, _treatment in CONTRASTS:
        mae = bootstrap[contrast_id]["MAE"]
        l2h = bootstrap[contrast_id]["L2H_rate"]
        lines.append(
            f"| {contrast_id} "
            f"| {_format_benefit_interval(mae)} "
            f"| {_format_value(mae.get('holm_adjusted_p'), '.4f')} "
            f"| {_format_benefit_interval(l2h)} "
            f"| {_format_value(l2h.get('holm_adjusted_p'), '.4f')} "
            f"| {by_id[contrast_id]['classification']} |"
        )
    lines.extend(
        [
            "",
            "Positive contrast values mean benefit under the preregistered "
            "endpoint-specific sign convention.",
            "",
            "P3−P2 remains a bundled, non-FLOP-matched effect and cannot "
            "establish improved rationale quality.",
            "",
            "This was the one-time test. Result-driven reruns are forbidden.",
        ]
    )
    (args.output_dir / "report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=PREREGISTRATION_PATH,
    )
    return parser.parse_args()


if __name__ == "__main__":
    collect(parse_args())
