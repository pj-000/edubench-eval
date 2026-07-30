"""Collect post-hoc mechanism-control results on the existing frozen test."""

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
from thesis_exp.exp54_rar_sft.collect_sorc_dpo_test_results import (
    INFERENTIAL_ENDPOINTS,
    LOWER_IS_BETTER,
    _finite_seed_mean,
    _sampled_metrics,
    _score_metrics,
)
from thesis_exp.exp54_rar_sft.run_mechanism_control_test_inference_vllm import (
    ARMS as NEW_ARMS,
    DEFAULT_CHECKPOINT_LOCK,
    SEEDS,
)
from thesis_exp.exp54_rar_sft.sorc_dpo_test_execution_contract import (
    EXPECTED_TEST_BLOB_SHA1,
    file_sha256,
    read_object,
    verify_completion_receipts,
)


COMPARATOR_ARMS = ("P0_R3_SFT", "P1_FIELD_DPO")
ALL_ARMS = (*NEW_ARMS, *COMPARATOR_ARMS)
CONTRASTS = (
    (
        "C1_BLOCK_BALANCED_SFT",
        "R3_TOKENAVG",
        "P0_R3_SFT",
    ),
    (
        "C2_FIELD_LOCAL_DPO",
        "P1_FULLSEQ",
        "P1_FIELD_DPO",
    ),
    (
        "C3_ACTUAL_ERROR_NEGATIVES",
        "P1_SYN_LR5E6",
        "P1_FIELD_DPO",
    ),
)
PRIMARY_ENDPOINTS = ("MAE", "L2H_rate")
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_731
MINIMUM_VALID_REPLICATES = 9_500
DEFAULT_PLAN = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/"
    "mechanism_control_test_plan_v1.json"
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: Any) -> None:
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


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    fields: list[str] = []
    for row in values:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(values)


def _validate_new_receipts(root: Path) -> None:
    expected = {
        f"{arm.lower()}/seed_{seed}"
        for arm in NEW_ARMS
        for seed in SEEDS
    }
    actual = {
        str(path.parent.relative_to(root))
        for path in root.glob("**/completion_receipt.json")
    }
    if actual != expected:
        raise ValueError("mechanism-control receipt inventory differs")
    for arm in NEW_ARMS:
        for seed in SEEDS:
            directory = root / arm.lower() / f"seed_{seed}"
            receipt_path = directory / "completion_receipt.json"
            receipt = read_object(receipt_path)
            predictions = directory / "predictions.jsonl"
            protocol = directory / "protocol.json"
            if (
                receipt.get("status")
                != "MECHANISM_CONTROL_TEST_RUN_COMPLETE"
                or receipt.get("arm") != arm
                or int(receipt.get("seed", -1)) != seed
                or int(receipt.get("rows", -1)) != 2_218
                or receipt.get("dev_accessed") is not False
                or receipt.get("test_accessed") is not True
                or receipt.get("old_predictions_regenerated") is not False
                or receipt.get("test_adaptive_tuning_or_retraining")
                is not False
                or file_sha256(predictions)
                != receipt["predictions_sha256"]
                or file_sha256(protocol) != receipt["protocol_sha256"]
            ):
                raise ValueError(f"{arm}/seed_{seed}: receipt differs")


def _load_run(
    *,
    root: Path,
    arm: str,
    seed: int,
    new: bool,
    checkpoint_lock: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[str]]:
    directory = root / arm.lower() / f"seed_{seed}"
    protocol = read_object(directory / "protocol.json")
    predictions = _read_jsonl(directory / "predictions.jsonl")
    expected_status = (
        "MECHANISM_CONTROL_TEST_INFERENCE_COMPLETE"
        if new
        else "EXP54_SORC_DPO_ONE_TIME_TEST_INFERENCE_COMPLETE"
    )
    if (
        protocol.get("status") != expected_status
        or protocol.get("arm") != arm
        or int(protocol.get("seed", -1)) != seed
        or protocol.get("protocol_id")
        != "RAR_SFT_VLLM_COMPACT_JSON_V1"
        or protocol.get("test_git_blob_sha1") != EXPECTED_TEST_BLOB_SHA1
        or int(protocol.get("test_rows", -1)) != len(predictions)
        or protocol.get("dev_accessed") is not False
        or protocol.get("test_accessed") is not True
        or protocol.get("scientific_metrics_computed") is not False
    ):
        raise ValueError(f"{arm}/seed_{seed}: protocol differs")
    if new and (
        protocol.get("old_predictions_regenerated") is not False
        or protocol["checkpoint"]["adapter_model_sha256"]
        != checkpoint_lock["adapter_hashes"][f"{arm}:seed{seed}"]
        or protocol.get("inference_hardware_difference_disclosed") is not True
    ):
        raise ValueError(f"{arm}/seed_{seed}: new checkpoint differs")
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
    return (
        {"metrics": metrics, "forced": forced},
        gold,
        pred,
        record_ids,
    )


def _load_all(
    *,
    new_root: Path,
    old_root: Path,
) -> tuple[
    dict[tuple[str, int], dict[str, Any]],
    dict[tuple[str, int], tuple[np.ndarray, np.ndarray]],
    list[str],
]:
    _validate_new_receipts(new_root)
    verify_completion_receipts(old_root)
    checkpoint_lock = read_object(DEFAULT_CHECKPOINT_LOCK)
    runs: dict[tuple[str, int], dict[str, Any]] = {}
    arrays: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    canonical_ids: list[str] | None = None
    canonical_gold: np.ndarray | None = None
    for arm in ALL_ARMS:
        root = new_root if arm in NEW_ARMS else old_root
        for seed in SEEDS:
            run, gold, pred, record_ids = _load_run(
                root=root,
                arm=arm,
                seed=seed,
                new=arm in NEW_ARMS,
                checkpoint_lock=checkpoint_lock,
            )
            if canonical_ids is None:
                canonical_ids = record_ids
                canonical_gold = gold
            elif record_ids != canonical_ids or not np.array_equal(
                gold,
                canonical_gold,
            ):
                raise ValueError("paired test row/gold vector differs")
            runs[(arm, seed)] = run
            arrays[(arm, seed)] = (gold, pred)
    assert canonical_ids is not None
    return runs, arrays, canonical_ids


def _point_estimates(
    runs: dict[tuple[str, int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float | None]]]:
    reported = (
        "MAE",
        "L2H_rate",
        "Exact",
        "Kendall_tau_b",
        "Signed_Bias",
        "absolute_Signed_Bias",
        "Label_1_Recall",
        "Label_2_Recall",
        "Label_5_Recall",
        "H2L_rate",
        "QWK",
        "strict_parse_rate",
        "forced_completion_rate",
    )
    per_seed = []
    aggregate: dict[str, dict[str, float | None]] = {}
    for arm in ALL_ARMS:
        for seed in SEEDS:
            metrics = runs[(arm, seed)]["metrics"]
            per_seed.append({"arm": arm, "seed": seed, **metrics})
        aggregate[arm] = {
            endpoint: _finite_seed_mean(
                [
                    runs[(arm, seed)]["metrics"].get(endpoint)
                    for seed in SEEDS
                ]
            )
            for endpoint in reported
        }
    return per_seed, aggregate


def _benefit(
    endpoint: str,
    *,
    baseline: float,
    treatment: float,
) -> float:
    return (
        baseline - treatment
        if endpoint in LOWER_IS_BETTER
        else treatment - baseline
    )


def _bootstrap(
    *,
    arrays: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]],
    runs: dict[tuple[str, int], dict[str, Any]],
    record_count: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    randomizer = np.random.default_rng(BOOTSTRAP_SEED)
    deltas: dict[tuple[str, str], list[float]] = defaultdict(list)
    omitted: dict[tuple[str, str], int] = defaultdict(int)
    forced_deltas: dict[str, list[float]] = defaultdict(list)
    needed_arms = sorted(
        {arm for _name, baseline, treatment in CONTRASTS for arm in (baseline, treatment)}
    )
    for _ in range(BOOTSTRAP_REPLICATES):
        indices = randomizer.integers(0, record_count, size=record_count)
        sampled: dict[tuple[str, int], dict[str, float | None]] = {}
        forced: dict[tuple[str, int], float] = {}
        for arm in needed_arms:
            for seed in SEEDS:
                gold, pred = arrays[(arm, seed)]
                sampled[(arm, seed)] = _sampled_metrics(
                    gold,
                    pred,
                    indices,
                )
                forced[(arm, seed)] = float(
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
                        _benefit(
                            endpoint,
                            baseline=float(baseline_mean),
                            treatment=float(treatment_mean),
                        )
                    )
            forced_deltas[contrast_id].append(
                float(
                    np.mean(
                        [
                            forced[(treatment, seed)]
                            - forced[(baseline, seed)]
                            for seed in SEEDS
                        ]
                    )
                )
            )
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for contrast_id, _baseline, _treatment in CONTRASTS:
        output[contrast_id] = {}
        for endpoint in INFERENTIAL_ENDPOINTS:
            values = np.asarray(deltas[(contrast_id, endpoint)])
            valid = len(values)
            item: dict[str, Any] = {
                "valid_replicates": valid,
                "omitted_replicates": omitted[(contrast_id, endpoint)],
                "minimum_valid_replicates_met": (
                    valid >= MINIMUM_VALID_REPLICATES
                ),
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
                                (1 + int(np.sum(values <= 0)))
                                / (valid + 1),
                                (1 + int(np.sum(values >= 0)))
                                / (valid + 1),
                            ),
                        ),
                    }
                )
            output[contrast_id][endpoint] = item
        forced_values = np.asarray(forced_deltas[contrast_id])
        output[contrast_id]["forced_completion_increase"] = {
            "definition": "treatment_minus_baseline_rate",
            "bootstrap_mean": float(np.mean(forced_values)),
            "ci95_low": float(np.quantile(forced_values, 0.025)),
            "ci95_high": float(np.quantile(forced_values, 0.975)),
        }
    return output


def _add_points_and_holm(
    *,
    aggregate: dict[str, dict[str, float | None]],
    contrasts: dict[str, dict[str, dict[str, Any]]],
) -> None:
    family = []
    for contrast_id, baseline, treatment in CONTRASTS:
        for endpoint in INFERENTIAL_ENDPOINTS:
            left = aggregate[baseline][endpoint]
            right = aggregate[treatment][endpoint]
            contrasts[contrast_id][endpoint]["point_benefit"] = (
                None
                if left is None or right is None
                else _benefit(
                    endpoint,
                    baseline=float(left),
                    treatment=float(right),
                )
            )
        forced_delta = (
            float(aggregate[treatment]["forced_completion_rate"])
            - float(aggregate[baseline]["forced_completion_rate"])
        )
        contrasts[contrast_id]["forced_completion_increase"][
            "point_treatment_minus_baseline"
        ] = forced_delta
        for endpoint in PRIMARY_ENDPOINTS:
            value = contrasts[contrast_id][endpoint].get(
                "two_sided_p_unadjusted"
            )
            if value is not None:
                family.append((float(value), contrast_id, endpoint))
    family.sort()
    running = 0.0
    size = len(family)
    for rank, (p_value, contrast_id, endpoint) in enumerate(family):
        running = max(running, min(1.0, (size - rank) * p_value))
        contrasts[contrast_id][endpoint]["holm_adjusted_p"] = min(
            1.0,
            running,
        )


def _classify(
    contrasts: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    output = []
    for contrast_id, baseline, treatment in CONTRASTS:
        primary = {
            endpoint: contrasts[contrast_id][endpoint]["point_benefit"]
            for endpoint in PRIMARY_ENDPOINTS
        }
        significant_positive = any(
            contrasts[contrast_id][endpoint].get("holm_adjusted_p", 1.0)
            < 0.05
            and float(primary[endpoint]) > 0
            for endpoint in PRIMARY_ENDPOINTS
        )
        significant_harm = any(
            contrasts[contrast_id][endpoint].get("ci95_high") is not None
            and float(contrasts[contrast_id][endpoint]["ci95_high"]) < 0
            for endpoint in PRIMARY_ENDPOINTS
        )
        if significant_harm:
            classification = "MATERIAL_HARM"
        elif significant_positive:
            classification = "STRONG_SUPPORT"
        elif any(float(value) > 0 for value in primary.values()):
            classification = "DIRECTIONAL_SUPPORT"
        elif all(abs(float(value)) < 0.01 for value in primary.values()):
            classification = "APPROXIMATELY_ZERO"
        else:
            classification = "UNSUPPORTED"
        output.append(
            {
                "contrast_id": contrast_id,
                "baseline_arm": baseline,
                "treatment_arm": treatment,
                "classification": classification,
                "primary_point_benefits": primary,
            }
        )
    return output


def collect(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    plan = read_object(args.plan)
    if (
        plan.get("status")
        != "FROZEN_POSTHOC_MECHANISM_TEST_ON_EXISTING_BENCHMARK"
        or plan.get("old_predictions_regenerated") is not False
        or plan.get("test_adaptive_tuning_or_retraining") is not False
        or plan.get("execution_hardware_difference", {}).get(
            "must_be_disclosed"
        )
        is not True
    ):
        raise ValueError("mechanism-control test plan differs")
    runs, arrays, record_ids = _load_all(
        new_root=args.new_root,
        old_root=args.old_root,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    per_seed, aggregate = _point_estimates(runs)
    contrasts = _bootstrap(
        arrays=arrays,
        runs=runs,
        record_count=len(record_ids),
    )
    _add_points_and_holm(
        aggregate=aggregate,
        contrasts=contrasts,
    )
    interpretations = _classify(contrasts)
    _write_csv(args.output_dir / "per_seed_metrics.csv", per_seed)
    _write_csv(
        args.output_dir / "multiseed_summary.csv",
        [{"arm": arm, **aggregate[arm]} for arm in ALL_ARMS],
    )
    _write_json(args.output_dir / "paired_bootstrap.json", contrasts)
    _write_json(
        args.output_dir / "final_results.json",
        {
            "schema_version": "exp54-mechanism-control-test-results-v1",
            "status": "MECHANISM_CONTROL_POSTHOC_TEST_COMPLETE",
            "test_record_count": len(record_ids),
            "new_checkpoint_run_count": 9,
            "new_arms": list(NEW_ARMS),
            "reused_comparator_arms": list(COMPARATOR_ARMS),
            "seeds": list(SEEDS),
            "multiseed_summary": aggregate,
            "contrasts": contrasts,
            "interpretation": interpretations,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "old_predictions_regenerated": False,
            "test_adaptive_tuning_or_retraining": False,
            "inference_hardware_difference_disclosed": True,
            "rationale_quality_claim_allowed": False,
            "dev_accessed": False,
            "test_accessed": True,
            "test_rerun_allowed": False,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    return parser.parse_args()


if __name__ == "__main__":
    collect(parse_args())
