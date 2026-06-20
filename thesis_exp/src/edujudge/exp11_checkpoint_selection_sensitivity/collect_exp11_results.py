"""Collect Exp11 checkpoint-selection sensitivity results."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp11_checkpoint_selection_sensitivity import (
    DEFAULT_GAMMA,
    DEFAULT_MAE_GUARD_DELTA,
    DEFAULT_MONO_BETA,
    DIAGNOSTIC_RULES,
    EXP11_LOCAL_RUNS_DIR,
    EXP11_REPORTS_DIR,
    EXP11_TABLES_DIR,
    SELECTION_RULES,
    ensure_exp11_dirs,
)
from thesis_exp.src.edujudge.utils.io import read_csv, relpath, write_csv, write_text


EPOCH_FIELDS = [
    "seed",
    "epoch",
    "global_step",
    "split",
    "diagnostic",
    "MAE",
    "QWK",
    "Accuracy",
    "Acc@5",
    "low_to_high",
    "low_to_high_count",
    "true_low_score_count",
    "label1_low_to_high",
    "label1_low_to_high_count",
    "label1_low_score_count",
    "label2_low_to_high",
    "label2_low_to_high_count",
    "label2_low_score_count",
    "monotonic_violation",
    "p1_lt_p2",
    "p2_lt_p3",
    "p3_lt_p4",
]
SOFT_FIELDS = [
    "seed",
    "epoch",
    "global_step",
    "split",
    "diagnostic",
    "gamma",
    "soft_low_to_high",
    "p_gt_3_low_mean",
    "label2_soft_low_to_high",
    "label2_p_gt_3_mean",
]
SELECTION_FIELDS = [
    "seed",
    "selection_rule",
    "selected_epoch",
    "selected_global_step",
    "rule_type",
    "uses_test_for_selection",
    "dev_MAE",
    "dev_QWK",
    "dev_low_to_high",
    "dev_low_to_high_count",
    "dev_soft_low_to_high",
    "dev_label2_soft_low_to_high",
    "dev_p_gt_3_low_mean",
    "dev_monotonic_violation",
    "selection_score",
    "tie_breaker",
]
TEST_SELECTED_FIELDS = [
    "seed",
    "selection_rule",
    "selected_epoch",
    "selected_global_step",
    "test_MAE",
    "test_QWK",
    "test_Accuracy",
    "test_Acc@5",
    "test_low_to_high",
    "test_low_to_high_count",
    "test_true_low_score_count",
    "test_label1_low_to_high",
    "test_label1_low_to_high_count",
    "test_label2_low_to_high",
    "test_label2_low_to_high_count",
    "test_monotonic_violation",
    "test_p1_lt_p2",
    "test_p2_lt_p3",
    "test_p3_lt_p4",
    "test_soft_low_to_high",
    "test_label2_soft_low_to_high",
    "test_p_gt_3_low_mean",
]
DELTA_FIELDS = [
    "seed",
    "selection_rule",
    "baseline_rule",
    "selected_epoch",
    "baseline_selected_epoch",
    "delta_test_MAE",
    "delta_test_QWK",
    "delta_test_Acc@5",
    "delta_test_low_to_high",
    "delta_test_low_to_high_count",
    "delta_test_monotonic_violation",
    "interpretation",
]
CONFIG_FIELDS = ["seed", "item", "value", "note"]
INVENTORY_FIELDS = [
    "seed",
    "epoch",
    "global_step",
    "checkpoint_path_local",
    "checkpoint_exists",
    "checkpoint_size_mb",
    "git_tracked",
    "note",
]
MULTISEED_FIELDS = [
    "seed",
    "selection_rule",
    "selected_epoch",
    "test_MAE",
    "test_QWK",
    "test_Acc@5",
    "test_low_to_high",
    "test_low_to_high_count",
    "test_monotonic_violation",
]
AGG_FIELDS = [
    "selection_rule",
    "num_seeds",
    "mean_test_MAE",
    "std_test_MAE",
    "mean_test_QWK",
    "std_test_QWK",
    "mean_test_low_to_high",
    "std_test_low_to_high",
    "mean_test_low_to_high_count",
    "mean_test_monotonic_violation",
    "std_test_monotonic_violation",
]


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _fmt(value: Any, digits: int = 4) -> str:
    value_float = _float(value)
    return f"{value_float:.{digits}f}" if math.isfinite(value_float) else str(value)


def _read_csv_if_exists(path: Path) -> list[dict[str, str]]:
    return read_csv(path) if path.exists() else []


def _seed_dirs(runs_root: Path) -> list[Path]:
    return sorted(path for path in runs_root.glob("seed_*/run") if path.is_dir())


def _seed_from_run_dir(run_dir: Path) -> int:
    parent = run_dir.parent.name
    return int(parent.rsplit("_", 1)[-1])


def _key(row: dict[str, Any]) -> tuple[int, int]:
    return (_int(row.get("seed")), _int(row.get("epoch")))


def _merge_epoch_soft(epoch_rows: list[dict[str, str]], soft_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    soft_by_key = {_key(row): row for row in soft_rows}
    merged = []
    for row in epoch_rows:
        soft = soft_by_key.get(_key(row), {})
        merged.append({**row, **{f"soft_{key}": value for key, value in soft.items() if key not in row}})
    return merged


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (_int(row.get("seed")), _int(row.get("epoch")), str(row.get("split", ""))))


def load_run_rows(runs_root: Path) -> dict[str, list[dict[str, Any]]]:
    dev_epoch: list[dict[str, Any]] = []
    test_epoch: list[dict[str, Any]] = []
    dev_soft: list[dict[str, Any]] = []
    test_soft: list[dict[str, Any]] = []
    dev_low: list[dict[str, Any]] = []
    test_low: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    for run_dir in _seed_dirs(runs_root):
        seed = _seed_from_run_dir(run_dir)
        tables = run_dir / "tables"
        dev_epoch.extend(_read_csv_if_exists(tables / "epoch_metrics_dev.csv"))
        test_epoch.extend(_read_csv_if_exists(tables / "epoch_metrics_test_diagnostic.csv"))
        dev_soft.extend(_read_csv_if_exists(tables / "soft_risk_metrics_dev.csv"))
        test_soft.extend(_read_csv_if_exists(tables / "soft_risk_metrics_test_diagnostic.csv"))
        dev_low.extend(_read_csv_if_exists(tables / "low_score_by_epoch_dev.csv"))
        test_low.extend(_read_csv_if_exists(tables / "low_score_by_epoch_test_diagnostic.csv"))
        config_path = run_dir.parent / "checkpoints" / "best" / "training_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        for item in [
            "run_id",
            "seed",
            "num_train_epochs",
            "data_dir",
            "qd_b1_checkpoint_dir",
            "pair_dataset_size_train",
            "pair_dataset_size_dev",
            "max_pairs_per_record",
            "max_pairs_per_low_record",
            "lambda_point",
            "lambda_pair",
            "lambda_anchor",
            "lambda_mono",
            "dataloader_mode",
            "force_pair_training",
            "learning_rate",
            "warmup_ratio",
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "gradient_accumulation_steps",
            "save_each_epoch",
            "eval_each_epoch",
            "keep_epoch_checkpoints_local",
            "selection_rules_enabled",
            "soft_risk_gamma",
        ]:
            config_rows.append({"seed": seed, "item": item, "value": config.get(item, ""), "note": ""})
        ckpt_root = run_dir.parent / "checkpoints"
        dev_rows = _read_csv_if_exists(tables / "epoch_metrics_dev.csv")
        for row in dev_rows:
            epoch = _int(row.get("epoch"))
            ckpt = ckpt_root / f"epoch_{epoch:02d}"
            files = list(ckpt.rglob("*")) if ckpt.exists() else []
            size_mb = sum(path.stat().st_size for path in files if path.is_file()) / (1024 * 1024) if files else 0.0
            inventory_rows.append(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "global_step": row.get("global_step", ""),
                    "checkpoint_path_local": relpath(ckpt),
                    "checkpoint_exists": ckpt.exists(),
                    "checkpoint_size_mb": size_mb,
                    "git_tracked": False,
                    "note": "Local ignored checkpoint directory under thesis_exp/runs.",
                }
            )
    return {
        "dev_epoch": _sort_rows(dev_epoch),
        "test_epoch": _sort_rows(test_epoch),
        "dev_soft": _sort_rows(dev_soft),
        "test_soft": _sort_rows(test_soft),
        "dev_low": _sort_rows(dev_low),
        "test_low": _sort_rows(test_low),
        "config": sorted(config_rows, key=lambda row: (_int(row.get("seed")), str(row.get("item")))),
        "inventory": sorted(inventory_rows, key=lambda row: (_int(row.get("seed")), _int(row.get("epoch")))),
    }


def _best_by(rows: list[dict[str, Any]], key: str, mode: str, tie_key: str = "MAE") -> dict[str, Any]:
    reverse = mode == "max"
    return sorted(rows, key=lambda row: (_float(row.get(key)) * (-1 if reverse else 1), _float(row.get(tie_key))))[0]


def select_epoch(rule: str, dev_rows: list[dict[str, Any]], dev_soft_rows: list[dict[str, Any]], delta: float, beta: float) -> tuple[dict[str, Any], float, str]:
    if not dev_rows:
        return {}, float("nan"), ""
    soft_by_epoch = {_int(row.get("epoch")): row for row in dev_soft_rows}
    merged = []
    for row in dev_rows:
        soft = soft_by_epoch.get(_int(row.get("epoch")), {})
        merged.append(
            {
                **row,
                "soft_low_to_high": soft.get("soft_low_to_high", ""),
                "p_gt_3_low_mean": soft.get("p_gt_3_low_mean", ""),
                "label2_soft_low_to_high": soft.get("label2_soft_low_to_high", ""),
            }
        )
    min_mae = min(_float(row.get("MAE")) for row in merged)
    eligible = [row for row in merged if _float(row.get("MAE")) <= min_mae + delta]
    if rule == "dev_mae_min":
        row = _best_by(merged, "MAE", "min")
        return row, _float(row.get("MAE")), "none"
    if rule == "dev_qwk_max":
        row = _best_by(merged, "QWK", "max")
        return row, _float(row.get("QWK")), "dev_MAE"
    if rule == "dev_low_to_high_min_diagnostic":
        row = sorted(merged, key=lambda item: (_float(item.get("low_to_high")), _float(item.get("MAE"))))[0]
        return row, _float(row.get("low_to_high")), "dev_MAE"
    if rule == "mae_guard_soft_risk":
        row = sorted(eligible, key=lambda item: (_float(item.get("soft_low_to_high")), _float(item.get("MAE"))))[0]
        return row, _float(row.get("soft_low_to_high")), "dev_MAE"
    if rule == "mae_guard_label2_soft_risk":
        row = sorted(eligible, key=lambda item: (_float(item.get("label2_soft_low_to_high")), _float(item.get("MAE"))))[0]
        return row, _float(row.get("label2_soft_low_to_high")), "dev_MAE"
    if rule == "mae_guard_p_gt_3_low_mean":
        row = sorted(eligible, key=lambda item: (_float(item.get("p_gt_3_low_mean")), _float(item.get("MAE"))))[0]
        return row, _float(row.get("p_gt_3_low_mean")), "dev_MAE"
    if rule == "mae_guard_soft_risk_mono":
        scored = []
        for item in eligible:
            score = _float(item.get("soft_low_to_high")) + beta * _float(item.get("monotonic_violation"))
            scored.append((score, _float(item.get("MAE")), item))
        score, _, row = sorted(scored, key=lambda item: (item[0], item[1]))[0]
        return row, score, "dev_MAE"
    if rule == "last_epoch_diagnostic":
        row = sorted(merged, key=lambda item: _int(item.get("epoch")))[-1]
        return row, _int(row.get("epoch")), "none"
    raise ValueError(f"Unsupported rule: {rule}")


def build_selection_tables(
    dev_epoch_rows: list[dict[str, Any]],
    test_epoch_rows: list[dict[str, Any]],
    dev_soft_rows: list[dict[str, Any]],
    test_soft_rows: list[dict[str, Any]],
    delta: float,
    beta: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    selection_rows: list[dict[str, Any]] = []
    selected_test_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    multiseed_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    seeds = sorted({_int(row.get("seed")) for row in dev_epoch_rows})
    test_by_seed_epoch = {(_int(row.get("seed")), _int(row.get("epoch"))): row for row in test_epoch_rows}
    test_soft_by_seed_epoch = {(_int(row.get("seed")), _int(row.get("epoch"))): row for row in test_soft_rows}
    selected_by_seed_rule: dict[tuple[int, str], dict[str, Any]] = {}
    for seed in seeds:
        seed_dev = [row for row in dev_epoch_rows if _int(row.get("seed")) == seed]
        seed_dev_soft = [row for row in dev_soft_rows if _int(row.get("seed")) == seed]
        baseline_selected: dict[str, Any] | None = None
        for rule in SELECTION_RULES:
            selected, score, tie_breaker = select_epoch(rule, seed_dev, seed_dev_soft, delta, beta)
            epoch = _int(selected.get("epoch"))
            soft = { _int(row.get("epoch")): row for row in seed_dev_soft }.get(epoch, {})
            selection_rows.append(
                {
                    "seed": seed,
                    "selection_rule": rule,
                    "selected_epoch": epoch,
                    "selected_global_step": selected.get("global_step", ""),
                    "rule_type": "diagnostic" if rule in DIAGNOSTIC_RULES else "candidate",
                    "uses_test_for_selection": False,
                    "dev_MAE": selected.get("MAE", ""),
                    "dev_QWK": selected.get("QWK", ""),
                    "dev_low_to_high": selected.get("low_to_high", ""),
                    "dev_low_to_high_count": selected.get("low_to_high_count", ""),
                    "dev_soft_low_to_high": soft.get("soft_low_to_high", ""),
                    "dev_label2_soft_low_to_high": soft.get("label2_soft_low_to_high", ""),
                    "dev_p_gt_3_low_mean": soft.get("p_gt_3_low_mean", ""),
                    "dev_monotonic_violation": selected.get("monotonic_violation", ""),
                    "selection_score": score,
                    "tie_breaker": tie_breaker,
                }
            )
            test = test_by_seed_epoch.get((seed, epoch), {})
            test_soft = test_soft_by_seed_epoch.get((seed, epoch), {})
            selected_test = {
                "seed": seed,
                "selection_rule": rule,
                "selected_epoch": epoch,
                "selected_global_step": selected.get("global_step", ""),
                "test_MAE": test.get("MAE", ""),
                "test_QWK": test.get("QWK", ""),
                "test_Accuracy": test.get("Accuracy", ""),
                "test_Acc@5": test.get("Acc@5", ""),
                "test_low_to_high": test.get("low_to_high", ""),
                "test_low_to_high_count": test.get("low_to_high_count", ""),
                "test_true_low_score_count": test.get("true_low_score_count", ""),
                "test_label1_low_to_high": test.get("label1_low_to_high", ""),
                "test_label1_low_to_high_count": test.get("label1_low_to_high_count", ""),
                "test_label2_low_to_high": test.get("label2_low_to_high", ""),
                "test_label2_low_to_high_count": test.get("label2_low_to_high_count", ""),
                "test_monotonic_violation": test.get("monotonic_violation", ""),
                "test_p1_lt_p2": test.get("p1_lt_p2", ""),
                "test_p2_lt_p3": test.get("p2_lt_p3", ""),
                "test_p3_lt_p4": test.get("p3_lt_p4", ""),
                "test_soft_low_to_high": test_soft.get("soft_low_to_high", ""),
                "test_label2_soft_low_to_high": test_soft.get("label2_soft_low_to_high", ""),
                "test_p_gt_3_low_mean": test_soft.get("p_gt_3_low_mean", ""),
            }
            selected_test_rows.append(selected_test)
            selected_by_seed_rule[(seed, rule)] = selected_test
            multiseed_rows.append(
                {
                    "seed": seed,
                    "selection_rule": rule,
                    "selected_epoch": epoch,
                    "test_MAE": selected_test["test_MAE"],
                    "test_QWK": selected_test["test_QWK"],
                    "test_Acc@5": selected_test["test_Acc@5"],
                    "test_low_to_high": selected_test["test_low_to_high"],
                    "test_low_to_high_count": selected_test["test_low_to_high_count"],
                    "test_monotonic_violation": selected_test["test_monotonic_violation"],
                }
            )
            if rule == "dev_mae_min":
                baseline_selected = selected_test
        baseline = baseline_selected or selected_by_seed_rule.get((seed, "dev_mae_min"), {})
        for rule in SELECTION_RULES:
            selected_test = selected_by_seed_rule.get((seed, rule), {})
            delta_l2h_count = _float(selected_test.get("test_low_to_high_count")) - _float(baseline.get("test_low_to_high_count"))
            delta_rows.append(
                {
                    "seed": seed,
                    "selection_rule": rule,
                    "baseline_rule": "dev_mae_min",
                    "selected_epoch": selected_test.get("selected_epoch", ""),
                    "baseline_selected_epoch": baseline.get("selected_epoch", ""),
                    "delta_test_MAE": _float(selected_test.get("test_MAE")) - _float(baseline.get("test_MAE")),
                    "delta_test_QWK": _float(selected_test.get("test_QWK")) - _float(baseline.get("test_QWK")),
                    "delta_test_Acc@5": _float(selected_test.get("test_Acc@5")) - _float(baseline.get("test_Acc@5")),
                    "delta_test_low_to_high": _float(selected_test.get("test_low_to_high"))
                    - _float(baseline.get("test_low_to_high")),
                    "delta_test_low_to_high_count": delta_l2h_count,
                    "delta_test_monotonic_violation": _float(selected_test.get("test_monotonic_violation"))
                    - _float(baseline.get("test_monotonic_violation")),
                    "interpretation": _interpret_delta(delta_l2h_count),
                }
            )
    for rule in SELECTION_RULES:
        rule_rows = [row for row in selected_test_rows if row["selection_rule"] == rule]
        aggregate_rows.append(
            {
                "selection_rule": rule,
                "num_seeds": len(rule_rows),
                "mean_test_MAE": _mean(row.get("test_MAE") for row in rule_rows),
                "std_test_MAE": _std(row.get("test_MAE") for row in rule_rows),
                "mean_test_QWK": _mean(row.get("test_QWK") for row in rule_rows),
                "std_test_QWK": _std(row.get("test_QWK") for row in rule_rows),
                "mean_test_low_to_high": _mean(row.get("test_low_to_high") for row in rule_rows),
                "std_test_low_to_high": _std(row.get("test_low_to_high") for row in rule_rows),
                "mean_test_low_to_high_count": _mean(row.get("test_low_to_high_count") for row in rule_rows),
                "mean_test_monotonic_violation": _mean(row.get("test_monotonic_violation") for row in rule_rows),
                "std_test_monotonic_violation": _std(row.get("test_monotonic_violation") for row in rule_rows),
            }
        )
    return selection_rows, selected_test_rows, delta_rows, multiseed_rows, aggregate_rows


def _interpret_delta(delta_count: float) -> str:
    if not math.isfinite(delta_count) or abs(delta_count) < 1e-12:
        return "No test low-to-high count change versus dev_mae_min under this seed."
    if delta_count < 0:
        return "Lower post-hoc test low-to-high count than dev_mae_min under this seed."
    return "Higher post-hoc test low-to-high count than dev_mae_min under this seed."


def _mean(values: Any) -> float:
    nums = [_float(value) for value in values]
    nums = [value for value in nums if math.isfinite(value)]
    return float(statistics.fmean(nums)) if nums else float("nan")


def _std(values: Any) -> float:
    nums = [_float(value) for value in values]
    nums = [value for value in nums if math.isfinite(value)]
    return float(statistics.stdev(nums)) if len(nums) > 1 else 0.0 if len(nums) == 1 else float("nan")


def build_report(
    dev_epoch_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    selected_test_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    completed_seed_count: int,
) -> str:
    if not selection_rows:
        return """# Exp11 Checkpoint Selection Sensitivity

Status: `NO_COMPLETED_RUNS`

No completed Exp11 seed run with per-epoch metrics was found under the local ignored runs directory.
Run `./thesis_exp/scripts/run_exp11_checkpoint_selection_sensitivity.sh` to generate formal local
run artifacts, then re-run the collector.

Test metrics are post-hoc diagnostic only and are not used for selection. The core risk metric is
low-to-high.
"""
    seed = _int(selection_rows[0].get("seed"))
    by_rule = {row["selection_rule"]: row for row in selection_rows if _int(row.get("seed")) == seed}
    test_by_rule = {row["selection_rule"]: row for row in selected_test_rows if _int(row.get("seed")) == seed}
    dev_l2h_values = sorted({_float(row.get("low_to_high")) for row in dev_epoch_rows if _int(row.get("seed")) == seed})
    soft_values = sorted(
        {
            _float(row.get("dev_soft_low_to_high"))
            for row in selection_rows
            if _int(row.get("seed")) == seed and math.isfinite(_float(row.get("dev_soft_low_to_high")))
        }
    )
    baseline = test_by_rule.get("dev_mae_min", {})
    best_delta = min(delta_rows, key=lambda row: _float(row.get("delta_test_low_to_high_count")))
    l2h_different = any(
        _float(row.get("test_low_to_high_count")) != _float(baseline.get("test_low_to_high_count"))
        for row in test_by_rule.values()
    )
    lines = [
        "# Exp11 Checkpoint Selection Sensitivity",
        "",
        f"Completed seeds: `{completed_seed_count}`",
        "",
        "This report uses dev-only checkpoint selection rules. Test metrics are post-hoc diagnostic only and are not used for selection rule tuning or training.",
        "",
        "## Selection Answers",
        "",
        "| question | selected epoch | dev MAE | dev low-to-high | test low-to-high | test MAE | test QWK |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    labels = [
        ("dev_mae_min", "dev_mae_min"),
        ("dev_qwk_max", "dev_qwk_max"),
        ("dev_low_to_high_min_diagnostic", "dev_low_to_high_min_diagnostic"),
        ("mae_guard_soft_risk", "mae_guard_soft_risk"),
        ("mae_guard_label2_soft_risk", "mae_guard_label2_soft_risk"),
        ("mae_guard_p_gt_3_low_mean", "mae_guard_p_gt_3_low_mean"),
        ("mae_guard_soft_risk_mono", "mae_guard_soft_risk_mono"),
        ("last_epoch_diagnostic", "last_epoch_diagnostic"),
    ]
    for rule, label in labels:
        sel = by_rule.get(rule, {})
        tst = test_by_rule.get(rule, {})
        lines.append(
            f"| {label} | {sel.get('selected_epoch', '')} | {_fmt(sel.get('dev_MAE'))} | "
            f"{_fmt(sel.get('dev_low_to_high'))} | {_fmt(tst.get('test_low_to_high'))} | "
            f"{_fmt(tst.get('test_MAE'))} | {_fmt(tst.get('test_QWK'))} |"
        )
    lines.extend(
        [
            "",
            "## Findings",
            "",
            f"- Different selection rules {'do' if l2h_different else 'do not'} change post-hoc test low-to-high under the first completed seed.",
            f"- Dev hard low-to-high unique values: `{[_fmt(value) for value in dev_l2h_values]}`. If this list has one value, hard count has little selection signal under this seed.",
            f"- Dev soft-risk values among selected rules: `{[_fmt(value) for value in soft_values]}`. Soft risk is useful when it varies across epochs inside the MAE guard.",
            f"- Best post-hoc delta versus dev_mae_min: rule `{best_delta.get('selection_rule')}` with delta test low-to-high count `{_fmt(best_delta.get('delta_test_low_to_high_count'), 0)}`.",
            "- If risk-aware selection does not reduce post-hoc test low-to-high, treat that as a negative result rather than evidence of improvement.",
            "- The low-score test subset is small, so count-level interpretation is necessary.",
            "",
            "## Selection Rules",
            "",
        ]
    )
    lines.extend(f"- `{rule}`" for rule in SELECTION_RULES)
    lines.extend(
        [
            "",
            "## Soft Risk Definition",
            "",
            "For true low-score samples (`y <= 2`), `soft_low_to_high = mean(sigmoid(gamma * (r_theta(x) - 3.5)))`, where `r_theta(x) = 1 + sum_t p_t(x)` and `gamma=4.0` by default.",
            "`p_gt_3_low_mean` is the low-score mean of `P(y > 3 | x)`, using the existing `prob_gt_3` prediction column.",
        ]
    )
    return "\n".join(lines)


def collect(runs_root: Path, delta: float, beta: float) -> dict[str, str]:
    ensure_exp11_dirs()
    rows = load_run_rows(runs_root)
    selection, selected_test, deltas, multiseed, aggregate = build_selection_tables(
        rows["dev_epoch"], rows["test_epoch"], rows["dev_soft"], rows["test_soft"], delta, beta
    )
    write_csv(EXP11_TABLES_DIR / "exp11_epoch_metrics_dev.csv", rows["dev_epoch"], fieldnames=EPOCH_FIELDS)
    write_csv(EXP11_TABLES_DIR / "exp11_epoch_metrics_test_diagnostic.csv", rows["test_epoch"], fieldnames=EPOCH_FIELDS)
    write_csv(EXP11_TABLES_DIR / "exp11_soft_risk_metrics_dev.csv", rows["dev_soft"], fieldnames=SOFT_FIELDS)
    write_csv(EXP11_TABLES_DIR / "exp11_soft_risk_metrics_test_diagnostic.csv", rows["test_soft"], fieldnames=SOFT_FIELDS)
    write_csv(EXP11_TABLES_DIR / "exp11_selection_rule_summary.csv", selection, fieldnames=SELECTION_FIELDS)
    write_csv(EXP11_TABLES_DIR / "exp11_selected_checkpoint_test_metrics.csv", selected_test, fieldnames=TEST_SELECTED_FIELDS)
    write_csv(EXP11_TABLES_DIR / "exp11_selection_rule_vs_dev_mae_baseline.csv", deltas, fieldnames=DELTA_FIELDS)
    write_csv(EXP11_TABLES_DIR / "exp11_run_config_summary.csv", rows["config"], fieldnames=CONFIG_FIELDS)
    write_csv(EXP11_TABLES_DIR / "exp11_checkpoint_inventory.csv", rows["inventory"], fieldnames=INVENTORY_FIELDS)
    write_csv(EXP11_TABLES_DIR / "exp11_low_score_by_epoch_dev.csv", rows["dev_low"])
    write_csv(EXP11_TABLES_DIR / "exp11_low_score_by_epoch_test_diagnostic.csv", rows["test_low"])
    write_csv(EXP11_TABLES_DIR / "exp11_multiseed_selection_summary.csv", multiseed, fieldnames=MULTISEED_FIELDS)
    write_csv(EXP11_TABLES_DIR / "exp11_multiseed_rule_aggregate.csv", aggregate, fieldnames=AGG_FIELDS)
    report = build_report(rows["dev_epoch"], selection, selected_test, deltas, len(_seed_dirs(runs_root)))
    write_text(EXP11_REPORTS_DIR / "exp11_checkpoint_selection_sensitivity_report.md", report)
    write_text(
        EXP11_REPORTS_DIR / "exp11_checkpoint_selection_sensitivity_review_package.md",
        report
        + "\n\n## Review Checklist\n\n"
        + "- Verify `uses_test_for_selection` is false for every selection rule.\n"
        + "- Verify test diagnostic tables are not used to tune selection rules.\n"
        + "- Verify no checkpoint, raw prediction, `.npy`, or `.npz` artifact is written under tracked Exp11 outputs.\n",
    )
    return {
        "report": relpath(EXP11_REPORTS_DIR / "exp11_checkpoint_selection_sensitivity_report.md"),
        "selection_summary": relpath(EXP11_TABLES_DIR / "exp11_selection_rule_summary.csv"),
        "selected_test": relpath(EXP11_TABLES_DIR / "exp11_selected_checkpoint_test_metrics.csv"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Exp11 checkpoint selection sensitivity results.")
    parser.add_argument("--runs_root", type=Path, default=EXP11_LOCAL_RUNS_DIR)
    parser.add_argument("--delta", type=float, default=DEFAULT_MAE_GUARD_DELTA)
    parser.add_argument("--beta", type=float, default=DEFAULT_MONO_BETA)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(collect(args.runs_root, args.delta, args.beta), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
