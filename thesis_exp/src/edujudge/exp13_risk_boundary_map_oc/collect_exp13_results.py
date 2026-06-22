"""Collect Exp13 risk-boundary MAP-OC results."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import yaml

from thesis_exp.src.edujudge.exp13_risk_boundary_map_oc import (
    CONFIG_BY_RUN,
    DEFAULT_SELECTION_DELTA,
    DEFAULT_SELECTION_RULE,
    EXP13_LOCAL_RUNS_DIR,
    EXP13_REPORTS_DIR,
    EXP13_RUNS,
    EXP13_TABLES_DIR,
    ensure_exp13_dirs,
)
from thesis_exp.src.edujudge.utils.io import read_csv, relpath, write_csv, write_text


DEV_RANK_FIELDS = [
    "mode",
    "run_name",
    "seed",
    "epoch",
    "global_step",
    "decode_mode",
    "selected",
    "eligible",
    "selection_rule",
    "selection_delta",
    "uses_test_for_selection",
    "dev_MAE",
    "dev_QWK",
    "dev_Accuracy",
    "dev_Acc@5",
    "dev_low_to_high",
    "dev_low_to_high_count",
    "dev_true_low_score_count",
    "dev_label1_low_to_high",
    "dev_label1_low_to_high_count",
    "dev_label2_low_to_high",
    "dev_label2_low_to_high_count",
    "dev_monotonic_violation",
    "dev_p1_lt_p2",
    "dev_p2_lt_p3",
    "dev_p3_lt_p4",
    "dev_p_gt_3_low_mean",
    "dev_p_gt_3_label1_mean",
    "dev_p_gt_3_label2_mean",
    "dev_p_gt_3_low_q90",
    "dev_p_gt_3_low_q95",
    "dev_low_count_with_p_gt_3_over_0p5",
    "dev_high_to_low",
    "dev_high_to_low_count",
    "dev_label4_recall",
    "dev_label5_recall",
]

FORMAL_FIELDS = [
    "mode",
    "run_name",
    "seed",
    "selected_epoch",
    "selected_global_step",
    "decode_mode",
    "selection_rule",
    "selection_delta",
    "uses_test_for_selection",
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
    "test_p_gt_3_low_mean",
    "test_p_gt_3_label1_mean",
    "test_p_gt_3_label2_mean",
    "test_p_gt_3_low_q90",
    "test_p_gt_3_low_q95",
    "test_low_count_with_p_gt_3_over_0p5",
    "test_high_to_low",
    "test_high_to_low_count",
    "test_label4_recall",
    "test_label5_recall",
]


def _float(value: Any, default: float = math.nan) -> float:
    try:
        text = str(value)
        if text == "":
            return default
        return float(text)
    except Exception:
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        text = str(value)
        if text == "":
            return default
        return int(float(text))
    except Exception:
        return default


def _run_dirs(mode: str, run_names: list[str]) -> list[tuple[str, int, Path]]:
    items: list[tuple[str, int, Path]] = []
    for run_name in run_names:
        root = EXP13_LOCAL_RUNS_DIR / mode / run_name
        for seed_dir in sorted(root.glob("seed_*")):
            seed = _int(seed_dir.name.replace("seed_", ""))
            run_dir = seed_dir / "run"
            if run_dir.exists():
                items.append((run_name, seed, run_dir))
    return items


def _merge_metric_soft(metric_rows: list[dict[str, str]], soft_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    soft_by_key = {
        (row.get("seed"), row.get("epoch"), row.get("global_step"), row.get("split"), row.get("decode_mode")): row
        for row in soft_rows
    }
    out = []
    for row in metric_rows:
        key = (row.get("seed"), row.get("epoch"), row.get("global_step"), row.get("split"), row.get("decode_mode"))
        merged = dict(row)
        merged.update({f"soft_{key}": value for key, value in soft_by_key.get(key, {}).items()})
        for name in [
            "p_gt_3_low_mean",
            "p_gt_3_label1_mean",
            "p_gt_3_label2_mean",
            "p_gt_3_low_q90",
            "p_gt_3_low_q95",
            "low_count_with_p_gt_3_over_0p5",
        ]:
            merged[name] = soft_by_key.get(key, {}).get(name, merged.get(name, ""))
        out.append(merged)
    return out


def _read_run_rows(mode: str, run_name: str, seed: int, run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    tables = run_dir / "tables"
    dev_metrics = read_csv(tables / "epoch_metrics_dev.csv") if (tables / "epoch_metrics_dev.csv").exists() else []
    dev_soft = read_csv(tables / "soft_risk_metrics_dev.csv") if (tables / "soft_risk_metrics_dev.csv").exists() else []
    test_metrics = (
        read_csv(tables / "epoch_metrics_test_diagnostic.csv") if (tables / "epoch_metrics_test_diagnostic.csv").exists() else []
    )
    test_soft = (
        read_csv(tables / "soft_risk_metrics_test_diagnostic.csv")
        if (tables / "soft_risk_metrics_test_diagnostic.csv").exists()
        else []
    )
    pred_dev = read_csv(tables / "prediction_distribution_dev.csv") if (tables / "prediction_distribution_dev.csv").exists() else []
    pred_test = (
        read_csv(tables / "prediction_distribution_test_diagnostic.csv")
        if (tables / "prediction_distribution_test_diagnostic.csv").exists()
        else []
    )
    out = {
        "dev": _merge_metric_soft(dev_metrics, dev_soft),
        "test": _merge_metric_soft(test_metrics, test_soft),
        "pred_dev": pred_dev,
        "pred_test": pred_test,
    }
    for rows in out.values():
        for row in rows:
            row["mode"] = mode
            row["run_name"] = run_name
            row["seed"] = seed
            row["source_run_dir"] = relpath(run_dir)
            row["selection_rule"] = DEFAULT_SELECTION_RULE
            row["uses_test_for_selection"] = False
    return out


def _select_row(rows: list[dict[str, Any]], delta: float) -> dict[str, Any] | None:
    projected = [row for row in rows if row.get("decode_mode") == "projected"]
    pool = projected or rows
    if not pool:
        return None
    best_mae = min(_float(row.get("MAE")) for row in pool)
    candidates = [row for row in pool if _float(row.get("MAE")) <= best_mae + delta]
    return sorted(
        candidates,
        key=lambda row: (
            _float(row.get("p_gt_3_low_mean")),
            _float(row.get("p_gt_3_label2_mean")),
            _float(row.get("MAE")),
            _int(row.get("epoch")),
        ),
    )[0]


def _dev_rank_row(row: dict[str, Any], selected: bool, eligible: bool, delta: float) -> dict[str, Any]:
    return {
        "mode": row.get("mode"),
        "run_name": row.get("run_name"),
        "seed": row.get("seed"),
        "epoch": row.get("epoch"),
        "global_step": row.get("global_step"),
        "decode_mode": row.get("decode_mode"),
        "selected": selected,
        "eligible": eligible,
        "selection_rule": DEFAULT_SELECTION_RULE,
        "selection_delta": delta,
        "uses_test_for_selection": False,
        "dev_MAE": row.get("MAE"),
        "dev_QWK": row.get("QWK"),
        "dev_Accuracy": row.get("Accuracy"),
        "dev_Acc@5": row.get("Acc@5"),
        "dev_low_to_high": row.get("low_to_high"),
        "dev_low_to_high_count": row.get("low_to_high_count"),
        "dev_true_low_score_count": row.get("true_low_score_count"),
        "dev_label1_low_to_high": row.get("label1_low_to_high"),
        "dev_label1_low_to_high_count": row.get("label1_low_to_high_count"),
        "dev_label2_low_to_high": row.get("label2_low_to_high"),
        "dev_label2_low_to_high_count": row.get("label2_low_to_high_count"),
        "dev_monotonic_violation": row.get("monotonic_violation"),
        "dev_p1_lt_p2": row.get("p1_lt_p2"),
        "dev_p2_lt_p3": row.get("p2_lt_p3"),
        "dev_p3_lt_p4": row.get("p3_lt_p4"),
        "dev_p_gt_3_low_mean": row.get("p_gt_3_low_mean"),
        "dev_p_gt_3_label1_mean": row.get("p_gt_3_label1_mean"),
        "dev_p_gt_3_label2_mean": row.get("p_gt_3_label2_mean"),
        "dev_p_gt_3_low_q90": row.get("p_gt_3_low_q90"),
        "dev_p_gt_3_low_q95": row.get("p_gt_3_low_q95"),
        "dev_low_count_with_p_gt_3_over_0p5": row.get("low_count_with_p_gt_3_over_0p5"),
        "dev_high_to_low": row.get("high_to_low"),
        "dev_high_to_low_count": row.get("high_to_low_count"),
        "dev_label4_recall": row.get("label4_recall"),
        "dev_label5_recall": row.get("label5_recall"),
    }


def _formal_row(selected: dict[str, Any], test_row: dict[str, Any], delta: float) -> dict[str, Any]:
    return {
        "mode": selected.get("mode"),
        "run_name": selected.get("run_name"),
        "seed": selected.get("seed"),
        "selected_epoch": selected.get("epoch"),
        "selected_global_step": selected.get("global_step"),
        "decode_mode": test_row.get("decode_mode"),
        "selection_rule": DEFAULT_SELECTION_RULE,
        "selection_delta": delta,
        "uses_test_for_selection": False,
        "test_MAE": test_row.get("MAE"),
        "test_QWK": test_row.get("QWK"),
        "test_Accuracy": test_row.get("Accuracy"),
        "test_Acc@5": test_row.get("Acc@5"),
        "test_low_to_high": test_row.get("low_to_high"),
        "test_low_to_high_count": test_row.get("low_to_high_count"),
        "test_true_low_score_count": test_row.get("true_low_score_count"),
        "test_label1_low_to_high": test_row.get("label1_low_to_high"),
        "test_label1_low_to_high_count": test_row.get("label1_low_to_high_count"),
        "test_label2_low_to_high": test_row.get("label2_low_to_high"),
        "test_label2_low_to_high_count": test_row.get("label2_low_to_high_count"),
        "test_monotonic_violation": test_row.get("monotonic_violation"),
        "test_p1_lt_p2": test_row.get("p1_lt_p2"),
        "test_p2_lt_p3": test_row.get("p2_lt_p3"),
        "test_p3_lt_p4": test_row.get("p3_lt_p4"),
        "test_p_gt_3_low_mean": test_row.get("p_gt_3_low_mean"),
        "test_p_gt_3_label1_mean": test_row.get("p_gt_3_label1_mean"),
        "test_p_gt_3_label2_mean": test_row.get("p_gt_3_label2_mean"),
        "test_p_gt_3_low_q90": test_row.get("p_gt_3_low_q90"),
        "test_p_gt_3_low_q95": test_row.get("p_gt_3_low_q95"),
        "test_low_count_with_p_gt_3_over_0p5": test_row.get("low_count_with_p_gt_3_over_0p5"),
        "test_high_to_low": test_row.get("high_to_low"),
        "test_high_to_low_count": test_row.get("high_to_low_count"),
        "test_label4_recall": test_row.get("label4_recall"),
        "test_label5_recall": test_row.get("label5_recall"),
    }


def _config_summary(run_names: list[str]) -> list[dict[str, Any]]:
    rows = []
    keys = [
        "base_run",
        "lambda_l2h_risk",
        "risk_weight_label1",
        "risk_weight_label2",
        "lambda_mono",
        "use_l2h_risk_loss",
        "use_t3_calibration_loss",
        "lambda_t3_calibration",
        "projection_in_point_loss",
        "projection_in_pair_score",
        "projection_in_anchor",
        "selection_rule",
    ]
    for run_name in run_names:
        path = CONFIG_BY_RUN.get(run_name)
        config = yaml.safe_load(path.read_text(encoding="utf-8")) if path and path.exists() else {}
        row = {"run_name": run_name, "config_path": relpath(path) if path else ""}
        row.update({key: config.get(key, "") for key in keys})
        rows.append(row)
    return rows


def collect(mode: str, run_names: list[str], delta: float) -> dict[str, Any]:
    ensure_exp13_dirs()
    all_dev: list[dict[str, Any]] = []
    all_test: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    formal_rows: list[dict[str, Any]] = []

    for run_name, seed, run_dir in _run_dirs(mode, run_names):
        rows = _read_run_rows(mode, run_name, seed, run_dir)
        all_dev.extend(rows["dev"])
        all_test.extend(rows["test"])
        pred_rows.extend(rows["pred_dev"])
        pred_rows.extend(rows["pred_test"])
        selected = _select_row(rows["dev"], delta)
        if selected is None:
            continue
        selected_rows.append(selected)
        best_mae = min(_float(row.get("MAE")) for row in rows["dev"] if row.get("decode_mode") == selected.get("decode_mode"))
        for row in rows["dev"]:
            if row.get("decode_mode") != selected.get("decode_mode"):
                continue
            eligible = _float(row.get("MAE")) <= best_mae + delta
            all_key = (row.get("run_name"), row.get("seed"), row.get("epoch"), row.get("decode_mode"))
            selected_key = (selected.get("run_name"), selected.get("seed"), selected.get("epoch"), selected.get("decode_mode"))
            row["_rank_row"] = _dev_rank_row(row, all_key == selected_key, eligible, delta)
        for test_row in rows["test"]:
            if _int(test_row.get("epoch")) == _int(selected.get("epoch")):
                formal_rows.append(_formal_row(selected, test_row, delta))

    rank_rows = [row["_rank_row"] for row in all_dev if "_rank_row" in row]
    rank_rows = sorted(
        rank_rows,
        key=lambda row: (
            row.get("run_name", ""),
            _int(row.get("seed")),
            row.get("decode_mode", ""),
            _int(row.get("epoch")),
        ),
    )
    selected_rank_rows = [row for row in rank_rows if str(row.get("selected")).lower() == "true"]
    write_csv(EXP13_TABLES_DIR / "exp13_dev_selection_ranking.csv", rank_rows, fieldnames=DEV_RANK_FIELDS)
    write_csv(EXP13_TABLES_DIR / "exp13_dev_selected_configs.csv", selected_rank_rows, fieldnames=DEV_RANK_FIELDS)
    write_csv(EXP13_TABLES_DIR / "exp13_formal_summary.csv", formal_rows, fieldnames=FORMAL_FIELDS)
    write_csv(EXP13_TABLES_DIR / "exp13_run_config_summary.csv", _config_summary(run_names))
    if pred_rows:
        write_csv(EXP13_TABLES_DIR / "exp13_prediction_distribution.csv", pred_rows)

    status = "COMPLETED" if selected_rank_rows else "NO_COMPLETED_RUNS"
    lines = [
        "# Exp13 Risk-Boundary MAP-OC",
        "",
        f"Mode: `{mode}`",
        f"Status: `{status}`",
        "",
        "Scout ranking is dev-only. Test metrics are not used for checkpoint selection, config selection, or tuning.",
        "",
        "## Dev-Only Selected Rows",
        "",
    ]
    if selected_rank_rows:
        lines.extend(["| run | seed | epoch | dev MAE | dev low-to-high | dev p_gt_3_low_mean |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for row in sorted(selected_rank_rows, key=lambda item: _float(item.get("dev_p_gt_3_low_mean"))):
            lines.append(
                f"| {row.get('run_name')} | {row.get('seed')} | {row.get('epoch')} | "
                f"{_float(row.get('dev_MAE')):.4f} | {_float(row.get('dev_low_to_high')):.4f} | "
                f"{_float(row.get('dev_p_gt_3_low_mean')):.4f} |"
            )
    else:
        lines.append("No completed Exp13 training rows were found under the local ignored runs directory.")
    lines.extend(
        [
            "",
            "## Method Notes",
            "",
            "- Exp13 adds a squared hinge risk-boundary loss on projected `q3 = P(y > 3)` for low-score samples.",
            "- The base setting is Exp12B `train_projection_point_pair` unless a config explicitly switches base.",
            "- PAVA projection preserves ordinal threshold positions; it is not sorting.",
            "- This is a next-stage experiment design for low-to-high risk calibration, not evidence that the problem is solved.",
        ]
    )
    write_text(EXP13_REPORTS_DIR / "exp13_risk_boundary_map_oc_report.md", "\n".join(lines))
    write_text(
        EXP13_REPORTS_DIR / "exp13_review_package.md",
        "\n".join(
            [
                "# Exp13 Review Package",
                "",
                "- Check `exp13_dev_selection_ranking.csv`; it must contain dev columns only.",
                "- Verify `uses_test_for_selection` is false.",
                "- Review risk loss settings in `exp13_run_config_summary.csv`.",
                "- In scout mode, do not use test metrics for config ranking.",
            ]
        ),
    )
    return {
        "status": status,
        "dev_rows": len(all_dev),
        "test_rows": len(all_test),
        "selected_rows": len(selected_rank_rows),
        "formal_rows": len(formal_rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Exp13 results.")
    parser.add_argument("--mode", default="scout")
    parser.add_argument("--runs", default=" ".join(EXP13_RUNS))
    parser.add_argument("--delta", type=float, default=DEFAULT_SELECTION_DELTA)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = collect(args.mode, args.runs.split(), args.delta)
    print(summary)


if __name__ == "__main__":
    main()
