"""Collect Exp14 logit-margin tail-risk scout results."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import yaml

from thesis_exp.src.edujudge.exp14_logit_margin_tail_risk_oc import (
    CONFIG_BY_RUN,
    DEFAULT_SELECTION_DELTA,
    DEFAULT_SELECTION_RULE,
    EXP14_LOCAL_RUNS_DIR,
    EXP14_REPORTS_DIR,
    EXP14_RUNS,
    EXP14_TABLES_DIR,
    ensure_exp14_dirs,
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
    "formal_candidate",
    "formal_candidate_reason",
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
    "dev_low_z3_mean",
    "dev_low_z3_q90",
    "dev_low_z3_q95",
    "dev_label1_z3_mean",
    "dev_label2_z3_mean",
    "dev_high_to_low",
    "dev_high_to_low_count",
    "dev_label4_recall",
    "dev_label5_recall",
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
        root = EXP14_LOCAL_RUNS_DIR / mode / run_name
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
        for name in [
            "p_gt_3_low_mean",
            "p_gt_3_label1_mean",
            "p_gt_3_label2_mean",
            "p_gt_3_low_q90",
            "p_gt_3_low_q95",
            "low_count_with_p_gt_3_over_0p5",
            "low_z3_mean",
            "low_z3_q90",
            "low_z3_q95",
            "label1_z3_mean",
            "label2_z3_mean",
        ]:
            merged[name] = soft_by_key.get(key, {}).get(name, merged.get(name, ""))
        out.append(merged)
    return out


def _read_run_rows(mode: str, run_name: str, seed: int, run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    tables = run_dir / "tables"
    dev_metrics = read_csv(tables / "epoch_metrics_dev.csv") if (tables / "epoch_metrics_dev.csv").exists() else []
    dev_soft = read_csv(tables / "soft_risk_metrics_dev.csv") if (tables / "soft_risk_metrics_dev.csv").exists() else []
    pred_dev = read_csv(tables / "prediction_distribution_dev.csv") if (tables / "prediction_distribution_dev.csv").exists() else []
    out = {"dev": _merge_metric_soft(dev_metrics, dev_soft), "pred_dev": pred_dev}
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
            _int(row.get("low_to_high_count")),
            _float(row.get("p_gt_3_low_mean")),
            _float(row.get("p_gt_3_label2_mean")),
            _float(row.get("MAE")),
            _int(row.get("epoch")),
        ),
    )[0]


def _formal_candidate(row: dict[str, Any]) -> tuple[bool, str]:
    checks = [
        ("low_to_high_count<=26", _int(row.get("low_to_high_count")) <= 26),
        ("MAE<=0.493", _float(row.get("MAE")) <= 0.493),
        ("high_to_low_count<=2", _int(row.get("high_to_low_count")) <= 2),
        ("label4_recall>=0.50", _float(row.get("label4_recall"), 0.0) >= 0.50),
        ("label5_recall>=0.70", _float(row.get("label5_recall"), 0.0) >= 0.70),
    ]
    return all(ok for _, ok in checks), "; ".join(f"{name}:{'PASS' if ok else 'FAIL'}" for name, ok in checks)


def _dev_rank_row(row: dict[str, Any], selected: bool, eligible: bool, delta: float) -> dict[str, Any]:
    formal_candidate, reason = _formal_candidate(row)
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
        "formal_candidate": formal_candidate,
        "formal_candidate_reason": reason,
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
        "dev_low_z3_mean": row.get("low_z3_mean"),
        "dev_low_z3_q90": row.get("low_z3_q90"),
        "dev_low_z3_q95": row.get("low_z3_q95"),
        "dev_label1_z3_mean": row.get("label1_z3_mean"),
        "dev_label2_z3_mean": row.get("label2_z3_mean"),
        "dev_high_to_low": row.get("high_to_low"),
        "dev_high_to_low_count": row.get("high_to_low_count"),
        "dev_label4_recall": row.get("label4_recall"),
        "dev_label5_recall": row.get("label5_recall"),
    }


def _config_summary(run_names: list[str]) -> list[dict[str, Any]]:
    rows = []
    keys = [
        "base_run",
        "lambda_l2h_logit_margin",
        "logit_margin_tail_fraction",
        "logit_margin_weight_label1",
        "logit_margin_weight_label2",
        "margin_prob_label1",
        "margin_prob_label2",
        "use_l2h_risk_loss",
        "use_l2h_logit_margin_loss",
        "projection_in_pair_score",
        "projection_in_point_loss",
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
    ensure_exp14_dirs()
    all_dev: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []

    for run_name, seed, run_dir in _run_dirs(mode, run_names):
        rows = _read_run_rows(mode, run_name, seed, run_dir)
        all_dev.extend(rows["dev"])
        pred_rows.extend(rows["pred_dev"])
        selected = _select_row(rows["dev"], delta)
        if selected is None:
            continue
        projected_rows = [row for row in rows["dev"] if row.get("decode_mode") == selected.get("decode_mode")]
        best_mae = min(_float(row.get("MAE")) for row in projected_rows) if projected_rows else math.nan
        for row in projected_rows:
            eligible = _float(row.get("MAE")) <= best_mae + delta
            all_key = (row.get("run_name"), row.get("seed"), row.get("epoch"), row.get("decode_mode"))
            selected_key = (selected.get("run_name"), selected.get("seed"), selected.get("epoch"), selected.get("decode_mode"))
            row["_rank_row"] = _dev_rank_row(row, all_key == selected_key, eligible, delta)

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
    selected_rows = [row for row in rank_rows if str(row.get("selected")).lower() == "true"]
    write_csv(EXP14_TABLES_DIR / "exp14_dev_selection_ranking.csv", rank_rows, fieldnames=DEV_RANK_FIELDS)
    write_csv(EXP14_TABLES_DIR / "exp14_dev_selected_configs.csv", selected_rows, fieldnames=DEV_RANK_FIELDS)
    write_csv(EXP14_TABLES_DIR / "exp14_run_config_summary.csv", _config_summary(run_names))
    if pred_rows:
        write_csv(EXP14_TABLES_DIR / "exp14_prediction_distribution.csv", pred_rows)

    status = "COMPLETED" if selected_rows else "NO_COMPLETED_RUNS"
    formal_candidates = [row for row in selected_rows if str(row.get("formal_candidate")).lower() == "true"]
    recommendation = "FORMAL_CANDIDATE_FOUND" if formal_candidates else "NO_FORMAL_RECOMMENDED"
    lines = [
        "# Exp14 Logit-Margin Tail-Risk OC",
        "",
        f"Mode: `{mode}`",
        f"Status: `{status}`",
        f"Formal recommendation: `{recommendation}`",
        "",
        "Scout ranking is dev-only. Test metrics are not used for checkpoint selection, config selection, or tuning.",
        "",
        "Exp13 formal dev baseline: dev_low_to_high_count = 28/57; best dev_MAE around 0.4827.",
        "",
        "A config is recommended for formal only if dev_low_to_high_count <= 26/57, dev_MAE <= 0.493, high-to-low does not clearly increase, and label4/label5 recall do not collapse.",
        "",
        "## Dev-Only Selected Rows",
        "",
    ]
    if selected_rows:
        lines.extend(
            [
                "| run | epoch | dev MAE | dev low-to-high count | dev p_gt_3_low_mean | dev low_z3_q95 | formal candidate |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in sorted(selected_rows, key=lambda item: (_int(item.get("dev_low_to_high_count")), _float(item.get("dev_MAE")))):
            lines.append(
                f"| {row.get('run_name')} | {row.get('epoch')} | {_float(row.get('dev_MAE')):.4f} | "
                f"{row.get('dev_low_to_high_count')}/{row.get('dev_true_low_score_count')} | "
                f"{_float(row.get('dev_p_gt_3_low_mean')):.4f} | {_float(row.get('dev_low_z3_q95')):.4f} | "
                f"{row.get('formal_candidate')} |"
            )
    else:
        lines.append("No completed Exp14 scout rows were found under the local ignored runs directory.")
    lines.extend(
        [
            "",
            "## Method Notes",
            "",
            "- Exp14 adds a squared hinge tail-risk loss directly on raw threshold-3 logit `z3` for `y <= 2` samples.",
            "- Decode still uses projected probabilities; the loss does not use hard decoded labels.",
            "- PAVA projection is reused for decoding and diagnostics; this experiment does not replace it with sorting.",
            "- This is a dev-only scout for the saturated threshold-3 boundary exposed by Exp13, not evidence that the problem is solved.",
        ]
    )
    write_text(EXP14_REPORTS_DIR / "exp14_logit_margin_tail_risk_oc_report.md", "\n".join(lines))
    return {"status": status, "dev_rows": len(all_dev), "selected_rows": len(selected_rows), "formal_candidates": len(formal_candidates)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Exp14 results.")
    parser.add_argument("--mode", default="scout")
    parser.add_argument("--runs", default=" ".join(EXP14_RUNS))
    parser.add_argument("--delta", type=float, default=DEFAULT_SELECTION_DELTA)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = collect(args.mode, args.runs.split(), args.delta)
    print(summary)


if __name__ == "__main__":
    main()
