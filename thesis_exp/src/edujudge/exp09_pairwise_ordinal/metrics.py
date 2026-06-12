"""Metrics and pairwise diagnostics for Exp9 pairwise ordinal training."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from thesis_exp.src.edujudge.exp02.train_ce_baseline import (
    high_score_metrics,
    low_score_metrics,
    per_bin_metrics,
)
from thesis_exp.src.edujudge.exp02.train_ce_baseline import compute_metrics as base_compute_metrics
from thesis_exp.src.edujudge.exp04.train_objective import add_extra_metrics, grouped_metrics
from thesis_exp.src.edujudge.exp07_rank_consistent.metrics import monotonicity_metrics
from thesis_exp.src.edujudge.utils.io import write_csv


def compute_ordinal_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = base_compute_metrics(predictions)
    return add_extra_metrics(metrics, predictions, "ordinal")


def prediction_distribution_rows(predictions: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    labels = [int(row["label_5"]) for row in predictions]
    preds = [int(row["pred_label_5"]) for row in predictions]
    total = len(preds)
    rows: list[dict[str, Any]] = []
    for label in [1, 2, 3, 4, 5]:
        true_count = sum(1 for value in labels if value == label)
        pred_count = sum(1 for value in preds if value == label)
        rows.append(
            {
                "split": split,
                "label_5": label,
                "true_count": true_count,
                "true_rate": true_count / total if total else float("nan"),
                "pred_count": pred_count,
                "pred_rate": pred_count / total if total else float("nan"),
            }
        )
    return rows


def save_metric_tables(output_dir: Any, results: list[Any]) -> None:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = [result.metrics for result in results]
    per_bin_rows: list[dict[str, Any]] = []
    low_rows: list[dict[str, Any]] = []
    high_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    language_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    monotonic_rows: list[dict[str, Any]] = []
    for result in results:
        split = result.metrics["split"]
        per_bin_rows.extend(per_bin_metrics(result.predictions, split))
        low_rows.append(low_score_metrics(result.metrics, split))
        high_rows.append(high_score_metrics(result.metrics, split))
        metric_rows.extend(grouped_metrics(result.predictions, split, "metric_canonical", "ordinal"))
        scenario_rows.extend(grouped_metrics(result.predictions, split, "scenario_canonical", "ordinal"))
        language_rows.extend(grouped_metrics(result.predictions, split, "language", "ordinal"))
        distribution_rows.extend(prediction_distribution_rows(result.predictions, split))
        monotonic_rows.append(monotonicity_metrics(result.probs, split))
    write_csv(tables_dir / "metrics_summary.csv", summary_rows)
    write_csv(tables_dir / "per_bin_metrics.csv", per_bin_rows)
    write_csv(tables_dir / "low_score_metrics.csv", low_rows)
    write_csv(tables_dir / "high_score_metrics.csv", high_rows)
    write_csv(tables_dir / "metric_level_metrics.csv", metric_rows)
    write_csv(tables_dir / "scenario_level_metrics.csv", scenario_rows)
    write_csv(tables_dir / "language_level_metrics.csv", language_rows)
    write_csv(tables_dir / "score_distribution.csv", distribution_rows)
    write_csv(tables_dir / "monotonicity_metrics.csv", monotonic_rows)


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _safe_rate(count: int, total: int) -> float:
    return float(count / total) if total else float("nan")


def _score_by_record(predictions: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in predictions:
        record_id = str(row.get("record_id") or row.get("id"))
        out[record_id] = float(row.get("pred_score_expected", row.get("pred_score_5", row.get("pred_label_5"))))
    return out


def _label_by_record(predictions: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in predictions:
        record_id = str(row.get("record_id") or row.get("id"))
        out[record_id] = int(row["label_5"])
    return out


def pairwise_diagnostic_rows(
    pairs: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    split: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    score_by_id = _score_by_record(predictions)
    label_by_id = _label_by_record(predictions)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    used_pairs: list[dict[str, Any]] = []
    for pair in pairs:
        win_id = str(pair["win_record_id"])
        lose_id = str(pair["lose_record_id"])
        if win_id not in score_by_id or lose_id not in score_by_id:
            continue
        score_gap = score_by_id[win_id] - score_by_id[lose_id]
        item = {
            **pair,
            "score_gap": score_gap,
            "win_score": score_by_id[win_id],
            "lose_score": score_by_id[lose_id],
            "win_label_eval": label_by_id.get(win_id, ""),
            "lose_label_eval": label_by_id.get(lose_id, ""),
            "pair_correct": score_gap > 0,
            "margin_satisfied": score_gap >= float(pair["pair_margin"]),
        }
        used_pairs.append(item)
        groups[("overall", "overall")].append(item)
        groups[("pair_type", str(pair["pair_type"]))].append(item)
        groups[("priority", str(pair.get("priority", "")))].append(item)
    accuracy_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    for (group_field, group_value), items in sorted(groups.items()):
        total = len(items)
        correct = sum(1 for item in items if bool(item["pair_correct"]))
        margin_ok = sum(1 for item in items if bool(item["margin_satisfied"]))
        score_gaps = [float(item["score_gap"]) for item in items]
        margins = [float(item["pair_margin"]) for item in items]
        accuracy_rows.append(
            {
                "split": split,
                "group_field": group_field,
                "group_value": group_value,
                "pair_count": total,
                "pair_accuracy": _safe_rate(correct, total),
                "margin_satisfied_rate": _safe_rate(margin_ok, total),
            }
        )
        gap_rows.append(
            {
                "split": split,
                "group_field": group_field,
                "group_value": group_value,
                "pair_count": total,
                "mean_score_gap": _safe_mean(score_gaps),
                "median_score_gap": float(np.median(score_gaps)) if score_gaps else float("nan"),
                "mean_pair_margin": _safe_mean(margins),
                "min_score_gap": float(np.min(score_gaps)) if score_gaps else float("nan"),
                "max_score_gap": float(np.max(score_gaps)) if score_gaps else float("nan"),
            }
        )
    return accuracy_rows, gap_rows


def save_pairwise_diagnostics(output_dir: Any, pairs: list[dict[str, Any]], result: Any, split: str) -> None:
    accuracy_rows, gap_rows = pairwise_diagnostic_rows(pairs, result.predictions, split)
    tables_dir = output_dir / "tables"
    write_csv(tables_dir / f"pairwise_{split}_accuracy.csv", accuracy_rows)
    write_csv(tables_dir / f"pairwise_{split}_score_gap_metrics.csv", gap_rows)
    if split == "dev":
        write_csv(tables_dir / "pairwise_dev_accuracy.csv", accuracy_rows)
        write_csv(tables_dir / "pairwise_score_gap_metrics.csv", gap_rows)
