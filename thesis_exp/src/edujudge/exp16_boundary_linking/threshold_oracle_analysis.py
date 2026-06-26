"""Threshold-oracle analysis for Exp16A fixed quality scores."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.src.edujudge.exp16_boundary_linking import EXP16_OUTPUT_DIR
from thesis_exp.src.edujudge.exp16_boundary_linking.metrics import compute_metrics
from thesis_exp.src.edujudge.utils.io import read_jsonl, relpath, write_csv, write_text


DEFAULT_VARIANTS = ("qmr", "metric_rubric")
DEFAULT_SPLITS = ("dev", "test")
INF = 10**12


def warn(message: str) -> None:
    print(f"WARNING: {message}")


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def prediction_path(input_root: Path, variant: str, split: str, seed: int) -> Path | None:
    candidates = [
        input_root / f"scout_seed{seed}" / variant / f"predictions_{split}.jsonl",
        input_root / "runs" / variant / f"seed_{seed}" / f"predictions_{split}.jsonl",
        input_root / variant / f"seed_{seed}" / f"predictions_{split}.jsonl",
        input_root / variant / f"predictions_{split}.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return path
    warn(f"missing predictions for variant={variant} split={split}; checked: {', '.join(relpath(p) for p in candidates)}")
    return None


def load_variant_predictions(input_root: Path, variant: str, seed: int) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for split in ["train", "dev", "test"]:
        path = prediction_path(input_root, variant, split, seed)
        if path is not None:
            out[split] = read_jsonl(path)
    return out


def fit_thresholds(scores: list[float], labels: list[int]) -> list[float]:
    pairs = sorted((float(score), int(label)) for score, label in zip(scores, labels) if not math.isnan(float(score)))
    n = len(pairs)
    if n < 5:
        if not pairs:
            return [0.0, 1.0, 2.0, 3.0]
        values = [score for score, _ in pairs]
        return [float(np.quantile(values, q)) for q in [0.2, 0.4, 0.6, 0.8]]
    sorted_scores = [score for score, _ in pairs]
    sorted_labels = [label for _, label in pairs]
    prefix = {label: [0] * (n + 1) for label in range(1, 6)}
    for idx, label in enumerate(sorted_labels, start=1):
        for value in range(1, 6):
            prefix[value][idx] = prefix[value][idx - 1] + abs(label - value)

    def cost(start: int, end: int, pred_label: int) -> int:
        return prefix[pred_label][end] - prefix[pred_label][start]

    dp = [[INF] * (n + 1) for _ in range(6)]
    back = [[-1] * (n + 1) for _ in range(6)]
    dp[0][0] = 0
    for k in range(1, 6):
        min_i = k
        max_i = n - (5 - k)
        for i in range(min_i, max_i + 1):
            best = INF
            best_j = -1
            for j in range(k - 1, i):
                score = dp[k - 1][j] + cost(j, i, k)
                if score < best:
                    best = score
                    best_j = j
            dp[k][i] = best
            back[k][i] = best_j
    ends: list[int] = []
    i = n
    for k in range(5, 0, -1):
        ends.append(i)
        i = back[k][i]
    ends = list(reversed(ends))
    thresholds: list[float] = []
    for end in ends[:-1]:
        left = sorted_scores[max(0, end - 1)]
        right = sorted_scores[min(n - 1, end)]
        thresholds.append(float((left + right) / 2.0))
    return thresholds


def predict_from_thresholds(score: float, thresholds: list[float]) -> int:
    return 1 + sum(float(score) > threshold for threshold in thresholds)


def rows_with_thresholds(rows: list[dict[str, Any]], thresholds_by_metric: dict[str, list[float]] | None, global_thresholds: list[float]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        thresholds = global_thresholds
        if thresholds_by_metric is not None:
            thresholds = thresholds_by_metric.get(str(row.get("metric") or ""), global_thresholds)
        pred = predict_from_thresholds(safe_float(row.get("quality_score_s")), thresholds)
        out.append(
            {
                "sample_id": row.get("sample_id"),
                "metric": row.get("metric"),
                "gold_label": int(row.get("gold_label")),
                "pred_label": int(pred),
                "probs": [1.0 if pred > threshold else 0.0 for threshold in [1, 2, 3, 4]],
            }
        )
    return out


def metric_thresholds(train_rows: list[dict[str, Any]], global_thresholds: list[float], min_metric_n: int) -> dict[str, list[float]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        groups[str(row.get("metric") or "")].append(row)
    out: dict[str, list[float]] = {}
    for metric, items in groups.items():
        if len(items) < min_metric_n:
            out[metric] = global_thresholds
            continue
        out[metric] = fit_thresholds(
            [safe_float(row.get("quality_score_s")) for row in items],
            [int(row.get("gold_label")) for row in items],
        )
    return out


def summary_row(variant: str, split: str, method: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = compute_metrics(rows, split=split)
    return {
        "variant": variant,
        "split": split,
        "threshold_method": method,
        "MAE": metrics.get("MAE"),
        "QWK": metrics.get("QWK"),
        "Accuracy": metrics.get("Accuracy"),
        "low_to_high_count": metrics.get("low_to_high_count"),
        "true_low_n": metrics.get("true_low_n"),
        "low_to_high_rate": metrics.get("low_to_high_rate"),
        "label1_recall": metrics.get("label1_recall", metrics.get("recall_label_1")),
        "label2_recall": metrics.get("label2_recall", metrics.get("recall_label_2")),
        "label5_recall": metrics.get("label5_recall", metrics.get("recall_label_5")),
        "monotonic_violation": metrics.get("monotonic_violation_rate"),
    }


def by_metric_rows(variant: str, split: str, method: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("metric") or "")].append(row)
    out: list[dict[str, Any]] = []
    for metric, items in sorted(groups.items()):
        metrics = compute_metrics(items, split=split)
        label2_n = sum(1 for row in items if int(row.get("gold_label", 0)) == 2)
        label2_hits = sum(1 for row in items if int(row.get("gold_label", 0)) == 2 and int(row.get("pred_label", 0)) == 2)
        out.append(
            {
                "variant": variant,
                "split": split,
                "threshold_method": method,
                "metric": metric,
                "n": len(items),
                "MAE": metrics.get("MAE"),
                "low_to_high_count": metrics.get("low_to_high_count"),
                "true_low_n": metrics.get("true_low_n"),
                "low_to_high_rate": metrics.get("low_to_high_rate"),
                "label2_n": label2_n,
                "label2_recall": float(label2_hits / label2_n) if label2_n else float("nan"),
            }
        )
    return out


def fmt(value: Any, digits: int = 4) -> str:
    try:
        if math.isnan(float(value)):
            return "nan"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def write_report(output_dir: Path, summary: list[dict[str, Any]], warnings: list[str]) -> None:
    lines = [
        "# Exp16A-Oracle Threshold Analysis",
        "",
        "This diagnostic fixes the learned quality score `s` and changes only thresholding rules. It does not train a model.",
        "",
    ]
    if warnings:
        lines += ["## Warnings", ""]
        lines.extend(f"- {message}" for message in warnings)
        lines.append("")
    lines += [
        "## Summary",
        "",
        "| variant | split | method | MAE | QWK | Accuracy | low-to-high | label2 recall |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| `{row['variant']}` | {row['split']} | `{row['threshold_method']}` | {fmt(row['MAE'])} | "
            f"{fmt(row['QWK'])} | {fmt(row['Accuracy'])} | {row['low_to_high_count']}/{row['true_low_n']} "
            f"({fmt(row['low_to_high_rate'])}) | {fmt(row['label2_recall'])} |"
        )
    lines += [
        "",
        "## Interpretation Guide",
        "",
        "- If global or metric thresholds improve label2 recall while MAE/QWK remain similar, the main weakness is tau calibration.",
        "- If threshold calibration cannot recover label2 recall, the learned quality score `s` is not separating label-2 answers from higher-score answers.",
        "- `oracle_dev_threshold_on_s` is a diagnostic upper bound only; it is not a valid model-selection result.",
        "",
        "## RQ1 Recommendation",
        "",
        "Use this table with the boundary-failure diagnosis: if fixed-s calibration helps, RQ1 should close with a calibration/boundary story; if it does not help, close RQ1 by noting that boundary generation alone is insufficient and move to RQ2 risk-aware learning.",
    ]
    write_text(output_dir / "threshold_oracle_report.md", "\n".join(lines))


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, Any]] = []
    metric_summary: list[dict[str, Any]] = []
    warnings: list[str] = []
    for variant in args.variants:
        data = load_variant_predictions(args.input_root, variant, args.seed)
        for split in args.splits:
            if split in data:
                original = data[split]
                summary.append(summary_row(variant, split, "boundary_tower_tau", original))
                metric_summary.extend(by_metric_rows(variant, split, "boundary_tower_tau", original))
        if "train" not in data:
            message = f"{variant}: missing train predictions; skipping global_threshold_on_s and metric_threshold_on_s"
            warn(message)
            warnings.append(message)
        else:
            train_rows = data["train"]
            global_thresholds = fit_thresholds(
                [safe_float(row.get("quality_score_s")) for row in train_rows],
                [int(row.get("gold_label")) for row in train_rows],
            )
            per_metric = metric_thresholds(train_rows, global_thresholds, args.min_metric_n)
            for split in args.splits:
                if split not in data:
                    continue
                global_rows = rows_with_thresholds(data[split], None, global_thresholds)
                metric_rows = rows_with_thresholds(data[split], per_metric, global_thresholds)
                summary.append(summary_row(variant, split, "global_threshold_on_s", global_rows))
                summary.append(summary_row(variant, split, "metric_threshold_on_s", metric_rows))
                metric_summary.extend(by_metric_rows(variant, split, "global_threshold_on_s", global_rows))
                metric_summary.extend(by_metric_rows(variant, split, "metric_threshold_on_s", metric_rows))
        if "dev" in data:
            oracle_thresholds = fit_thresholds(
                [safe_float(row.get("quality_score_s")) for row in data["dev"]],
                [int(row.get("gold_label")) for row in data["dev"]],
            )
            oracle_rows = rows_with_thresholds(data["dev"], None, oracle_thresholds)
            summary.append(summary_row(variant, "dev", "oracle_dev_threshold_on_s", oracle_rows))
            metric_summary.extend(by_metric_rows(variant, "dev", "oracle_dev_threshold_on_s", oracle_rows))
    write_csv(args.output_dir / "threshold_oracle_summary.csv", summary)
    write_csv(args.output_dir / "threshold_oracle_by_metric.csv", metric_summary)
    write_report(args.output_dir, summary, warnings)
    return {"summary_rows": len(summary), "output_dir": relpath(args.output_dir)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fixed-s threshold-oracle diagnostics for Exp16A.")
    parser.add_argument("--input_root", type=Path, default=EXP16_OUTPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=EXP16_OUTPUT_DIR / "rq1_threshold_oracle")
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_metric_n", type=int, default=30)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
