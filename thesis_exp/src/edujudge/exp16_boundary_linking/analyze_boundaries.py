"""Analyze Exp16A boundary-linking predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp16_boundary_linking.metrics import margins_by_gold_label, threshold_stats_by_metric
from thesis_exp.src.edujudge.utils.io import read_jsonl, write_csv, write_text


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def analyze(predictions_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = read_jsonl(predictions_path)
    split = "dev" if "dev" in predictions_path.name else "test"
    by_metric = threshold_stats_by_metric(predictions, split=split)
    margins = margins_by_gold_label(predictions, split=split)
    low_cases = [row for row in predictions if bool(row.get("is_low_to_high"))]
    write_csv(output_dir / "boundary_by_metric.csv", by_metric)
    write_csv(output_dir / "margins_by_gold_label.csv", margins)
    write_csv(output_dir / "low_to_high_cases.csv", low_cases)
    l2h_n = len(low_cases)
    l2h_high_s = sum(1 for row in low_cases if float(row["quality_score_s"]) > float(row["tau3"]))
    l2h_low_tau = sum(1 for row in low_cases if float(row["margin_tau3"]) > 0)
    lines = [
        "# Exp16A Boundary Analysis",
        "",
        f"Prediction file: `{predictions_path}`",
        f"Rows: `{len(predictions)}`",
        f"Low-to-high cases: `{l2h_n}`",
        "",
        "## Margin By Gold Label",
        "",
        "| gold | n | s mean | tau2 mean | tau3 mean | margin tau3 mean |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in margins:
        lines.append(
            f"| {row['gold_label']} | {row['n']} | {_fmt(row['quality_score_s_mean'])} | "
            f"{_fmt(row['tau2_mean'])} | {_fmt(row['tau3_mean'])} | {_fmt(row['margin_tau3_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Low-To-High Diagnosis",
            "",
            f"- Cases where `s > tau3`: `{l2h_high_s}` / `{l2h_n}`.",
            f"- Cases with positive `margin_tau3`: `{l2h_low_tau}` / `{l2h_n}`.",
            "- If most low-to-high cases have high `s`, the quality tower is overestimating low answers.",
            "- If a metric has unusually low `tau3/tau4`, the boundary tower may be too permissive for that context.",
            "",
            "## Metric Boundary Summary",
            "",
            "| metric | n | low-to-high | tau2 mean | tau3 mean | tau4 mean |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in by_metric:
        lines.append(
            f"| {row['metric']} | {row['n']} | {row['low_to_high_count']}/{row['true_low_n']} | "
            f"{_fmt(row['tau2_mean'])} | {_fmt(row['tau3_mean'])} | {_fmt(row['tau4_mean'])} |"
        )
    write_text(output_dir / "boundary_analysis.md", "\n".join(lines))
    return {"rows": len(predictions), "low_to_high_cases": l2h_n, "output_dir": str(output_dir)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze Exp16A boundary predictions.")
    parser.add_argument("--predictions_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze(args.predictions_path, args.output_dir)
    print(result)


if __name__ == "__main__":
    main()
