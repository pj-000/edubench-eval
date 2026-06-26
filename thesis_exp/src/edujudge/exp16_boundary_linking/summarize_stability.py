"""Summarize Exp16A qmr/metric_rubric multi-seed stability."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from thesis_exp.src.edujudge.exp16_boundary_linking import EXP16_OUTPUT_DIR
from thesis_exp.src.edujudge.utils.io import relpath, write_csv, write_text


DEFAULT_VARIANTS = ("qmr", "metric_rubric")
DEFAULT_SEEDS = (42, 43, 44)


def warn(message: str) -> None:
    print(f"WARNING: {message}")


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def fmt(value: Any, digits: int = 4) -> str:
    try:
        if math.isnan(float(value)):
            return "nan"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def metrics_path(input_root: Path, variant: str, seed: int) -> Path | None:
    candidates = [
        input_root / f"scout_seed{seed}" / variant / "metrics_dev.json",
        input_root / "runs" / variant / f"seed_{seed}" / "metrics_dev.json",
        input_root / variant / f"seed_{seed}" / "metrics_dev.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    warn(f"missing dev metrics for variant={variant} seed={seed}; checked: {', '.join(relpath(p) for p in candidates)}")
    return None


def selected_epoch(input_root: Path, variant: str, seed: int, metrics: dict[str, Any]) -> Any:
    if "epoch" in metrics:
        return metrics["epoch"]
    candidates = [
        input_root / f"scout_seed{seed}" / variant / "config.json",
        input_root / "runs" / variant / f"seed_{seed}" / "config.json",
        input_root / variant / f"seed_{seed}" / "config.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                config = json.loads(path.read_text())
                return config.get("best_metrics", {}).get("epoch", "")
            except Exception:
                return ""
    return ""


def read_rows(input_root: Path, variants: list[str], seeds: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        for seed in seeds:
            path = metrics_path(input_root, variant, seed)
            if path is None:
                continue
            metrics = json.loads(path.read_text())
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "dev_MAE": metrics.get("MAE"),
                    "dev_QWK": metrics.get("QWK"),
                    "dev_Accuracy": metrics.get("Accuracy"),
                    "dev_low_to_high_count": metrics.get("low_to_high_count"),
                    "dev_true_low_n": metrics.get("true_low_n"),
                    "dev_low_to_high_rate": metrics.get("low_to_high_rate"),
                    "dev_high_to_low_count": metrics.get("high_to_low_count"),
                    "dev_high_to_low_rate": metrics.get("high_to_low_rate"),
                    "dev_label1_recall": metrics.get("label1_recall", metrics.get("recall_label_1")),
                    "dev_label2_recall": metrics.get("label2_recall", metrics.get("recall_label_2")),
                    "dev_label5_recall": metrics.get("label5_recall", metrics.get("recall_label_5")),
                    "dev_monotonic_violation": metrics.get("monotonic_violation_rate"),
                    "selected_epoch": selected_epoch(input_root, variant, seed, metrics),
                }
            )
    return rows


def aggregate(rows: list[dict[str, Any]], variants: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for variant in variants:
        items = [row for row in rows if row["variant"] == variant]
        if not items:
            continue

        def values(key: str) -> list[float]:
            return [safe_float(row.get(key)) for row in items if not math.isnan(safe_float(row.get(key)))]

        mae = values("dev_MAE")
        qwk = values("dev_QWK")
        l2h = values("dev_low_to_high_rate")
        out.append(
            {
                "variant": variant,
                "n_seeds": len(items),
                "mean_MAE": mean(mae) if mae else float("nan"),
                "std_MAE": pstdev(mae) if len(mae) > 1 else 0.0,
                "mean_QWK": mean(qwk) if qwk else float("nan"),
                "std_QWK": pstdev(qwk) if len(qwk) > 1 else 0.0,
                "mean_low_to_high_rate": mean(l2h) if l2h else float("nan"),
                "std_low_to_high_rate": pstdev(l2h) if len(l2h) > 1 else 0.0,
                "mean_label1_recall": mean(values("dev_label1_recall")) if values("dev_label1_recall") else float("nan"),
                "mean_label2_recall": mean(values("dev_label2_recall")) if values("dev_label2_recall") else float("nan"),
                "mean_label5_recall": mean(values("dev_label5_recall")) if values("dev_label5_recall") else float("nan"),
                "all_monotonic_zero": all(abs(safe_float(row.get("dev_monotonic_violation"))) <= 1e-8 for row in items),
            }
        )
    return out


def write_report(output_dir: Path, rows: list[dict[str, Any]], agg: list[dict[str, Any]], seeds: list[int]) -> None:
    lines = [
        "# Exp16A-Stability Summary",
        "",
        "This report summarizes qmr and metric_rubric dev stability across available seeds. Missing seeds are warnings, not failures.",
        "",
        f"Requested seeds: `{', '.join(str(seed) for seed in seeds)}`.",
        f"Available rows: `{len(rows)}`.",
        "",
        "## Aggregate",
        "",
        "| variant | n seeds | mean MAE | std MAE | mean QWK | std QWK | mean low-to-high | mean label2 recall | monotonic zero |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in agg:
        lines.append(
            f"| `{row['variant']}` | {row['n_seeds']} | {fmt(row['mean_MAE'])} | {fmt(row['std_MAE'])} | "
            f"{fmt(row['mean_QWK'])} | {fmt(row['std_QWK'])} | {fmt(row['mean_low_to_high_rate'])} | "
            f"{fmt(row['mean_label2_recall'])} | {row['all_monotonic_zero']} |"
        )
    qmr = next((row for row in agg if row["variant"] == "qmr"), None)
    mr = next((row for row in agg if row["variant"] == "metric_rubric"), None)
    lines += ["", "## Interpretation", ""]
    if qmr and mr:
        if qmr["n_seeds"] < 2 or mr["n_seeds"] < 2:
            lines.append("- Multi-seed stability is not yet established; run seed 43/44 before making a stability claim.")
        if safe_float(qmr["mean_MAE"]) < safe_float(mr["mean_MAE"]):
            lines.append("- qmr has lower available-seed dev MAE than metric_rubric.")
        else:
            lines.append("- metric_rubric has lower available-seed dev MAE than qmr.")
        if safe_float(qmr["mean_low_to_high_rate"]) < safe_float(mr["mean_low_to_high_rate"]):
            lines.append("- qmr has lower available-seed low-to-high rate.")
        else:
            lines.append("- metric_rubric has lower or equal available-seed low-to-high rate.")
        if safe_float(qmr["mean_label2_recall"]) == 0.0 and safe_float(mr["mean_label2_recall"]) == 0.0:
            lines.append("- label2 recall remains zero for both variants in the available seeds.")
    else:
        lines.append("- Not enough variant rows are available for qmr vs metric_rubric comparison.")
    lines += [
        "",
        "## RQ1 Exit Condition",
        "",
        "RQ1 can be treated as ready to close only after the boundary diagnostic and at least the available-seed stability summary support a consistent boundary-generation story. If seed 43/44 are missing, this report should be read as a seed42 snapshot rather than a stability result.",
    ]
    write_text(output_dir / "exp16a_stability_report.md", "\n".join(lines))


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.input_root, args.variants, args.seeds)
    agg = aggregate(rows, args.variants)
    write_csv(args.output_dir / "exp16a_stability_summary.csv", rows)
    write_csv(args.output_dir / "exp16a_stability_aggregate.csv", agg)
    write_report(args.output_dir, rows, agg, args.seeds)
    return {"rows": len(rows), "output_dir": relpath(args.output_dir)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Exp16A qmr/metric_rubric multi-seed stability.")
    parser.add_argument("--input_root", type=Path, default=EXP16_OUTPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=EXP16_OUTPUT_DIR / "rq1_stability")
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
