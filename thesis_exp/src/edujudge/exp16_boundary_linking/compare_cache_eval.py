"""Compare original Exp16A RQ1 diagnosis with eval-time boundary-cache diagnosis."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.src.edujudge.exp16_boundary_linking import EXP16_OUTPUT_DIR
from thesis_exp.src.edujudge.utils.io import read_csv, relpath, write_csv, write_text


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def rate(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def fmt(value: Any, digits: int = 4) -> str:
    try:
        if math.isnan(float(value)):
            return "nan"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def percentile(values: list[float], q: float) -> float:
    clean = [value for value in values if not math.isnan(value)]
    return float(np.percentile(clean, q)) if clean else float("nan")


def load_overall_metrics(diag_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    overall_path = diag_dir / "overall_metrics.csv"
    if overall_path.exists():
        return {
            (str(row["variant"]), str(row["split"])): row
            for row in read_csv(overall_path)
            if row.get("variant") and row.get("split")
        }
    margins_path = diag_dir / "margins_by_label.csv"
    if not margins_path.exists():
        return {}
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(margins_path):
        grouped[(str(row.get("variant") or ""), str(row.get("split") or ""))].append(row)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, rows in grouped.items():
        label_rows = {int(float(row["gold_label"])): row for row in rows if row.get("gold_label")}
        low_rows = [label_rows[label] for label in [1, 2] if label in label_rows]
        low_n = sum(safe_float(row.get("n")) for row in low_rows)
        low_to_high = sum(safe_float(row.get("pred_ge4_count")) for row in low_rows)
        out[key] = {
            "variant": key[0],
            "split": key[1],
            "low_to_high_rate": rate(low_to_high, low_n),
            "low_to_high_count": low_to_high,
            "true_low_n": low_n,
            "label1_recall": safe_float(label_rows.get(1, {}).get("recall")),
            "label2_recall": safe_float(label_rows.get(2, {}).get("recall")),
            "label5_recall": safe_float(label_rows.get(5, {}).get("recall")),
            "monotonic_violation_rate": float("nan"),
        }
    return out


def load_stability_metrics(diag_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    summary_path = diag_dir / "boundary_key_stability_summary.csv"
    if summary_path.exists():
        return {
            (str(row["variant"]), str(row["split"])): {
                "unstable_group_count": safe_float(row.get("unstable_group_count")),
                "p95_max_abs_tau_diff": safe_float(row.get("p95_max_abs_tau_diff")),
                "max_abs_tau_diff": safe_float(row.get("max_max_abs_tau_diff")),
            }
            for row in read_csv(summary_path)
            if row.get("variant") and row.get("split")
        }
    stability_path = diag_dir / "boundary_key_stability.csv"
    if not stability_path.exists():
        return {}
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(stability_path):
        grouped[(str(row.get("variant") or ""), str(row.get("split") or ""))].append(row)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, rows in grouped.items():
        max_diffs = [safe_float(row.get("max_abs_tau_diff")) for row in rows]
        out[key] = {
            "unstable_group_count": sum(1 for value in max_diffs if value > 1e-5),
            "p95_max_abs_tau_diff": percentile(max_diffs, 95),
            "max_abs_tau_diff": max([value for value in max_diffs if not math.isnan(value)], default=float("nan")),
        }
    return out


def conclusion_changed(row: dict[str, Any]) -> bool:
    cache_label2 = safe_float(row.get("cache_label2_recall"))
    cache_l2h = safe_float(row.get("cache_low_to_high_rate"))
    orig_label2 = safe_float(row.get("original_label2_recall"))
    if math.isnan(cache_label2) or math.isnan(cache_l2h):
        return False
    return (cache_label2 - orig_label2) > 0.05 or cache_l2h < 0.3


def build_rows(original_dir: Path, cache_dir: Path) -> list[dict[str, Any]]:
    original_metrics = load_overall_metrics(original_dir)
    cache_metrics = load_overall_metrics(cache_dir)
    original_stability = load_stability_metrics(original_dir)
    cache_stability = load_stability_metrics(cache_dir)
    cache_keys = set(cache_metrics) | set(cache_stability)
    keys = sorted(cache_keys or (set(original_metrics) | set(original_stability)))
    rows: list[dict[str, Any]] = []
    for key in keys:
        original = original_metrics.get(key, {})
        cache = cache_metrics.get(key, {})
        original_stab = original_stability.get(key, {})
        cache_stab = cache_stability.get(key, {})
        row = {
            "variant": key[0],
            "split": key[1],
            "original_low_to_high_rate": safe_float(original.get("low_to_high_rate")),
            "cache_low_to_high_rate": safe_float(cache.get("low_to_high_rate")),
            "delta_low_to_high_rate": safe_float(cache.get("low_to_high_rate")) - safe_float(original.get("low_to_high_rate")),
            "original_label1_recall": safe_float(original.get("label1_recall")),
            "cache_label1_recall": safe_float(cache.get("label1_recall")),
            "original_label2_recall": safe_float(original.get("label2_recall")),
            "cache_label2_recall": safe_float(cache.get("label2_recall")),
            "original_label5_recall": safe_float(original.get("label5_recall")),
            "cache_label5_recall": safe_float(cache.get("label5_recall")),
            "original_monotonic_violation": safe_float(original.get("monotonic_violation_rate")),
            "cache_monotonic_violation": safe_float(cache.get("monotonic_violation_rate")),
            "original_unstable_group_count": safe_float(original_stab.get("unstable_group_count")),
            "cache_unstable_group_count": safe_float(cache_stab.get("unstable_group_count")),
            "original_p95_max_abs_tau_diff": safe_float(original_stab.get("p95_max_abs_tau_diff")),
            "cache_p95_max_abs_tau_diff": safe_float(cache_stab.get("p95_max_abs_tau_diff")),
            "original_max_abs_tau_diff": safe_float(original_stab.get("max_abs_tau_diff")),
            "cache_max_abs_tau_diff": safe_float(cache_stab.get("max_abs_tau_diff")),
        }
        row["changes_s_separation_conclusion"] = conclusion_changed(row)
        rows.append(row)
    return rows


def write_report(output_dir: Path, rows: list[dict[str, Any]], original_dir: Path, cache_dir: Path) -> None:
    changed = any(bool(row.get("changes_s_separation_conclusion")) for row in rows)
    lines = [
        "# Exp16A Boundary Cache vs Original Diagnosis",
        "",
        f"Original diagnosis: `{relpath(original_dir)}`.",
        f"Boundary-cache diagnosis: `{relpath(cache_dir)}`.",
        "",
        "Boundary cache is an eval/export-only diagnostic. It does not change training, loss, or checkpoint selection.",
        "",
        "## Summary",
        "",
        "| variant | split | low-to-high original -> cache | label2 recall original -> cache | unstable groups original -> cache | p95 max tau diff original -> cache | max tau diff original -> cache |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['variant']}` | {row['split']} | "
            f"{fmt(row['original_low_to_high_rate'])} -> {fmt(row['cache_low_to_high_rate'])} | "
            f"{fmt(row['original_label2_recall'])} -> {fmt(row['cache_label2_recall'])} | "
            f"{fmt(row['original_unstable_group_count'], 0)} -> {fmt(row['cache_unstable_group_count'], 0)} | "
            f"{fmt(row['original_p95_max_abs_tau_diff'], 6)} -> {fmt(row['cache_p95_max_abs_tau_diff'], 6)} | "
            f"{fmt(row['original_max_abs_tau_diff'], 6)} -> {fmt(row['cache_max_abs_tau_diff'], 6)} |"
        )
    lines += [
        "",
        "## Conclusion",
        "",
    ]
    if changed:
        lines.append(
            "Boundary cache changes at least one split enough that the label2/quality-score separation conclusion should be rechecked manually."
        )
    else:
        lines.append(
            "Boundary cache does not change the main RQ1 conclusion: pointwise boundary linking still fails to recover label-2 answers, so the remaining failure is not explained only by repeated boundary encoding noise."
        )
    lines.append(
        "If cache reduces boundary_key tau variation to near zero while label2 recall remains zero and low-to-high remains high, the evidence still supports the interpretation that quality_score_s does not separate label-2 answers from middle/high-score answers well enough."
    )
    write_text(output_dir / "cache_vs_original_summary.md", "\n".join(lines))


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args.original_dir, args.cache_dir)
    write_csv(args.output_dir / "cache_vs_original_metrics.csv", rows)
    write_report(args.output_dir, rows, args.original_dir, args.cache_dir)
    return {
        "rows": len(rows),
        "output_dir": relpath(args.output_dir),
        "changed_conclusion": any(bool(row.get("changes_s_separation_conclusion")) for row in rows),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare original Exp16A RQ1 diagnosis with boundary-cache eval diagnosis.")
    parser.add_argument("--original_dir", type=Path, default=EXP16_OUTPUT_DIR / "rq1_diagnosis")
    parser.add_argument("--cache_dir", type=Path, default=EXP16_OUTPUT_DIR / "rq1_cache_eval" / "rq1_diagnosis")
    parser.add_argument("--output_dir", type=Path, default=EXP16_OUTPUT_DIR / "rq1_cache_eval")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
