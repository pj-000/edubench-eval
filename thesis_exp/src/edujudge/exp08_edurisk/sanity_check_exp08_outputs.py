"""Output sanity checks for a completed Exp8 QD-ER1 run."""

from __future__ import annotations

import argparse
import math
from typing import Any

from thesis_exp.src.edujudge.exp08_edurisk import EXP08_OUTPUT_DIR, EXP08_TABLES_DIR, ensure_exp08_dirs, exp08_run_dir
from thesis_exp.src.edujudge.utils.io import read_csv, relpath, write_csv, write_text


REQUIRED_RUN_FILES = [
    "predictions/predictions_dev.jsonl",
    "predictions/predictions_test.jsonl",
    "arrays/dev_test_arrays.npz",
    "tables/metrics_summary.csv",
    "tables/low_score_metrics.csv",
    "tables/high_score_metrics.csv",
    "tables/monotonicity_metrics.csv",
    "tables/expected_risk_metrics.csv",
    "tables/cumulative_decoding_vs_argmax_decoding.csv",
    "tables/loss_component_scale.csv",
    "run_metadata.json",
    "run_summary.md",
]


def add(rows: list[dict[str, Any]], check_name: str, passed: bool, details: Any = "") -> None:
    rows.append({"check_name": check_name, "status": "PASS" if passed else "FAIL", "details": details})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Exp8 QD-ER1 run outputs.")
    parser.add_argument("--smoke", action="store_true", help="Check the smoke-test run directory instead of formal run.")
    return parser.parse_args()


def _float_is_zero(value: Any, tolerance: float = 1e-8) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and abs(number) <= tolerance


def main() -> None:
    args = parse_args()
    ensure_exp08_dirs()
    rows: list[dict[str, Any]] = []
    run_kind = "smoke" if args.smoke else "formal"
    run_dir = exp08_run_dir(args.smoke)
    add(rows, f"{run_kind} run dir exists", run_dir.exists(), relpath(run_dir))
    for rel in REQUIRED_RUN_FILES:
        path = run_dir / rel
        add(rows, f"required output exists {rel}", path.exists(), relpath(path))
    metrics_path = run_dir / "tables" / "metrics_summary.csv"
    if metrics_path.exists():
        metrics = read_csv(metrics_path)
        test = next((row for row in metrics if row.get("split") == "test"), {})
        add(rows, "test metrics row exists", bool(test), test)
        add(
            rows,
            "test monotonic violation is zero",
            _float_is_zero(test.get("monotonic_violation_rate")),
            test.get("monotonic_violation_rate"),
        )
    suffix = "_smoke" if args.smoke else ""
    csv_path = EXP08_TABLES_DIR / f"sanity_check_exp08_outputs{suffix}.csv"
    md_path = EXP08_OUTPUT_DIR / f"sanity_check_exp08_outputs{suffix}.md"
    write_csv(csv_path, rows)
    failures = [row for row in rows if row["status"] != "PASS"]
    lines = [
        "# Exp8 Output Sanity Check",
        "",
        f"Run kind: `{run_kind}`",
        f"Status: `{'PASS' if not failures else 'FAIL'}`",
        "",
        "| check | status | details |",
        "| --- | --- | --- |",
    ]
    lines.extend(f"| {row['check_name']} | {row['status']} | {row.get('details', '')} |" for row in rows)
    write_text(md_path, "\n".join(lines))
    if failures:
        raise SystemExit(f"Exp8 output sanity failed. See {relpath(md_path)}")
    print("Exp8 output sanity PASS")


if __name__ == "__main__":
    main()
