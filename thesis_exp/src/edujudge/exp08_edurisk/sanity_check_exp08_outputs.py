"""Output sanity checks for a completed Exp8 QD-ER1 run."""

from __future__ import annotations

from pathlib import Path
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


def main() -> None:
    ensure_exp08_dirs()
    rows: list[dict[str, Any]] = []
    run_dir = exp08_run_dir(False)
    add(rows, "formal run dir exists", run_dir.exists(), relpath(run_dir))
    for rel in REQUIRED_RUN_FILES:
        path = run_dir / rel
        add(rows, f"required output exists {rel}", path.exists(), relpath(path))
    metrics_path = run_dir / "tables" / "metrics_summary.csv"
    if metrics_path.exists():
        metrics = read_csv(metrics_path)
        test = next((row for row in metrics if row.get("split") == "test"), {})
        add(rows, "test metrics row exists", bool(test), test)
        add(rows, "test monotonic violation is zero", test.get("monotonic_violation_rate") in {"0", "0.0", "0.0000"}, test.get("monotonic_violation_rate"))
    write_csv(EXP08_TABLES_DIR / "sanity_check_exp08_outputs.csv", rows)
    failures = [row for row in rows if row["status"] != "PASS"]
    lines = [
        "# Exp8 Output Sanity Check",
        "",
        f"Status: `{'PASS' if not failures else 'FAIL'}`",
        "",
        "| check | status | details |",
        "| --- | --- | --- |",
    ]
    lines.extend(f"| {row['check_name']} | {row['status']} | {row.get('details', '')} |" for row in rows)
    write_text(EXP08_OUTPUT_DIR / "sanity_check_exp08_outputs.md", "\n".join(lines))
    if failures:
        raise SystemExit(f"Exp8 output sanity failed. See {relpath(EXP08_OUTPUT_DIR / 'sanity_check_exp08_outputs.md')}")
    print("Exp8 output sanity PASS")


if __name__ == "__main__":
    main()
