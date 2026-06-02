"""Sanity checks for generated Exp1 artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from thesis_exp.src.edujudge.audit import (
    EVALUATORS,
    EXPECTED_TEST_ROWS,
    EXP01_FIGURES_DIR,
    EXP01_OUTPUT_DIR,
    EXP01_TABLES_DIR,
    ensure_exp01_dirs,
    is_synthetic_or_sampled_path,
    markdown_table,
    read_jsonl,
    relpath,
    write_csv,
    write_text,
)


def _check_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(check: str, status: str, observed: Any, expected: Any, notes: str = "") -> None:
        rows.append({"check": check, "status": status, "observed": observed, "expected": expected, "notes": notes})

    predictions_path = EXP01_OUTPUT_DIR / "predictions_aligned.jsonl"
    add("predictions_aligned.jsonl exists", "PASS" if predictions_path.exists() else "FAIL", predictions_path.exists(), True)
    aligned = read_jsonl(predictions_path) if predictions_path.exists() else []
    add(
        "predictions_aligned rows == 2218",
        "PASS" if len(aligned) == EXPECTED_TEST_ROWS else "WARN",
        len(aligned),
        EXPECTED_TEST_ROWS,
        "Non-blocking warning if the locked split changes.",
    )

    for filename in [
        "evaluator_metrics.csv",
        "per_bin_metrics.csv",
        "low_score_metrics.csv",
        "alignment_coverage.csv",
    ]:
        path = EXP01_TABLES_DIR / filename
        try:
            table = pd.read_csv(path)
            add(f"{filename} readable", "PASS", f"{len(table)} rows / {len(table.columns)} cols", "readable CSV")
        except Exception as exc:  # noqa: BLE001
            add(f"{filename} readable", "FAIL", type(exc).__name__, "readable CSV", str(exc))

    coverage_path = EXP01_TABLES_DIR / "alignment_coverage.csv"
    if coverage_path.exists():
        coverage = pd.read_csv(coverage_path)
        found = coverage[coverage["n_aligned"] > 0]["evaluator"].tolist()
    else:
        found = []
    add("at least one evaluator found", "PASS" if found else "FAIL", found, ">=1 evaluator")

    invalid_values: list[str] = []
    for record in aligned:
        for spec in EVALUATORS:
            value = record.get(f"pred_{spec.field_suffix}")
            valid = record.get(f"valid_{spec.field_suffix}")
            if valid and not (1 <= float(value) <= 5):
                invalid_values.append(f"{record.get('record_id')}:{spec.name}:{value}")
    add("all valid predictions in 1-5", "PASS" if not invalid_values else "FAIL", len(invalid_values), 0)

    synthetic_sources = set()
    for record in aligned:
        for spec in EVALUATORS:
            source = record.get(f"alignment_source_{spec.field_suffix}")
            if source and is_synthetic_or_sampled_path(source):
                synthetic_sources.add(source)
    add("no synthetic/sample rows used", "PASS" if not synthetic_sources else "FAIL", sorted(synthetic_sources), [])

    per_bin_path = EXP01_TABLES_DIR / "per_bin_metrics.csv"
    calibration_ok = False
    if per_bin_path.exists():
        per_bin = pd.read_csv(per_bin_path)
        calibration_ok = "mean_pred" in per_bin.columns and per_bin["mean_pred"].notna().any()
    add("calibration curve data exists", "PASS" if calibration_ok else "FAIL", calibration_ok, True)

    png_count = len(list(EXP01_FIGURES_DIR.glob("*.png")))
    pdf_count = len(list(EXP01_FIGURES_DIR.glob("*.pdf")))
    add("at least 10 png figures exist", "PASS" if png_count >= 10 else "FAIL", png_count, ">=10")
    add("at least 10 pdf figures exist", "PASS" if pdf_count >= 10 else "FAIL", pdf_count, ">=10")

    for filename in ["report.md", "review_package.md"]:
        path = EXP01_OUTPUT_DIR / filename
        if path.exists():
            line_count = len(path.read_text(encoding="utf-8").splitlines())
            status = "PASS" if line_count >= 10 else "FAIL"
            observed = f"{line_count} lines"
        else:
            status = "FAIL"
            observed = "missing"
        add(f"{filename} exists and readable", status, observed, "multi-line markdown")
    return rows


def run_sanity_check() -> list[dict[str, Any]]:
    ensure_exp01_dirs()
    rows = _check_rows()
    write_csv(EXP01_TABLES_DIR / "sanity_check_exp01.csv", rows, ["check", "status", "observed", "expected", "notes"])
    overall = "PASS" if all(row["status"] == "PASS" for row in rows) else "WARN/FAIL"
    text = f"""# Exp1 Sanity Check

Overall status: **{overall}**

{markdown_table(rows, ["check", "status", "observed", "expected", "notes"])}
"""
    write_text(EXP01_OUTPUT_DIR / "sanity_check_exp01.md", text)
    return rows


def main() -> None:
    rows = run_sanity_check()
    statuses = {row["status"] for row in rows}
    print(f"Exp1 sanity statuses: {', '.join(sorted(statuses))}")
    print(f"Outputs: {relpath(EXP01_OUTPUT_DIR / 'sanity_check_exp01.md')}")


if __name__ == "__main__":
    main()

