"""Readability and safety checks for Exp15 outputs."""

from __future__ import annotations

import csv
import py_compile
import subprocess
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp15_posthoc_ordinal_calibration import (
    EXP15_CONFIG_DIR,
    EXP15_OUTPUT_DIR,
    EXP15_REPORTS_DIR,
    EXP15_TABLES_DIR,
    ensure_exp15_dirs,
)
from thesis_exp.src.edujudge.utils.io import relpath, write_csv, write_text


SOURCE_PATHS = [
    Path("thesis_exp/src/edujudge/exp15_posthoc_ordinal_calibration/__init__.py"),
    Path("thesis_exp/src/edujudge/exp15_posthoc_ordinal_calibration/calibration.py"),
    Path("thesis_exp/src/edujudge/exp15_posthoc_ordinal_calibration/run_exp15_posthoc_ordinal_calibration.py"),
    Path("thesis_exp/src/edujudge/exp15_posthoc_ordinal_calibration/smoke_check_exp15.py"),
    Path("thesis_exp/src/edujudge/exp15_posthoc_ordinal_calibration/readability_check_exp15.py"),
]
SCRIPT_PATHS = [
    Path("thesis_exp/scripts/run_exp15_posthoc_ordinal_calibration.sh"),
    Path("thesis_exp/scripts/sync_exp15_posthoc_ordinal_calibration_to_server.sh"),
]
REQUIRED_TABLES = [
    "exp15_source_inventory.csv",
    "exp15_calibration_dev_ranking.csv",
    "exp15_selected_calibrators.csv",
    "exp15_test_diagnostic_metrics.csv",
    "exp15_selected_test_metrics.csv",
    "exp15_prediction_distribution.csv",
]
REQUIRED_REPORTS = ["exp15_posthoc_ordinal_calibration_report.md"]
REQUIRED_CONFIGS = ["exp15_posthoc_ordinal_calibration.yaml"]
HEAVY_SUFFIXES = {".pt", ".pth", ".bin", ".ckpt", ".safetensors", ".npy", ".npz", ".jsonl"}
RAW_NAMES = {"predictions", "arrays", "logs"}


def add(rows: list[dict[str, Any]], check_name: str, status: bool, details: Any = "") -> None:
    rows.append({"check_name": check_name, "status": "PASS" if status else "FAIL", "details": details})


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _csv_headers(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or [])


def main() -> None:
    ensure_exp15_dirs()
    rows: list[dict[str, Any]] = []
    for path in SOURCE_PATHS:
        try:
            py_compile.compile(str(path), doraise=True)
            add(rows, f"py_compile {relpath(path)}", True)
        except py_compile.PyCompileError as exc:
            add(rows, f"py_compile {relpath(path)}", False, str(exc))
    for path in SCRIPT_PATHS:
        result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        add(rows, f"bash -n {relpath(path)}", result.returncode == 0, (result.stderr or result.stdout).strip())
    for name in REQUIRED_CONFIGS:
        path = EXP15_CONFIG_DIR / name
        add(rows, f"required config {name}", path.exists(), relpath(path))
    for name in REQUIRED_TABLES:
        path = EXP15_TABLES_DIR / name
        add(rows, f"required table {name}", path.exists(), relpath(path))
        if path.exists():
            try:
                _csv_rows(path)
                add(rows, f"CSV readable {name}", True)
            except Exception as exc:
                add(rows, f"CSV readable {name}", False, f"{type(exc).__name__}: {exc}")
    for name in REQUIRED_REPORTS:
        path = EXP15_REPORTS_DIR / name
        add(rows, f"required report {name}", path.exists(), relpath(path))

    ranking_path = EXP15_TABLES_DIR / "exp15_calibration_dev_ranking.csv"
    if ranking_path.exists():
        headers = _csv_headers(ranking_path)
        add(rows, "dev ranking contains no test columns", not any(header.startswith("test_") for header in headers))
        ranking_rows = _csv_rows(ranking_path)
        add(
            rows,
            "dev ranking uses_test_for_selection false",
            all(str(row.get("uses_test_for_selection", "")).lower() == "false" for row in ranking_rows),
        )

    test_path = EXP15_TABLES_DIR / "exp15_selected_test_metrics.csv"
    if test_path.exists():
        test_rows = _csv_rows(test_path)
        add(
            rows,
            "test metrics marked final diagnostic only",
            all(str(row.get("uses_test_for_selection", "")).lower() == "false" for row in test_rows),
        )

    report_path = EXP15_REPORTS_DIR / "exp15_posthoc_ordinal_calibration_report.md"
    if report_path.exists():
        text = report_path.read_text(encoding="utf-8").lower()
        add(rows, "report states dev-only diagnosis", "dev-only" in text and "does not use" in text and "test metrics" in text)
        add(rows, "report avoids overclaim", not any(word in text for word in ["proves", "solves", "guarantees"]))

    heavy = []
    for path in EXP15_OUTPUT_DIR.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in HEAVY_SUFFIXES or any(part in RAW_NAMES for part in path.parts):
            heavy.append(relpath(path))
    add(rows, "no heavy/raw artifacts under tracked Exp15 outputs", not heavy, heavy)

    write_csv(EXP15_TABLES_DIR / "readability_check_exp15.csv", rows)
    failures = [row for row in rows if row["status"] != "PASS"]
    lines = [
        "# Exp15 Readability Check",
        "",
        f"Status: `{'PASS' if not failures else 'FAIL'}`",
        "",
        "| check | status | details |",
        "| --- | --- | --- |",
    ]
    lines.extend(f"| {row['check_name']} | {row['status']} | {row.get('details', '')} |" for row in rows)
    write_text(EXP15_REPORTS_DIR / "exp15_readability_report.md", "\n".join(lines))
    write_text(EXP15_OUTPUT_DIR / "readability_check_exp15.md", "\n".join(lines))
    if failures:
        raise SystemExit(f"Exp15 readability failed. See {relpath(EXP15_REPORTS_DIR / 'exp15_readability_report.md')}")
    print("Exp15 readability PASS")


if __name__ == "__main__":
    main()
