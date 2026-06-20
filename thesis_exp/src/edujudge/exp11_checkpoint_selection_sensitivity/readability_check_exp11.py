"""Readability and safety checks for Exp11 outputs."""

from __future__ import annotations

import csv
import py_compile
import subprocess
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp11_checkpoint_selection_sensitivity import (
    EXP11_OUTPUT_DIR,
    EXP11_REPORTS_DIR,
    EXP11_TABLES_DIR,
    SELECTION_RULES,
    ensure_exp11_dirs,
)
from thesis_exp.src.edujudge.utils.io import relpath, write_csv, write_text


REQUIRED_TABLES = [
    "exp11_epoch_metrics_dev.csv",
    "exp11_epoch_metrics_test_diagnostic.csv",
    "exp11_soft_risk_metrics_dev.csv",
    "exp11_soft_risk_metrics_test_diagnostic.csv",
    "exp11_selection_rule_summary.csv",
    "exp11_selected_checkpoint_test_metrics.csv",
    "exp11_selection_rule_vs_dev_mae_baseline.csv",
    "exp11_run_config_summary.csv",
    "exp11_checkpoint_inventory.csv",
    "exp11_low_score_by_epoch_dev.csv",
    "exp11_low_score_by_epoch_test_diagnostic.csv",
]
REQUIRED_REPORTS = [
    "exp11_checkpoint_selection_sensitivity_report.md",
    "exp11_checkpoint_selection_sensitivity_review_package.md",
]
SOURCE_PATHS = [
    Path("thesis_exp/src/edujudge/exp09_pairwise_ordinal/train_qdpr2_anchored_pairwise.py"),
    Path("thesis_exp/src/edujudge/exp11_checkpoint_selection_sensitivity/__init__.py"),
    Path("thesis_exp/src/edujudge/exp11_checkpoint_selection_sensitivity/collect_exp11_results.py"),
    Path("thesis_exp/src/edujudge/exp11_checkpoint_selection_sensitivity/preflight_exp11.py"),
    Path("thesis_exp/src/edujudge/exp11_checkpoint_selection_sensitivity/readability_check_exp11.py"),
]
SCRIPT_PATHS = [
    Path("thesis_exp/scripts/run_exp11_checkpoint_selection_sensitivity.sh"),
    Path("thesis_exp/scripts/run_exp11_checkpoint_selection_sensitivity_smoke.sh"),
]
HEAVY_SUFFIXES = {".pt", ".pth", ".bin", ".ckpt", ".safetensors", ".npy", ".npz"}
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
    ensure_exp11_dirs()
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
    for name in REQUIRED_TABLES:
        path = EXP11_TABLES_DIR / name
        add(rows, f"required table {name}", path.exists(), relpath(path))
        if path.exists():
            try:
                _csv_rows(path)
                add(rows, f"CSV readable {name}", True)
            except Exception as exc:
                add(rows, f"CSV readable {name}", False, f"{type(exc).__name__}: {exc}")
    for name in REQUIRED_REPORTS:
        path = EXP11_REPORTS_DIR / name
        add(rows, f"required report {name}", path.exists(), relpath(path))
    selection_path = EXP11_TABLES_DIR / "exp11_selection_rule_summary.csv"
    selected_test_path = EXP11_TABLES_DIR / "exp11_selected_checkpoint_test_metrics.csv"
    if selection_path.exists():
        selection_rows = _csv_rows(selection_path)
        add(
            rows,
            "uses_test_for_selection is always false",
            all(str(row.get("uses_test_for_selection", "")).lower() == "false" for row in selection_rows),
        )
        rules = {row.get("selection_rule") for row in selection_rows}
        add(rows, "selection rules complete if any seed completed", not selection_rows or set(SELECTION_RULES).issubset(rules), sorted(rules))
    if selected_test_path.exists():
        selected_rows = _csv_rows(selected_test_path)
        rules = {row.get("selection_rule") for row in selected_rows}
        add(rows, "selected checkpoint test metrics include each rule if any seed completed", not selected_rows or set(SELECTION_RULES).issubset(rules), sorted(rules))
    add(
        rows,
        "test diagnostic table filename is explicit",
        (EXP11_TABLES_DIR / "exp11_epoch_metrics_test_diagnostic.csv").exists()
        and (EXP11_TABLES_DIR / "exp11_soft_risk_metrics_test_diagnostic.csv").exists(),
    )
    report_path = EXP11_REPORTS_DIR / "exp11_checkpoint_selection_sensitivity_report.md"
    if report_path.exists():
        text = report_path.read_text(encoding="utf-8").lower()
        add(rows, "report says test metrics are not used for selection", "test metrics are post-hoc diagnostic only" in text)
        add(rows, "report mentions low-to-high", "low-to-high" in text)
    metric_path = EXP11_TABLES_DIR / "exp11_epoch_metrics_dev.csv"
    if metric_path.exists():
        headers = _csv_headers(metric_path)
        add(rows, "low-to-high has ratio and count columns", "low_to_high" in headers and "low_to_high_count" in headers)
    heavy = []
    for path in EXP11_OUTPUT_DIR.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in HEAVY_SUFFIXES or any(part in RAW_NAMES for part in path.parts):
            heavy.append(relpath(path))
    add(rows, "no heavy/raw artifacts under tracked Exp11 outputs", not heavy, heavy)
    write_csv(EXP11_TABLES_DIR / "readability_check_exp11.csv", rows)
    failures = [row for row in rows if row["status"] != "PASS"]
    lines = [
        "# Exp11 Readability Check",
        "",
        f"Status: `{'PASS' if not failures else 'FAIL'}`",
        "",
        "| check | status | details |",
        "| --- | --- | --- |",
    ]
    lines.extend(f"| {row['check_name']} | {row['status']} | {row.get('details', '')} |" for row in rows)
    write_text(EXP11_OUTPUT_DIR / "readability_check_exp11.md", "\n".join(lines))
    if failures:
        raise SystemExit(f"Exp11 readability failed. See {relpath(EXP11_OUTPUT_DIR / 'readability_check_exp11.md')}")
    print("Exp11 readability PASS")


if __name__ == "__main__":
    main()
