"""Readability and safety checks for Exp14 outputs."""

from __future__ import annotations

import csv
import py_compile
import subprocess
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp14_logit_margin_tail_risk_oc import (
    EXP14_CONFIG_DIR,
    EXP14_OUTPUT_DIR,
    EXP14_REPORTS_DIR,
    EXP14_TABLES_DIR,
    ensure_exp14_dirs,
)
from thesis_exp.src.edujudge.utils.io import relpath, write_csv, write_text


REQUIRED_CONFIGS = [
    "exp14_score_logit_margin_lam0p01_alllow.yaml",
    "exp14_score_logit_margin_lam0p02_alllow.yaml",
    "exp14_score_logit_margin_lam0p05_alllow.yaml",
    "exp14_score_tail_logit_margin_lam0p02_top0p50.yaml",
    "exp14_score_tail_logit_margin_lam0p05_top0p50.yaml",
    "exp14_score_tail_logit_margin_lam0p02_top0p25.yaml",
    "exp14_point_pair_tail_logit_margin_lam0p02_top0p50.yaml",
]
REQUIRED_TABLES = [
    "exp14_dev_selection_ranking.csv",
    "exp14_dev_selected_configs.csv",
    "exp14_run_config_summary.csv",
    "exp14_preflight.csv",
]
REQUIRED_REPORTS = [
    "exp14_logit_margin_tail_risk_oc_report.md",
    "exp14_preflight_report.md",
]
SOURCE_PATHS = [
    Path("thesis_exp/src/edujudge/exp09_pairwise_ordinal/train_qdpr2_anchored_pairwise.py"),
    Path("thesis_exp/src/edujudge/exp09_pairwise_ordinal/losses.py"),
    Path("thesis_exp/src/edujudge/exp14_logit_margin_tail_risk_oc/__init__.py"),
    Path("thesis_exp/src/edujudge/exp14_logit_margin_tail_risk_oc/logit_margin_losses.py"),
    Path("thesis_exp/src/edujudge/exp14_logit_margin_tail_risk_oc/preflight_exp14.py"),
    Path("thesis_exp/src/edujudge/exp14_logit_margin_tail_risk_oc/smoke_check_exp14.py"),
    Path("thesis_exp/src/edujudge/exp14_logit_margin_tail_risk_oc/collect_exp14_results.py"),
    Path("thesis_exp/src/edujudge/exp14_logit_margin_tail_risk_oc/readability_check_exp14.py"),
]
SCRIPT_PATHS = [
    Path("thesis_exp/scripts/run_exp14_logit_margin_tail_risk_oc.sh"),
    Path("thesis_exp/scripts/run_exp14_logit_margin_tail_risk_oc_smoke.sh"),
    Path("thesis_exp/scripts/sync_exp14_logit_margin_tail_risk_oc_to_server.sh"),
]
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
    ensure_exp14_dirs()
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
        path = EXP14_CONFIG_DIR / name
        add(rows, f"required config {name}", path.exists(), relpath(path))
    for name in REQUIRED_TABLES:
        path = EXP14_TABLES_DIR / name
        add(rows, f"required table {name}", path.exists(), relpath(path))
        if path.exists():
            try:
                _csv_rows(path)
                add(rows, f"CSV readable {name}", True)
            except Exception as exc:
                add(rows, f"CSV readable {name}", False, f"{type(exc).__name__}: {exc}")
    for name in REQUIRED_REPORTS:
        path = EXP14_REPORTS_DIR / name
        add(rows, f"required report {name}", path.exists(), relpath(path))

    ranking_path = EXP14_TABLES_DIR / "exp14_dev_selection_ranking.csv"
    if ranking_path.exists():
        headers = _csv_headers(ranking_path)
        add(rows, "dev ranking contains no test columns", not any(header.startswith("test_") for header in headers))
        add(rows, "dev ranking has z3 fields", "dev_low_z3_q95" in headers and "dev_label2_z3_mean" in headers)
        ranking_rows = _csv_rows(ranking_path)
        add(
            rows,
            "uses_test_for_selection is always false in dev ranking",
            all(str(row.get("uses_test_for_selection", "")).lower() == "false" for row in ranking_rows),
        )

    report_path = EXP14_REPORTS_DIR / "exp14_logit_margin_tail_risk_oc_report.md"
    if report_path.exists():
        text = report_path.read_text(encoding="utf-8").lower()
        add(rows, "report says scout ranking is dev-only", "dev-only" in text and "test metrics are not used" in text)
        add(rows, "report includes formal recommendation status", "formal recommendation" in text)
        add(rows, "report avoids overclaim", not any(word in text for word in ["proves", "solves", "guarantees"]))

    heavy = []
    for path in EXP14_OUTPUT_DIR.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in HEAVY_SUFFIXES or any(part in RAW_NAMES for part in path.parts):
            heavy.append(relpath(path))
    add(rows, "no heavy/raw artifacts under tracked Exp14 outputs", not heavy, heavy)

    write_csv(EXP14_TABLES_DIR / "readability_check_exp14.csv", rows)
    failures = [row for row in rows if row["status"] != "PASS"]
    lines = [
        "# Exp14 Readability Check",
        "",
        f"Status: `{'PASS' if not failures else 'FAIL'}`",
        "",
        "| check | status | details |",
        "| --- | --- | --- |",
    ]
    lines.extend(f"| {row['check_name']} | {row['status']} | {row.get('details', '')} |" for row in rows)
    write_text(EXP14_REPORTS_DIR / "exp14_readability_report.md", "\n".join(lines))
    write_text(EXP14_OUTPUT_DIR / "readability_check_exp14.md", "\n".join(lines))
    if failures:
        raise SystemExit(f"Exp14 readability failed. See {relpath(EXP14_REPORTS_DIR / 'exp14_readability_report.md')}")
    print("Exp14 readability PASS")


if __name__ == "__main__":
    main()
