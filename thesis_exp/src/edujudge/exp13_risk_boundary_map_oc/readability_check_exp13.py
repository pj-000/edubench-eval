"""Readability and safety checks for Exp13 outputs."""

from __future__ import annotations

import csv
import py_compile
import subprocess
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp13_risk_boundary_map_oc import (
    EXP13_CONFIG_DIR,
    EXP13_OUTPUT_DIR,
    EXP13_REPORTS_DIR,
    EXP13_TABLES_DIR,
    ensure_exp13_dirs,
)
from thesis_exp.src.edujudge.utils.io import relpath, write_csv, write_text


REQUIRED_CONFIGS = [
    "exp13_point_pair_proj_l2h_lam0p10.yaml",
    "exp13_point_pair_proj_l2h_lam0p20.yaml",
    "exp13_point_pair_proj_l2h_lam0p40.yaml",
    "exp13_point_pair_proj_l2h_label2_w1p5_lam0p20.yaml",
    "exp13_point_pair_proj_l2h_lam0p20_no_mono.yaml",
    "exp13_score_proj_l2h_lam0p20.yaml",
    "exp13_map_oc_full_l2h_lam0p20.yaml",
    "exp13_score_proj_t3_brier_lam0p05.yaml",
]
REQUIRED_TABLES = [
    "exp13_dev_selection_ranking.csv",
    "exp13_formal_summary.csv",
    "exp13_run_config_summary.csv",
]
REQUIRED_REPORTS = [
    "exp13_risk_boundary_map_oc_report.md",
    "exp13_preflight_report.md",
]
SOURCE_PATHS = [
    Path("thesis_exp/src/edujudge/exp09_pairwise_ordinal/train_qdpr2_anchored_pairwise.py"),
    Path("thesis_exp/src/edujudge/exp09_pairwise_ordinal/losses.py"),
    Path("thesis_exp/src/edujudge/exp13_risk_boundary_map_oc/__init__.py"),
    Path("thesis_exp/src/edujudge/exp13_risk_boundary_map_oc/risk_boundary_losses.py"),
    Path("thesis_exp/src/edujudge/exp13_risk_boundary_map_oc/preflight_exp13.py"),
    Path("thesis_exp/src/edujudge/exp13_risk_boundary_map_oc/smoke_check_exp13.py"),
    Path("thesis_exp/src/edujudge/exp13_risk_boundary_map_oc/collect_exp13_results.py"),
    Path("thesis_exp/src/edujudge/exp13_risk_boundary_map_oc/validate_formal_lock.py"),
    Path("thesis_exp/src/edujudge/exp13_risk_boundary_map_oc/readability_check_exp13.py"),
]
SCRIPT_PATHS = [
    Path("thesis_exp/scripts/run_exp13_risk_boundary_map_oc.sh"),
    Path("thesis_exp/scripts/run_exp13_risk_boundary_map_oc_formal_locked.sh"),
    Path("thesis_exp/scripts/run_exp13_risk_boundary_map_oc_smoke.sh"),
    Path("thesis_exp/scripts/sync_exp13_risk_boundary_map_oc_to_server.sh"),
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
    ensure_exp13_dirs()
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
    config_dir = Path("thesis_exp/configs/exp13_risk_boundary_map_oc")
    for name in REQUIRED_CONFIGS:
        path = config_dir / name
        add(rows, f"required config {name}", path.exists(), relpath(path))
    for name in REQUIRED_TABLES:
        path = EXP13_TABLES_DIR / name
        add(rows, f"required table {name}", path.exists(), relpath(path))
        if path.exists():
            try:
                _csv_rows(path)
                add(rows, f"CSV readable {name}", True)
            except Exception as exc:
                add(rows, f"CSV readable {name}", False, f"{type(exc).__name__}: {exc}")
    for name in REQUIRED_REPORTS:
        path = EXP13_REPORTS_DIR / name
        add(rows, f"required report {name}", path.exists(), relpath(path))

    ranking_path = EXP13_TABLES_DIR / "exp13_dev_selection_ranking.csv"
    if ranking_path.exists():
        headers = _csv_headers(ranking_path)
        add(rows, "dev ranking has p_gt_3 risk fields", "dev_p_gt_3_low_mean" in headers and "dev_p_gt_3_label2_mean" in headers)
        add(rows, "dev ranking contains no test columns", not any(header.startswith("test_") for header in headers))
        ranking_rows = _csv_rows(ranking_path)
        add(
            rows,
            "uses_test_for_selection is always false in dev ranking",
            all(str(row.get("uses_test_for_selection", "")).lower() == "false" for row in ranking_rows),
        )

    selected_path = EXP13_TABLES_DIR / "exp13_dev_selected_configs.csv"
    if selected_path.exists():
        selected_headers = _csv_headers(selected_path)
        add(rows, "dev selected configs contain no test columns", not any(header.startswith("test_") for header in selected_headers))

    report_path = EXP13_REPORTS_DIR / "exp13_risk_boundary_map_oc_report.md"
    if report_path.exists():
        text = report_path.read_text(encoding="utf-8").lower()
        add(rows, "report says scout ranking is dev-only", "dev-only" in text and "test metrics are not used" in text)
        add(rows, "report mentions low-to-high", "low-to-high" in text)
        add(rows, "report avoids overclaim", not any(word in text for word in ["proves", "solves", "guarantees"]))

    formal_lock_path = EXP13_CONFIG_DIR / "exp13_formal_locked.yaml"
    formal_lock_report = EXP13_REPORTS_DIR / "exp13_formal_lock_report.md"
    if formal_lock_path.exists():
        add(rows, "formal lock report exists", formal_lock_report.exists(), relpath(formal_lock_report))
        if formal_lock_report.exists():
            formal_text = formal_lock_report.read_text(encoding="utf-8").lower()
            add(
                rows,
                "formal lock report avoids overclaim",
                not any(word in formal_text for word in ["proves", "solves", "guarantees"]),
            )
            add(
                rows,
                "formal lock report states test is final evaluation only",
                "test metrics are only for final held-out evaluation" in formal_text
                and "no test labels are used for tuning" in formal_text,
            )

    formal_summary_path = EXP13_TABLES_DIR / "exp13_formal_summary.csv"
    if formal_summary_path.exists() and formal_lock_path.exists():
        formal_headers = _csv_headers(formal_summary_path)
        if any(header.startswith("test_") for header in formal_headers):
            formal_text = formal_lock_report.read_text(encoding="utf-8").lower() if formal_lock_report.exists() else ""
            add(
                rows,
                "formal summary test metrics are documented as final evaluation",
                "test metrics are only for final held-out evaluation" in formal_text,
            )

    heavy = []
    for path in EXP13_OUTPUT_DIR.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in HEAVY_SUFFIXES or any(part in RAW_NAMES for part in path.parts):
            heavy.append(relpath(path))
    add(rows, "no heavy/raw artifacts under tracked Exp13 outputs", not heavy, heavy)

    write_csv(EXP13_TABLES_DIR / "readability_check_exp13.csv", rows)
    failures = [row for row in rows if row["status"] != "PASS"]
    lines = [
        "# Exp13 Readability Check",
        "",
        f"Status: `{'PASS' if not failures else 'FAIL'}`",
        "",
        "| check | status | details |",
        "| --- | --- | --- |",
    ]
    lines.extend(f"| {row['check_name']} | {row['status']} | {row.get('details', '')} |" for row in rows)
    write_text(EXP13_REPORTS_DIR / "exp13_readability_report.md", "\n".join(lines))
    write_text(EXP13_OUTPUT_DIR / "readability_check_exp13.md", "\n".join(lines))
    if failures:
        raise SystemExit(f"Exp13 readability failed. See {relpath(EXP13_REPORTS_DIR / 'exp13_readability_report.md')}")
    print("Exp13 readability PASS")


if __name__ == "__main__":
    main()
