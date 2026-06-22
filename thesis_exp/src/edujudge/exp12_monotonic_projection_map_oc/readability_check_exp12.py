"""Readability and safety checks for Exp12 outputs."""

from __future__ import annotations

import csv
import py_compile
import subprocess
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp12_monotonic_projection_map_oc import EXP12_OUTPUT_DIR, EXP12_REPORTS_DIR, EXP12_TABLES_DIR, ensure_exp12_dirs
from thesis_exp.src.edujudge.utils.io import relpath, write_csv, write_text


REQUIRED_TABLES = [
    "exp12a_decode_projection_metrics.csv",
    "exp12a_raw_vs_projected_selected.csv",
    "exp12a_projection_effect_by_seed_epoch.csv",
    "exp12a_low_score_projection_distribution.csv",
    "exp12a_monotonic_by_threshold.csv",
    "exp12b_train_metrics_dev.csv",
    "exp12b_train_metrics_test_diagnostic.csv",
    "exp12b_selected_checkpoint_test_metrics.csv",
    "exp12b_ablation_summary.csv",
    "exp12b_raw_vs_projected_metrics.csv",
    "exp12b_low_to_high_by_label.csv",
    "exp12b_low_score_prediction_distribution.csv",
    "exp12b_monotonic_by_threshold.csv",
    "exp12b_projection_delta_summary.csv",
    "exp12b_vs_qdb1_qdpr2_exp11.csv",
    "exp12_run_config_summary.csv",
    "exp12_checkpoint_inventory.csv",
]
REQUIRED_REPORTS = [
    "exp12_monotonic_projection_map_oc_report.md",
    "exp12_monotonic_projection_map_oc_review_package.md",
    "exp12_preflight_report.md",
]
SOURCE_PATHS = [
    Path("thesis_exp/src/edujudge/exp09_pairwise_ordinal/train_qdpr2_anchored_pairwise.py"),
    Path("thesis_exp/src/edujudge/exp09_pairwise_ordinal/losses.py"),
    Path("thesis_exp/src/edujudge/exp12_monotonic_projection_map_oc/__init__.py"),
    Path("thesis_exp/src/edujudge/exp12_monotonic_projection_map_oc/monotone_projection.py"),
    Path("thesis_exp/src/edujudge/exp12_monotonic_projection_map_oc/test_monotone_projection.py"),
    Path("thesis_exp/src/edujudge/exp12_monotonic_projection_map_oc/collect_exp12_results.py"),
    Path("thesis_exp/src/edujudge/exp12_monotonic_projection_map_oc/preflight_exp12.py"),
    Path("thesis_exp/src/edujudge/exp12_monotonic_projection_map_oc/readability_check_exp12.py"),
]
SCRIPT_PATHS = [
    Path("thesis_exp/scripts/run_exp12_monotonic_projection_map_oc.sh"),
    Path("thesis_exp/scripts/run_exp12_monotonic_projection_map_oc_smoke.sh"),
    Path("thesis_exp/scripts/sync_exp12_monotonic_projection_map_oc_to_server.sh"),
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
    ensure_exp12_dirs()
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
        path = EXP12_TABLES_DIR / name
        add(rows, f"required table {name}", path.exists(), relpath(path))
        if path.exists():
            try:
                _csv_rows(path)
                add(rows, f"CSV readable {name}", True)
            except Exception as exc:
                add(rows, f"CSV readable {name}", False, f"{type(exc).__name__}: {exc}")
    for name in REQUIRED_REPORTS:
        path = EXP12_REPORTS_DIR / name
        add(rows, f"required report {name}", path.exists(), relpath(path))

    metric_path = EXP12_TABLES_DIR / "exp12a_raw_vs_projected_selected.csv"
    if metric_path.exists():
        headers = _csv_headers(metric_path)
        add(rows, "projected monotonic violation fields exist", "monotonic_violation" in headers and "p1_lt_p2" in headers)
        add(rows, "raw/projected decode_mode field exists", "decode_mode" in headers)
        add(rows, "low-to-high has ratio and count", "low_to_high" in headers and "low_to_high_count" in headers)
        add(rows, "projection delta columns exist", "mean_projection_l2_delta" in headers and "mean_projection_linf_delta" in headers)

    selected_path = EXP12_TABLES_DIR / "exp12b_selected_checkpoint_test_metrics.csv"
    if selected_path.exists():
        selected_rows = _csv_rows(selected_path)
        add(
            rows,
            "uses_test_for_selection is always false",
            all(str(row.get("uses_test_for_selection", "")).lower() == "false" for row in selected_rows),
        )

    report_path = EXP12_REPORTS_DIR / "exp12_monotonic_projection_map_oc_report.md"
    if report_path.exists():
        text = report_path.read_text(encoding="utf-8").lower()
        add(rows, "report says test metrics are not used for selection", "not used for checkpoint selection" in text)
        add(rows, "report mentions low-to-high", "low-to-high" in text)
        add(rows, "report has clear Exp12A status", "exp12a status:" in text)
        add(rows, "report has clear Exp12B status", "exp12b status:" in text)
        add(rows, "no fake training result when missing", "no_completed_training_runs" in text or "completed" in text)
        forbidden = ["proves", "solves", "fully eliminates", "guarantees", "dpo", "rlhf", "reinforcement learning"]
        add(rows, "report avoids forbidden overclaim/RL language", not any(word in text for word in forbidden), forbidden)

    heavy = []
    for path in EXP12_OUTPUT_DIR.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in HEAVY_SUFFIXES or any(part in RAW_NAMES for part in path.parts):
            heavy.append(relpath(path))
    add(rows, "no heavy/raw artifacts under tracked Exp12 outputs", not heavy, heavy)

    write_csv(EXP12_TABLES_DIR / "readability_check_exp12.csv", rows)
    failures = [row for row in rows if row["status"] != "PASS"]
    lines = [
        "# Exp12 Readability Check",
        "",
        f"Status: `{'PASS' if not failures else 'FAIL'}`",
        "",
        "| check | status | details |",
        "| --- | --- | --- |",
    ]
    lines.extend(f"| {row['check_name']} | {row['status']} | {row.get('details', '')} |" for row in rows)
    write_text(EXP12_OUTPUT_DIR / "readability_check_exp12.md", "\n".join(lines))
    if failures:
        raise SystemExit(f"Exp12 readability failed. See {relpath(EXP12_OUTPUT_DIR / 'readability_check_exp12.md')}")
    print("Exp12 readability PASS")


if __name__ == "__main__":
    main()
