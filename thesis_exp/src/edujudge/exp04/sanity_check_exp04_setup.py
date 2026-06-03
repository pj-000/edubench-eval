"""Preflight setup checks for Exp4 before any smoke or formal training."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp04 import (
    A4_FIXED_DATASET_DIR,
    EXPECTED_SPLIT_ROWS,
    EXP03_A4_DATASET_DIR,
    EXP03_A4_RUN_DIR,
    EXP04_OUTPUT_DIR,
    EXP04_TABLES_DIR,
    ensure_exp04_dirs,
)
from thesis_exp.src.edujudge.exp04.build_exp04_dataset import build_exp04_dataset
from thesis_exp.src.edujudge.exp04.train_objective import ordinal_targets
from thesis_exp.src.edujudge.utils.io import REPO_ROOT, read_jsonl, relpath, write_csv, write_text


SCRIPT_PATHS = [
    REPO_ROOT / "run_exp04_train_objectives.sh",
    REPO_ROOT / "thesis_exp" / "scripts" / "run_exp04_train_objectives.sh",
]
EXP04_SRC_DIR = REPO_ROOT / "thesis_exp" / "src" / "edujudge" / "exp04"
EXP3_A4_REQUIRED = [
    "predictions/predictions_dev.jsonl",
    "predictions/predictions_test.jsonl",
    "tables/metrics_summary.csv",
    "tables/per_bin_metrics.csv",
    "tables/low_score_metrics.csv",
    "tables/high_score_metrics.csv",
    "tables/metric_level_metrics.csv",
    "tables/scenario_level_metrics.csv",
    "arrays/dev_test_arrays.npz",
]


def add(rows: list[dict[str, Any]], check: str, status: str, observed: Any, expected: Any, notes: str = "") -> None:
    rows.append({"check": check, "status": status, "observed": observed, "expected": expected, "notes": notes})


def markdown_table(rows: list[dict[str, Any]]) -> str:
    columns = ["check", "status", "observed", "expected", "notes"]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        cells = []
        for col in columns:
            value = str(row.get(col, "")).replace("|", "\\|")
            if len(value) > 110:
                value = value[:107] + "..."
            cells.append(value)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def command_status(args: list[str]) -> tuple[str, str]:
    result = subprocess.run(args, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    output = " ".join(result.stdout.split())
    return ("PASS" if result.returncode == 0 else "FAIL", output[-700:] if output else "ok")


def check_exp3_a4(rows: list[dict[str, Any]]) -> None:
    add(rows, "Exp3 A4 run exists", "PASS" if EXP03_A4_RUN_DIR.exists() else "FAIL", relpath(EXP03_A4_RUN_DIR), "exists")
    add(rows, "Exp3 A4 dataset exists", "PASS" if EXP03_A4_DATASET_DIR.exists() else "FAIL", relpath(EXP03_A4_DATASET_DIR), "exists")
    missing = [rel for rel in EXP3_A4_REQUIRED if not (EXP03_A4_RUN_DIR / rel).exists()]
    add(rows, "O1 can reuse Exp3 A4 artifacts", "PASS" if not missing else "FAIL", f"missing={missing}", "all required files present")


def check_fixed_dataset(rows: list[dict[str, Any]]) -> None:
    try:
        build_rows = build_exp04_dataset(force=True)
        build_status = "PASS" if all(row.get("status") == "PASS" for row in build_rows) else "FAIL"
        add(rows, "Exp4 fixed A4 dataset can be built", build_status, [row.get("status") for row in build_rows], "all PASS")
    except Exception as exc:
        add(rows, "Exp4 fixed A4 dataset can be built", "FAIL", f"{type(exc).__name__}: {exc}", "build succeeds")
        return

    split_ids: dict[str, set[str]] = {}
    for split, expected in EXPECTED_SPLIT_ROWS.items():
        path = A4_FIXED_DATASET_DIR / f"{split}.jsonl"
        data = read_jsonl(path)
        add(rows, f"{split} row count", "PASS" if len(data) == expected else "FAIL", len(data), expected)
        source_path_ok = "synthetic" not in str(path).lower() and "sample" not in str(path).lower()
        add(rows, f"{split} no synthetic/sample path", "PASS" if source_path_ok else "FAIL", relpath(path), "fixed A4 dataset path")
        split_ids[split] = {str(row.get("record_id") or row.get("id")) for row in data}
        missing_human_mean = sum(1 for row in data if row.get("human_mean_5") in (None, ""))
        add(rows, f"{split} O2 human_mean_5 target", "PASS" if missing_human_mean == 0 else "FAIL", missing_human_mean, 0)
        ordinal = [ordinal_targets(int(row["label_5"])) for row in data[:5]]
        shape_ok = all(len(vector) == 4 for vector in ordinal)
        add(rows, f"{split} O3 ordinal target shape", "PASS" if shape_ok else "FAIL", "5x4 sample" if shape_ok else ordinal, "[n,4]")

    overlaps = {
        "train_dev": len(split_ids.get("train", set()) & split_ids.get("dev", set())),
        "train_test": len(split_ids.get("train", set()) & split_ids.get("test", set())),
        "dev_test": len(split_ids.get("dev", set()) & split_ids.get("test", set())),
    }
    add(rows, "No test leakage by record_id", "PASS" if all(value == 0 for value in overlaps.values()) else "FAIL", overlaps, "all overlaps 0")


def check_scripts_and_modules(rows: list[dict[str, Any]]) -> None:
    for path in SCRIPT_PATHS:
        status, output = command_status(["bash", "-n", str(path.relative_to(REPO_ROOT))])
        add(rows, f"bash -n {relpath(path)}", status, output, "ok")
    py_files = sorted(EXP04_SRC_DIR.glob("*.py"))
    status, output = command_status([sys.executable, "-m", "py_compile", *[str(path.relative_to(REPO_ROOT)) for path in py_files]])
    add(rows, "exp04 modules py_compile", status, output, "ok")


def check_no_tracked_checkpoints(rows: list[dict[str, Any]]) -> None:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "thesis_exp/artifacts",
            "*.safetensors",
            "*.pt",
            "*.pth",
            "*.bin",
            "*/hf_cache/*",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    tracked = [line for line in result.stdout.splitlines() if line.strip()]
    status = "PASS" if result.returncode == 0 and not tracked else "FAIL"
    add(rows, "No checkpoint files tracked", status, tracked, "[]")


def run_setup_sanity() -> list[dict[str, Any]]:
    ensure_exp04_dirs()
    rows: list[dict[str, Any]] = []
    check_exp3_a4(rows)
    check_fixed_dataset(rows)
    check_scripts_and_modules(rows)
    check_no_tracked_checkpoints(rows)
    write_csv(EXP04_TABLES_DIR / "sanity_check_exp04_setup.csv", rows)
    overall = "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL"
    write_text(
        EXP04_OUTPUT_DIR / "sanity_check_exp04_setup.md",
        f"""# Exp4 Setup Sanity Check

Overall status: **{overall}**

{markdown_table(rows)}
""",
    )
    try:
        from thesis_exp.src.edujudge.exp04.write_exp04_report import write_review_package

        write_review_package()
    except Exception as exc:
        add(rows, "review package refresh", "FAIL", f"{type(exc).__name__}: {exc}", "refresh ok")
        write_csv(EXP04_TABLES_DIR / "sanity_check_exp04_setup.csv", rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Exp4 setup before smoke or formal training.")
    return parser.parse_args()


def main() -> None:
    parse_args()
    rows = run_setup_sanity()
    statuses = sorted({row["status"] for row in rows})
    print(f"Exp4 setup sanity statuses: {', '.join(statuses)}")
    print(f"Output: {relpath(EXP04_OUTPUT_DIR / 'sanity_check_exp04_setup.md')}")
    if any(row["status"] != "PASS" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
