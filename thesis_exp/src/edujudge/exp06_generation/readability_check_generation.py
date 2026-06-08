"""Readability and lightweight integrity checks for Exp6 generation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import py_compile
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp06_generation import (
    EXP06_GENERATION_SRC_DIR,
    EXP06_MINI_BATCH_OUTPUT_DIR,
    ensure_mini_batch_dirs,
)
from thesis_exp.src.edujudge.exp06_generation.mini_batch_common import mini_report_path, write_mini_table
from thesis_exp.src.edujudge.utils.io import relpath, write_text


FIELDS = ["check_name", "path", "status", "details"]
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_MARKDOWN_LINE = 300


def add(rows: list[dict[str, Any]], name: str, path: Path, status: str, details: str) -> None:
    rows.append({"check_name": name, "path": relpath(path), "status": status, "details": details})


def check_lf(path: Path, rows: list[dict[str, Any]]) -> None:
    data = path.read_bytes()
    if b"\r\n" in data or b"\r" in data:
        add(rows, "lf_endings", path, "FAIL", "CRLF or CR newline found")
    else:
        add(rows, "lf_endings", path, "PASS", "LF/no CR newline")


def check_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            parsed = list(csv.reader(handle))
        add(rows, "csv_readable", path, "PASS", f"rows={len(parsed)}")
    except Exception as exc:  # noqa: BLE001 - audit script reports exception class.
        add(rows, "csv_readable", path, "FAIL", f"{type(exc).__name__}: {exc}")


def check_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    count = 0
    line_no = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                json.loads(line)
                count += 1
        add(rows, "jsonl_loadable", path, "PASS", f"records={count}")
    except Exception as exc:  # noqa: BLE001 - audit script reports exception class.
        add(rows, "jsonl_loadable", path, "FAIL", f"line={line_no}; {type(exc).__name__}: {exc}")


def check_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    too_long = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if len(line) > MAX_MARKDOWN_LINE:
            too_long.append(line_no)
    if too_long:
        add(rows, "markdown_line_length", path, "WARN", f"lines_over_{MAX_MARKDOWN_LINE}={too_long[:10]}")
    else:
        add(rows, "markdown_line_length", path, "PASS", f"all lines <= {MAX_MARKDOWN_LINE}")


def check_python(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        py_compile.compile(str(path), doraise=True)
        add(rows, "python_py_compile", path, "PASS", "compiled")
    except py_compile.PyCompileError as exc:
        add(rows, "python_py_compile", path, "FAIL", str(exc).splitlines()[-1])


def run_readability() -> list[dict[str, Any]]:
    ensure_mini_batch_dirs()
    rows: list[dict[str, Any]] = []
    artifact_paths = [path for path in EXP06_MINI_BATCH_OUTPUT_DIR.rglob("*") if path.is_file()]
    for path in sorted(artifact_paths):
        size = path.stat().st_size
        add(rows, "file_size_under_10mb", path, "PASS" if size <= MAX_FILE_BYTES else "FAIL", f"bytes={size}")
        check_lf(path, rows)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            check_csv(path, rows)
        elif suffix == ".jsonl":
            check_jsonl(path, rows)
        elif suffix == ".md":
            check_markdown(path, rows)

    for path in sorted(EXP06_GENERATION_SRC_DIR.glob("*.py")):
        check_python(path, rows)

    write_mini_table("readability_check_generation.csv", rows, FIELDS)
    write_report(rows)
    return rows


def write_report(rows: list[dict[str, Any]]) -> None:
    failures = [row for row in rows if row["status"] == "FAIL"]
    warnings = [row for row in rows if row["status"] == "WARN"]
    status = "FAIL" if failures else "WARN" if warnings else "PASS"
    text = f"""# Exp6-3 Generation Readability Check

Overall status: **{status}**

- Checks: **{len(rows)}**
- Failures: **{len(failures)}**
- Warnings: **{len(warnings)}**

Outputs checked under `{relpath(EXP06_MINI_BATCH_OUTPUT_DIR)}` and Python files under
`{relpath(EXP06_GENERATION_SRC_DIR)}`.
"""
    write_text(mini_report_path("readability_check_generation.md"), text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    rows = run_readability()
    failures = sum(1 for row in rows if row["status"] == "FAIL")
    print(f"Wrote readability check rows={len(rows)} failures={failures}")


if __name__ == "__main__":
    main()
