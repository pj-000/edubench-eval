"""Readability checks for Exp10 QD-PR2 module ablation files."""

from __future__ import annotations

import csv
import py_compile
import subprocess
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp10_qdpr2_module_ablation import (
    EXP10_CONFIG_DIR,
    EXP10_OUTPUT_DIR,
    EXP10_TABLES_DIR,
    ensure_exp10_dirs,
)
from thesis_exp.src.edujudge.utils.io import relpath, write_csv, write_text


SCRIPT_PATHS = [
    Path("thesis_exp/scripts/run_exp10_qdpr2_module_ablation.sh"),
    Path("thesis_exp/scripts/run_exp10_qdpr2_module_ablation_smoke.sh"),
    Path("thesis_exp/scripts/sync_exp10_qdpr2_module_ablation_to_server.sh"),
]
SOURCE_PATHS = [
    Path("thesis_exp/src/edujudge/exp09_pairwise_ordinal/losses.py"),
    Path("thesis_exp/src/edujudge/exp09_pairwise_ordinal/train_qdpr2_anchored_pairwise.py"),
    Path("thesis_exp/src/edujudge/exp10_qdpr2_module_ablation/__init__.py"),
    Path("thesis_exp/src/edujudge/exp10_qdpr2_module_ablation/collect_exp10_results.py"),
    Path("thesis_exp/src/edujudge/exp10_qdpr2_module_ablation/preflight_exp10_ablation_matrix.py"),
    Path("thesis_exp/src/edujudge/exp10_qdpr2_module_ablation/sanity_check_exp10_setup.py"),
    Path("thesis_exp/src/edujudge/exp10_qdpr2_module_ablation/readability_check_exp10.py"),
    Path("thesis_exp/src/edujudge/exp10_qdpr2_module_ablation/smoke_check_exp10.py"),
]
TEXT_SUFFIXES = {".py", ".sh", ".md", ".csv", ".yaml", ".yml"}
COLLAPSE_SUFFIXES = {".py", ".sh", ".md", ".csv"}
FIRST_LINE_SUFFIXES = {".py", ".sh", ".md"}
SKIP_OUTPUT_DIRS = {"runs", "logs"}


def add(rows: list[dict[str, Any]], check_name: str, path: Path, passed: bool, details: Any = "") -> None:
    rows.append(
        {
            "check_name": check_name,
            "path": relpath(path),
            "status": "PASS" if passed else "FAIL",
            "details": details,
        }
    )


def candidate_files() -> list[Path]:
    files = [path for path in SOURCE_PATHS + SCRIPT_PATHS if path.exists()]
    if EXP10_CONFIG_DIR.exists():
        files.extend(sorted(path for path in EXP10_CONFIG_DIR.glob("*.yaml") if path.is_file()))
    if EXP10_OUTPUT_DIR.exists():
        for path in sorted(EXP10_OUTPUT_DIR.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(EXP10_OUTPUT_DIR)
            if relative.parts and relative.parts[0] in SKIP_OUTPUT_DIRS:
                continue
            files.append(path)
    unique = []
    seen = set()
    for path in files:
        if path.suffix.lower() in TEXT_SUFFIXES and path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def check_py_compile(rows: list[dict[str, Any]]) -> None:
    for path in SOURCE_PATHS:
        try:
            py_compile.compile(str(path), doraise=True)
            add(rows, "python py_compile", path, True)
        except py_compile.PyCompileError as exc:
            add(rows, "python py_compile", path, False, str(exc))


def check_bash(rows: list[dict[str, Any]]) -> None:
    for path in SCRIPT_PATHS:
        result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        add(rows, "shell bash -n", path, result.returncode == 0, (result.stderr or result.stdout).strip())


def check_shell_header(rows: list[dict[str, Any]], path: Path) -> None:
    if path.suffix.lower() != ".sh":
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    first = lines[0] if len(lines) >= 1 else ""
    second = lines[1] if len(lines) >= 2 else ""
    add(rows, "shell first line is bash shebang", path, first == "#!/usr/bin/env bash", first)
    add(rows, "shell second line is strict mode", path, second == "set -euo pipefail", second)


def check_csv(rows: list[dict[str, Any]], path: Path) -> None:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            list(csv.DictReader(handle))
        add(rows, "CSV readable", path, True)
    except Exception as exc:
        add(rows, "CSV readable", path, False, f"{type(exc).__name__}: {exc}")


def check_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    prose = [line for line in lines if not line.lstrip().startswith("|")]
    max_len = max((len(line) for line in prose), default=0)
    add(rows, "Markdown max prose line length", path, max_len < 300, max_len)


def check_line_endings(rows: list[dict[str, Any]], path: Path) -> None:
    data = path.read_bytes()
    has_crlf = b"\r\n" in data
    has_cr_only = b"\r" in data.replace(b"\r\n", b"")
    add(rows, "no CRLF line endings", path, not has_crlf)
    add(rows, "no CR-only line endings", path, not has_cr_only)
    add(rows, "LF line endings", path, not has_crlf and not has_cr_only)
    if path.suffix.lower() in FIRST_LINE_SUFFIXES and data:
        first_line = data.split(b"\n", 1)[0].rstrip(b"\r")
        add(rows, "first line <= 1000 chars", path, len(first_line) <= 1000, len(first_line))
    if path.suffix.lower() in COLLAPSE_SUFFIXES and data:
        line_count = len(data.splitlines())
        small_header_only_csv = path.suffix.lower() == ".csv" and line_count == 1 and len(data) <= 1000
        add(rows, "not collapsed-line file", path, line_count > 1 or small_header_only_csv, line_count)
        if path.suffix.lower() in {".md", ".csv"}:
            add(rows, "not large one-line md/csv", path, not (line_count <= 1 and len(data) > 1000), f"lines={line_count}; bytes={len(data)}")


def main() -> None:
    ensure_exp10_dirs()
    rows: list[dict[str, Any]] = []
    check_py_compile(rows)
    check_bash(rows)
    for path in candidate_files():
        if path.suffix.lower() == ".csv":
            check_csv(rows, path)
        elif path.suffix.lower() == ".md":
            check_markdown(rows, path)
        elif path.suffix.lower() == ".sh":
            check_shell_header(rows, path)
        check_line_endings(rows, path)
    write_csv(EXP10_TABLES_DIR / "readability_check_exp10.csv", rows)
    failures = [row for row in rows if row["status"] != "PASS"]
    lines = [
        "# Exp10 Readability Check",
        "",
        f"Status: `{'PASS' if not failures else 'FAIL'}`",
        "",
        "| check | path | status | details |",
        "| --- | --- | --- | --- |",
    ]
    lines.extend(f"| {row['check_name']} | {row['path']} | {row['status']} | {row.get('details', '')} |" for row in rows)
    write_text(EXP10_OUTPUT_DIR / "readability_check_exp10.md", "\n".join(lines))
    if failures:
        raise SystemExit(f"Exp10 readability failed. See {relpath(EXP10_OUTPUT_DIR)}")
    print("Exp10 readability PASS")


if __name__ == "__main__":
    main()
