"""Readability and format checks for Exp9 files."""

from __future__ import annotations

import csv
import json
import py_compile
import re
import subprocess
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ModuleNotFoundError:  # Keep server-side readability independent of optional dataframe deps.
    pd = None

from thesis_exp.src.edujudge.exp09_pairwise_ordinal import (
    EXP09_CONFIG_DIR,
    EXP09_OUTPUT_DIR,
    EXP09_SRC_DIR,
    EXP09_TABLES_DIR,
    ensure_exp09_dirs,
)
from thesis_exp.src.edujudge.exp09_pairwise_ordinal.data import tracked_weight_files
from thesis_exp.src.edujudge.utils.io import relpath, write_csv, write_text


SCRIPT_PATHS = [
    Path("thesis_exp/scripts/run_exp09_qdpr1_smoke.sh"),
    Path("thesis_exp/scripts/run_exp09_qdpr1_train.sh"),
    Path("thesis_exp/scripts/sync_exp09_qdpr1_to_server.sh"),
]
TEXT_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".md", ".csv", ".json", ".txt"}
CHECKPOINT_SUFFIXES = {".bin", ".safetensors", ".pt", ".pth", ".ckpt"}
COLLAPSE_CHECK_SUFFIXES = {".py", ".sh", ".md", ".csv"}
FIRST_LINE_CHECK_SUFFIXES = TEXT_SUFFIXES
LARGE_SINGLE_LINE_BYTES = 1000
FIRST_LINE_MAX_CHARS = 1000
MIN_LINE_COUNTS = {
    Path("thesis_exp/src/edujudge/exp09_pairwise_ordinal/pair_builder.py"): 180,
    Path("thesis_exp/src/edujudge/exp09_pairwise_ordinal/train_qdpr1_pairwise.py"): 300,
    Path("thesis_exp/src/edujudge/exp09_pairwise_ordinal/sanity_check_exp09_setup.py"): 150,
    Path("thesis_exp/src/edujudge/exp09_pairwise_ordinal/readability_check_exp09.py"): 100,
}
SKIP_OUTPUT_DIRS = {"runs", "smoke_test", "logs", "arrays", "predictions", "pairs"}
SELF_OUTPUT_FILES = {Path("readability_check_exp09.md"), Path("tables/readability_check_exp09.csv")}


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
    files = sorted(EXP09_SRC_DIR.glob("*.py"))
    files.extend(sorted(EXP09_CONFIG_DIR.glob("*.yaml")))
    files.extend(path for path in SCRIPT_PATHS if path.exists())
    if EXP09_OUTPUT_DIR.exists():
        for path in sorted(EXP09_OUTPUT_DIR.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(EXP09_OUTPUT_DIR)
            if relative.parts and relative.parts[0] in SKIP_OUTPUT_DIRS:
                continue
            if relative in SELF_OUTPUT_FILES:
                continue
            files.append(path)
    unique: list[Path] = []
    seen = set()
    for path in files:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def check_py_compile(rows: list[dict[str, Any]]) -> None:
    for path in sorted(EXP09_SRC_DIR.glob("*.py")):
        try:
            py_compile.compile(str(path), doraise=True)
            add(rows, "python py_compile", path, True)
        except py_compile.PyCompileError as exc:
            add(rows, "python py_compile", path, False, str(exc))


def check_bash(rows: list[dict[str, Any]]) -> None:
    for path in SCRIPT_PATHS:
        result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        add(rows, "shell bash -n", path, result.returncode == 0, (result.stderr or result.stdout).strip())


def check_csv(rows: list[dict[str, Any]], path: Path) -> None:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            list(csv.DictReader(handle))
        add(rows, "CSV readable", path, True)
    except Exception as exc:
        add(rows, "CSV readable", path, False, f"{type(exc).__name__}: {exc}")
    if pd is None:
        add(rows, "CSV pandas readable", path, True)
        return
    try:
        pd.read_csv(path)
        add(rows, "CSV pandas readable", path, True)
    except Exception as exc:
        add(rows, "CSV pandas readable", path, False, f"{type(exc).__name__}: {exc}")


def check_json(rows: list[dict[str, Any]], path: Path) -> None:
    try:
        json.loads(path.read_text(encoding="utf-8"))
        add(rows, "JSON readable", path, True)
    except Exception as exc:
        add(rows, "JSON readable", path, False, f"{type(exc).__name__}: {exc}")


def check_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    prose_lines = [line for line in lines if not line.lstrip().startswith("|")]
    max_len = max((len(line) for line in prose_lines), default=0)
    table_max = max((len(line) for line in lines if line.lstrip().startswith("|")), default=0)
    add(rows, "Markdown max prose line length", path, max_len < 300, f"prose={max_len}; table={table_max}")


def check_line_endings(rows: list[dict[str, Any]], path: Path) -> None:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return
    data = path.read_bytes()
    has_crlf = b"\r\n" in data
    has_cr_only = b"\r" in data.replace(b"\r\n", b"")
    add(rows, "no CRLF line endings", path, not has_crlf)
    add(rows, "no CR-only line endings", path, not has_cr_only)
    add(rows, "LF line endings", path, not has_crlf and not has_cr_only)
    if path.suffix.lower() in FIRST_LINE_CHECK_SUFFIXES and data:
        first_line = data.split(b"\n", 1)[0].rstrip(b"\r")
        add(rows, "first line <= 1000 chars", path, len(first_line) <= FIRST_LINE_MAX_CHARS, len(first_line))
    if path.suffix.lower() in COLLAPSE_CHECK_SUFFIXES and data:
        line_count = len(data.splitlines())
        add(rows, "not collapsed-line file", path, line_count > 1, line_count)
        if path.suffix.lower() in {".md", ".csv"}:
            collapsed_large = line_count <= 1 and len(data) > LARGE_SINGLE_LINE_BYTES
            add(
                rows,
                "not large one-line md/csv",
                path,
                not collapsed_large,
                f"lines={line_count}; bytes={len(data)}",
            )
        minimum = MIN_LINE_COUNTS.get(Path(relpath(path)))
        if minimum is not None:
            add(rows, "minimum line count", path, line_count > minimum, f"lines={line_count}; min>{minimum}")


def check_size_and_weights(rows: list[dict[str, Any]], path: Path) -> None:
    size = path.stat().st_size
    add(rows, "file size below 20MB", path, size < 20_000_000, size)
    add(rows, "not checkpoint/weight artifact", path, path.suffix.lower() not in CHECKPOINT_SUFFIXES)


def check_no_api_key(rows: list[dict[str, Any]], path: Path) -> None:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return
    text = path.read_text(encoding="utf-8", errors="ignore")
    api_env_name = "OPENAI" + "_API_KEY"
    has_key = api_env_name in text or re.search(r"\bsk-[A-Za-z0-9_-]{20,}", text) is not None
    add(rows, "no API key string", path, not has_key)


def main() -> None:
    ensure_exp09_dirs()
    rows: list[dict[str, Any]] = []
    check_py_compile(rows)
    check_bash(rows)
    for path in candidate_files():
        suffix = path.suffix.lower()
        if suffix == ".csv":
            check_csv(rows, path)
        elif suffix == ".json":
            check_json(rows, path)
        elif suffix == ".md":
            check_markdown(rows, path)
        check_line_endings(rows, path)
        check_size_and_weights(rows, path)
        check_no_api_key(rows, path)
    weights = tracked_weight_files()
    add(rows, "no tracked checkpoint/weights", Path("."), not weights, ", ".join(weights))
    write_csv(EXP09_TABLES_DIR / "readability_check_exp09.csv", rows)
    failures = [row for row in rows if row["status"] != "PASS"]
    lines = [
        "# Exp9 Readability Check",
        "",
        f"Status: `{'PASS' if not failures else 'FAIL'}`",
        "",
        "| check | path | status | details |",
        "| --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {row['check_name']} | {row['path']} | {row['status']} | {row.get('details', '')} |" for row in rows
    )
    write_text(EXP09_OUTPUT_DIR / "readability_check_exp09.md", "\n".join(lines))
    if failures:
        raise SystemExit(f"Exp9 readability failed. See {relpath(EXP09_OUTPUT_DIR)}")
    print("Exp9 readability PASS")


if __name__ == "__main__":
    main()
