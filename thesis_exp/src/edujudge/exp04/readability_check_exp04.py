"""Line ending, parser, and artifact readability checks for Exp4."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp04 import EXP04_OUTPUT_DIR, EXP04_TABLES_DIR, ensure_exp04_dirs
from thesis_exp.src.edujudge.utils.io import REPO_ROOT, relpath, write_csv, write_text


SCRIPT_PATHS = [
    REPO_ROOT / "run_exp04_train_objectives.sh",
    REPO_ROOT / "thesis_exp" / "scripts" / "run_exp04_train_objectives.sh",
]
EXP04_SRC_DIR = REPO_ROOT / "thesis_exp" / "src" / "edujudge" / "exp04"


def one_line(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def add(rows: list[dict[str, Any]], check: str, path: Path | str, status: str, observed: Any, expected: Any, notes: str = "") -> None:
    rows.append(
        {
            "check": one_line(check),
            "path": relpath(path) if isinstance(path, Path) else str(path),
            "status": one_line(status),
            "observed": one_line(observed),
            "expected": one_line(expected),
            "notes": one_line(notes),
        }
    )


def markdown_table(rows: list[dict[str, Any]]) -> str:
    columns = ["check", "path", "status", "observed", "expected", "notes"]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        cells = []
        for col in columns:
            value = str(row.get(col, "")).replace("\n", " ").replace("|", "\\|")
            if len(value) > 90:
                value = value[:87] + "..."
            cells.append(value)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def command_status(args: list[str]) -> tuple[str, str]:
    result = subprocess.run(args, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    output = one_line(result.stdout)
    return ("PASS" if result.returncode == 0 else "FAIL", output[-700:] if output else "ok")


def check_text(rows: list[dict[str, Any]], path: Path, min_lines: int, check: str) -> None:
    if not path.exists():
        add(rows, check, path, "FAIL", "missing", "exists")
        return
    text = path.read_text(encoding="utf-8")
    line_count = len(text.splitlines())
    has_crlf = "\r\n" in text
    add(rows, check, path, "PASS" if line_count > min_lines else "FAIL", line_count, f">{min_lines}")
    add(rows, f"{check} LF line endings", path, "PASS" if not has_crlf else "FAIL", "CRLF" if has_crlf else "LF", "LF")


def check_scripts(rows: list[dict[str, Any]]) -> None:
    for path in SCRIPT_PATHS:
        check_text(rows, path, min_lines=2 if path.name == "run_exp04_train_objectives.sh" and path.parent == REPO_ROOT else 20, check="script line count")
        if path.exists():
            status, output = command_status(["bash", "-n", str(path.relative_to(REPO_ROOT))])
            add(rows, "bash -n", path, status, output, "ok")


def check_python_modules(rows: list[dict[str, Any]]) -> None:
    py_files = sorted(EXP04_SRC_DIR.glob("*.py"))
    for path in py_files:
        check_text(rows, path, min_lines=10, check="python module line count")
    status, output = command_status([sys.executable, "-m", "py_compile", *[str(path.relative_to(REPO_ROOT)) for path in py_files]])
    add(rows, "py_compile exp04 modules", EXP04_SRC_DIR, status, output, "ok")


def check_csv_files(rows: list[dict[str, Any]]) -> None:
    try:
        import pandas as pd
    except Exception as exc:
        add(rows, "pandas import", "pandas", "FAIL", f"{type(exc).__name__}: {exc}", "pandas import ok")
        return
    for path in sorted(EXP04_OUTPUT_DIR.rglob("*.csv")):
        try:
            frame = pd.read_csv(path)
            add(rows, "pandas.read_csv", path, "PASS", f"rows={len(frame)} cols={len(frame.columns)}", "readable")
        except Exception as exc:
            add(rows, "pandas.read_csv", path, "FAIL", f"{type(exc).__name__}: {exc}", "readable")


def check_jsonl_files(rows: list[dict[str, Any]]) -> None:
    for path in sorted(EXP04_OUTPUT_DIR.rglob("*.jsonl")):
        parsed = 0
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if line.strip():
                        json.loads(line)
                        parsed += 1
            add(rows, "jsonl line json.loads", path, "PASS", f"rows={parsed}", "all nonempty lines parse")
        except Exception as exc:
            add(rows, "jsonl line json.loads", path, "FAIL", f"line={line_number} {type(exc).__name__}: {exc}", "all nonempty lines parse")


def check_markdown_files(rows: list[dict[str, Any]]) -> None:
    for path in sorted(EXP04_OUTPUT_DIR.rglob("*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        max_len = max((len(line) for line in lines), default=0)
        add(rows, "markdown max line length", path, "PASS" if max_len < 300 else "FAIL", max_len, "<300")


def run_readability_check() -> list[dict[str, Any]]:
    ensure_exp04_dirs()
    rows: list[dict[str, Any]] = []
    check_scripts(rows)
    check_python_modules(rows)
    check_csv_files(rows)
    check_jsonl_files(rows)
    check_markdown_files(rows)
    write_csv(EXP04_TABLES_DIR / "readability_check_exp04.csv", rows)
    overall = "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL"
    write_text(
        EXP04_OUTPUT_DIR / "readability_check_exp04.md",
        f"""# Exp4 Readability Check

Overall status: **{overall}**

{markdown_table(rows)}
""",
    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Exp4 readability.")
    return parser.parse_args()


def main() -> None:
    parse_args()
    rows = run_readability_check()
    statuses = sorted({row["status"] for row in rows})
    print(f"Exp4 readability statuses: {', '.join(statuses)}")
    print(f"Output: {relpath(EXP04_OUTPUT_DIR / 'readability_check_exp04.md')}")
    if any(row["status"] != "PASS" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
