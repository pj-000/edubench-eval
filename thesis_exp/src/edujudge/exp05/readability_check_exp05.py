"""Line ending, parser, and readability checks for Exp5."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp05 import EXP05_OUTPUT_DIR, EXP05_TABLES_DIR, ensure_exp05_dirs
from thesis_exp.src.edujudge.utils.io import REPO_ROOT, relpath, write_csv, write_text


SCRIPT_PATHS = [
    REPO_ROOT / "thesis_exp" / "scripts" / "run_exp05_l1_smoke.sh",
    REPO_ROOT / "thesis_exp" / "scripts" / "run_exp05_l1_train.sh",
    REPO_ROOT / "thesis_exp" / "scripts" / "run_exp05_l2_smoke.sh",
    REPO_ROOT / "thesis_exp" / "scripts" / "run_exp05_l2_train.sh",
    REPO_ROOT / "thesis_exp" / "scripts" / "run_exp05_l3b_smoke.sh",
    REPO_ROOT / "thesis_exp" / "scripts" / "run_exp05_l3b_train.sh",
]
EXP05_SRC_DIR = REPO_ROOT / "thesis_exp" / "src" / "edujudge" / "exp05"
EXP05_CONFIG_DIR = REPO_ROOT / "thesis_exp" / "configs" / "exp05_low_score_loss"


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
            value = str(row.get(col, "")).replace("|", "\\|")
            if len(value) > 90:
                value = value[:87] + "..."
            cells.append(value)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def command_status(args: list[str]) -> tuple[str, str]:
    result = subprocess.run(args, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    output = one_line(result.stdout)
    return ("PASS" if result.returncode == 0 else "FAIL", output[-700:] if output else "ok")


def byte_line_status(path: Path) -> tuple[str, str]:
    data = path.read_bytes()
    if b"\r" in data:
        return "FAIL", "CRLF or mixed CR bytes detected"
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        return "FAIL", f"utf8 decode error: {exc}"
    return "PASS", f"LF bytes; lines={len(text.splitlines())} bytes={len(data)}"


def check_text(rows: list[dict[str, Any]], path: Path, min_lines: int, check: str) -> None:
    if not path.exists():
        add(rows, check, path, "FAIL", "missing", "exists")
        return
    text = path.read_text(encoding="utf-8")
    line_count = len(text.splitlines())
    add(rows, check, path, "PASS" if line_count > min_lines else "FAIL", line_count, f">{min_lines}")
    byte_status, byte_observed = byte_line_status(path)
    add(rows, f"{check} LF line endings", path, byte_status, byte_observed, "LF only")


def check_shell_scripts(rows: list[dict[str, Any]]) -> None:
    for path in SCRIPT_PATHS:
        check_text(rows, path, min_lines=30, check="script line count")
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
            add(rows, "shell shebang", path, "PASS" if lines and lines[0] == "#!/usr/bin/env bash" else "FAIL", lines[0] if lines else "", "#!/usr/bin/env bash")
            add(rows, "shell pipefail", path, "PASS" if "set -euo pipefail" in lines[:4] else "FAIL", "present" if "set -euo pipefail" in lines[:4] else "missing", "present")
            status, output = command_status(["bash", "-n", str(path.relative_to(REPO_ROOT))])
            add(rows, "bash -n", path, status, output, "ok")


def check_future_import(rows: list[dict[str, Any]], path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    stripped = [line.strip() for line in lines if line.strip()]
    status = "PASS"
    observed = "ok"
    if not stripped:
        status = "FAIL"
        observed = "empty"
    elif stripped[0].startswith(('"""', "'''")):
        quote = stripped[0][:3]
        end_idx = 0
        if not (stripped[0].endswith(quote) and len(stripped[0]) > 3):
            for idx, line in enumerate(stripped[1:], start=1):
                if line.endswith(quote):
                    end_idx = idx
                    break
        future_idx = end_idx + 1
        if future_idx >= len(stripped) or stripped[future_idx] != "from __future__ import annotations":
            status = "FAIL"
            observed = stripped[future_idx] if future_idx < len(stripped) else "missing"
    elif stripped[0] != "from __future__ import annotations":
        status = "FAIL"
        observed = stripped[0]
    add(rows, "future import placement", path, status, observed, "docstring then future import")


def check_python_modules(rows: list[dict[str, Any]]) -> None:
    py_files = sorted(EXP05_SRC_DIR.glob("*.py"))
    for path in py_files:
        check_text(rows, path, min_lines=5, check="python module line count")
        check_future_import(rows, path)
    status, output = command_status([sys.executable, "-m", "py_compile", *[str(path.relative_to(REPO_ROOT)) for path in py_files]])
    add(rows, "py_compile exp05 modules", EXP05_SRC_DIR, status, output, "ok")


def check_config_files(rows: list[dict[str, Any]]) -> None:
    for path in sorted(EXP05_CONFIG_DIR.glob("*.yaml")):
        check_text(rows, path, min_lines=5, check="config line count")
        byte_status, byte_observed = byte_line_status(path)
        add(rows, "config LF line endings", path, byte_status, byte_observed, "LF only")


def check_csv_files(rows: list[dict[str, Any]]) -> None:
    try:
        import pandas as pd
    except Exception as exc:
        add(rows, "pandas import", "pandas", "FAIL", f"{type(exc).__name__}: {exc}", "pandas import ok")
        return
    for path in sorted(EXP05_OUTPUT_DIR.rglob("*.csv")):
        if path.name.startswith("._"):
            continue
        try:
            frame = pd.read_csv(path)
            add(rows, "pandas.read_csv", path, "PASS", f"rows={len(frame)} cols={len(frame.columns)}", "readable")
        except Exception as exc:
            add(rows, "pandas.read_csv", path, "FAIL", f"{type(exc).__name__}: {exc}", "readable")


def check_jsonl_files(rows: list[dict[str, Any]]) -> None:
    for path in sorted(EXP05_OUTPUT_DIR.rglob("*.jsonl")):
        if path.name.startswith("._"):
            continue
        parsed = 0
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if line.strip():
                        json.loads(line)
                        parsed += 1
            add(rows, "jsonl line json.loads", path, "PASS", f"rows={parsed}", "all nonempty lines parse")
        except Exception as exc:
            add(rows, "jsonl line json.loads", path, "FAIL", f"line={line_number} {type(exc).__name__}: {exc}", "parse")


def check_markdown_files(rows: list[dict[str, Any]]) -> None:
    for path in sorted(EXP05_OUTPUT_DIR.rglob("*.md")):
        if path.name.startswith("._"):
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        prose_lines = [line for line in lines if not line.lstrip().startswith("|")]
        max_len = max((len(line) for line in prose_lines), default=0)
        table_max = max((len(line) for line in lines if line.lstrip().startswith("|")), default=0)
        add(
            rows,
            "markdown max prose line length",
            path,
            "PASS" if max_len < 300 else "FAIL",
            f"prose={max_len}; table={table_max}",
            "<300 for non-table lines",
        )


def run_readability_check() -> list[dict[str, Any]]:
    ensure_exp05_dirs()
    rows: list[dict[str, Any]] = []
    check_shell_scripts(rows)
    check_python_modules(rows)
    check_config_files(rows)
    check_csv_files(rows)
    check_jsonl_files(rows)
    check_markdown_files(rows)
    write_csv(EXP05_TABLES_DIR / "readability_check_exp05.csv", rows)
    overall = "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL"
    write_text(
        EXP05_OUTPUT_DIR / "readability_check_exp05.md",
        f"""# Exp5 Readability Check

Overall status: **{overall}**

{markdown_table(rows)}
""",
    )
    try:
        from thesis_exp.src.edujudge.exp05.write_exp05_report import write_review_package

        write_review_package()
    except Exception as exc:
        add(rows, "review package refresh", EXP05_OUTPUT_DIR / "review_package.md", "FAIL", f"{type(exc).__name__}: {exc}", "refresh ok")
        write_csv(EXP05_TABLES_DIR / "readability_check_exp05.csv", rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Exp5 readability.")
    return parser.parse_args()


def main() -> None:
    parse_args()
    rows = run_readability_check()
    statuses = sorted({row["status"] for row in rows})
    print(f"Exp5 readability statuses: {', '.join(statuses)}")
    print(f"Output: {relpath(EXP05_OUTPUT_DIR / 'readability_check_exp05.md')}")
    if any(row["status"] != "PASS" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
