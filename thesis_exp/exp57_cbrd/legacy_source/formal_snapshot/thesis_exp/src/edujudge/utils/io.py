"""I/O and path helpers for Exp 0.

All generated artifacts are intentionally kept inside ``thesis_exp``.
"""

from __future__ import annotations

import csv
import json
import re
import textwrap
import zipfile
from pathlib import Path
from typing import Any, Iterable, Iterator


REPO_ROOT = Path(__file__).resolve().parents[4]
THESIS_DIR = REPO_ROOT / "thesis_exp"
OUTPUT_DIR = THESIS_DIR / "outputs" / "exp00_data"
TABLES_DIR = OUTPUT_DIR / "tables"
SAMPLES_DIR = OUTPUT_DIR / "samples"
FIGURES_DIR = OUTPUT_DIR / "figures"
PROCESSED_DIR = THESIS_DIR / "data" / "processed"
SPLITS_DIR = THESIS_DIR / "data" / "splits"
CACHE_DIR = THESIS_DIR / ".cache"
MARKDOWN_WRAP_WIDTH = 100
MARKDOWN_TABLE_CELL_WIDTH = 72


def ensure_exp_dirs() -> None:
    for path in [
        THESIS_DIR,
        THESIS_DIR / "configs",
        THESIS_DIR / "src" / "edujudge" / "data",
        THESIS_DIR / "src" / "edujudge" / "plots",
        THESIS_DIR / "src" / "edujudge" / "utils",
        PROCESSED_DIR,
        SPLITS_DIR,
        OUTPUT_DIR,
        TABLES_DIR,
        SAMPLES_DIR,
        FIGURES_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def relpath(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def json_default(value: Any) -> str:
    return str(value)


def to_csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=json_default)
    return str(value)


def write_csv(path: Path | str, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        seen: list[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        fieldnames = seen
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: to_csv_cell(row.get(key, "")) for key in fieldnames})


def read_csv(path: Path | str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path | str, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True, default=json_default)


def write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=json_default) + "\n")


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def _is_markdown_passthrough(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.startswith("|"):
        return True
    if stripped.startswith(("#", "-", "*", ">", "```")):
        return True
    if re.match(r"^\d+\.\s", stripped):
        return True
    return False


def format_markdown_text(text: str) -> str:
    """Keep generated Markdown readable in raw GitHub views."""
    out: list[str] = []
    in_code = False
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            out.append(line)
            continue
        if in_code or _is_markdown_passthrough(line):
            out.append(line)
            continue
        out.extend(textwrap.wrap(line, width=MARKDOWN_WRAP_WIDTH, break_long_words=False, break_on_hyphens=False) or [""])
    return "\n".join(out)


def write_text(path: Path | str, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".md":
        text = format_markdown_text(text)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def iter_json_records(path: Path | str) -> Iterator[tuple[int, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if line.strip():
                    yield idx, json.loads(line)
        return
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            with path.open("r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if line.strip():
                        yield idx, json.loads(line)
            return
        if isinstance(data, list):
            for idx, item in enumerate(data):
                yield idx, item
        elif isinstance(data, dict):
            for idx, (key, value) in enumerate(data.items()):
                if isinstance(value, dict):
                    item = {"__key__": key, **value}
                else:
                    item = {"__key__": key, "__value__": value}
                yield idx, item


def count_json_records(path: Path | str) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            with path.open("r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        if isinstance(data, (list, dict)):
            return len(data)
        return 1
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            return len(zf.infolist())
    return 0


def flatten_keys(value: Any, prefix: str = "", max_depth: int = 3) -> set[str]:
    if max_depth < 0:
        return set()
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            keys.add(name)
            keys.update(flatten_keys(child, name, max_depth - 1))
    elif isinstance(value, list):
        for child in value[:5]:
            name = f"{prefix}[]" if prefix else "[]"
            keys.add(name)
            keys.update(flatten_keys(child, name, max_depth - 1))
    return keys


def md_table(rows: list[dict[str, Any]], fieldnames: list[str], max_rows: int = 30) -> str:
    shown = rows[:max_rows]
    if not shown:
        return "_No rows._"
    header = "| " + " | ".join(fieldnames) + " |"
    sep = "| " + " | ".join(["---"] * len(fieldnames)) + " |"
    body = []
    for row in shown:
        cells = []
        for key in fieldnames:
            cell = to_csv_cell(row.get(key, "")).replace("\n", "<br>")
            if len(cell) > MARKDOWN_TABLE_CELL_WIDTH:
                cell = cell[: MARKDOWN_TABLE_CELL_WIDTH - 3] + "..."
            cells.append(cell.replace("|", "\\|"))
        body.append("| " + " | ".join(cells) + " |")
    suffix = ""
    if len(rows) > max_rows:
        suffix = f"\n\n_Showing {max_rows} of {len(rows)} rows._"
    return "\n".join([header, sep, *body]) + suffix


def candidate_paths() -> list[Path]:
    """Return existing raw/source-like files, excluding generated Exp 0 outputs."""
    suffixes = {".json", ".jsonl", ".zip", ".py", ".pdf", ".xlsx", ".xls", ".csv"}
    ignored_parts = {".git", "__pycache__"}
    paths: list[Path] = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(REPO_ROOT)
        if any(part in ignored_parts for part in rel.parts):
            continue
        if rel.parts and rel.parts[0] == "thesis_exp":
            continue
        if path.suffix.lower() in suffixes:
            paths.append(path)
    return sorted(paths, key=lambda p: relpath(p))
