"""Preflight checks for the Exp2 CE baseline training setup."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp02 import (
    EXPECTED_SPLIT_ROWS,
    EXP02_DATA_DIR,
    EXP02_OUTPUT_DIR,
    EXP02_TABLES_DIR,
    ensure_exp02_dirs,
)
from thesis_exp.src.edujudge.exp02.build_exp02_dataset import build_exp02_dataset
from thesis_exp.src.edujudge.utils.io import REPO_ROOT, read_jsonl, relpath, write_csv, write_text


REQUIRED_PROMPT_PARTS = [
    "Question:\n",
    "Answer:\n",
    "Evaluation Dimension:\n",
    "Predict the human-aligned educational quality score from 1 to 5.",
]

FORBIDDEN_TEMPLATE_RE = re.compile(
    r"(?im)^\s*(\[Rubric\]|\[Metadata\]|Rubric\s*:|Metadata\s*:|Subject\s*:|"
    r"Scenario\s*:|Generator model\s*:|Education level\s*:|Language\s*:|"
    r"Metric group\s*:|Score anchors\s*:)"
)

REQUIRED_GITIGNORE_PATTERNS = [
    "thesis_exp/artifacts/",
    "**/checkpoints/",
    "**/*.safetensors",
    "**/*.bin",
    "**/*.pt",
    "**/*.pth",
    "**/optimizer.pt",
    "**/scheduler.pt",
    "**/trainer_state.json",
    "**/rng_state.pth",
    "**/hf_cache/",
    "wandb/",
    "runs/",
]


def add(rows: list[dict[str, Any]], check: str, status: str, observed: Any, expected: Any, notes: str = "") -> None:
    rows.append({"check": check, "status": status, "observed": observed, "expected": expected, "notes": notes})


def markdown_table(rows: list[dict[str, Any]]) -> str:
    columns = ["check", "status", "observed", "expected", "notes"]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")).replace("|", "\\|") for col in columns) + " |")
    return "\n".join(lines)


def command_status(args: list[str]) -> tuple[str, str]:
    result = subprocess.run(args, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    output = result.stdout.strip()
    return ("PASS" if result.returncode == 0 else "FAIL", output[-500:] if output else "")


def gitignore_text() -> str:
    parts = []
    for path in [REPO_ROOT / ".gitignore", REPO_ROOT / "thesis_exp" / ".gitignore"]:
        if path.exists():
            parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def check_split(rows: list[dict[str, Any]], split: str, data: list[dict[str, Any]], expected_rows: int) -> None:
    add(rows, f"{split} row count", "PASS" if len(data) == expected_rows else "FAIL", len(data), expected_rows)

    labels = [row.get("label") for row in data]
    labels_5 = [row.get("label_5") for row in data]
    label_range_ok = all(isinstance(label, int) and 0 <= label <= 4 for label in labels)
    label_5_range_ok = all(isinstance(label, int) and 1 <= label <= 5 for label in labels_5)
    add(rows, f"{split} label range 0..4", "PASS" if label_range_ok else "FAIL", sorted(set(labels)), "0..4")
    add(rows, f"{split} label_5 range 1..5", "PASS" if label_5_range_ok else "FAIL", sorted(set(labels_5)), "1..5")

    template_names = {row.get("template_name") for row in data}
    add(
        rows,
        f"{split} template_name",
        "PASS" if template_names == {"qa_metric_baseline"} else "FAIL",
        sorted(str(value) for value in template_names),
        "qa_metric_baseline",
    )

    missing_required = 0
    forbidden_rows = 0
    for row in data:
        text = str(row.get("text", ""))
        if not all(part in text for part in REQUIRED_PROMPT_PARTS):
            missing_required += 1
        if FORBIDDEN_TEMPLATE_RE.search(text):
            forbidden_rows += 1
    add(rows, f"{split} prompt required fields", "PASS" if missing_required == 0 else "FAIL", missing_required, 0)
    add(rows, f"{split} prompt excludes rubric/metadata", "PASS" if forbidden_rows == 0 else "FAIL", forbidden_rows, 0)


def run_sanity_check() -> list[dict[str, Any]]:
    ensure_exp02_dirs()
    build_exp02_dataset()

    rows: list[dict[str, Any]] = []
    split_data: dict[str, list[dict[str, Any]]] = {}
    for split, expected_rows in EXPECTED_SPLIT_ROWS.items():
        data = read_jsonl(EXP02_DATA_DIR / f"{split}.jsonl")
        split_data[split] = data
        check_split(rows, split, data, expected_rows)

    for key in ["record_id", "triple_key"]:
        split_values = {
            split: {str(row.get(key)) for row in data if row.get(key)}
            for split, data in split_data.items()
        }
        for left, right in [("train", "dev"), ("train", "test"), ("dev", "test")]:
            overlap = split_values[left] & split_values[right]
            add(rows, f"{left}/{right} {key} overlap", "PASS" if not overlap else "FAIL", len(overlap), 0)

    ignore_text = gitignore_text()
    missing_patterns = [pattern for pattern in REQUIRED_GITIGNORE_PATTERNS if pattern not in ignore_text]
    add(
        rows,
        ".gitignore model artifact coverage",
        "PASS" if not missing_patterns else "FAIL",
        missing_patterns,
        "all required patterns present",
        "checks root .gitignore and thesis_exp/.gitignore",
    )

    bash_status, bash_output = command_status(["bash", "-n", "thesis_exp/scripts/run_exp02_train_ce_0_6b.sh"])
    add(rows, "bash -n run_exp02_train_ce_0_6b.sh", bash_status, bash_output or "ok", "ok")

    exp02_py_files = sorted(str(path.relative_to(REPO_ROOT)) for path in (REPO_ROOT / "thesis_exp/src/edujudge/exp02").glob("*.py"))
    py_status, py_output = command_status([sys.executable, "-m", "py_compile", *exp02_py_files])
    add(rows, "py_compile exp02 modules", py_status, py_output or "ok", "ok")

    write_csv(EXP02_TABLES_DIR / "sanity_check_exp02_train_setup.csv", rows)
    overall = "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL"
    write_text(
        EXP02_OUTPUT_DIR / "sanity_check_exp02_train_setup.md",
        f"""# Exp2 Train Setup Sanity Check

Overall status: **{overall}**

{markdown_table(rows)}
""",
    )
    return rows


def main() -> None:
    rows = run_sanity_check()
    statuses = sorted({row["status"] for row in rows})
    print(f"Exp2 train setup sanity statuses: {', '.join(statuses)}")
    print(f"Outputs: {relpath(EXP02_OUTPUT_DIR / 'sanity_check_exp02_train_setup.md')}")
    failed = [row for row in rows if row["status"] != "PASS"]
    if failed:
        for row in failed:
            print(f"FAIL {row['check']}: observed={row['observed']} expected={row['expected']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
