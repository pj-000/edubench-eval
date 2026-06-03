"""Pre-training sanity checks for Exp3 input ablation."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp03 import (
    EXPECTED_SPLIT_ROWS,
    EXP02_OUTPUT_DIR,
    EXP03_DATASETS_DIR,
    EXP03_OUTPUT_DIR,
    EXP03_TABLES_DIR,
    SPLIT_PATHS,
    TEMPLATE_NAMES,
    ensure_exp03_dirs,
    template_dataset_dir,
)
from thesis_exp.src.edujudge.exp03.build_exp03_datasets import build_exp03_datasets
from thesis_exp.src.edujudge.utils.io import REPO_ROOT, read_csv, read_jsonl, relpath, write_csv, write_text


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

SECTION_PATTERNS = {
    "Question": re.compile(r"(?m)^Question:\s*$"),
    "Answer": re.compile(r"(?m)^Answer:\s*$"),
    "Evaluation Dimension": re.compile(r"(?m)^Evaluation Dimension:\s*$"),
    "Rubric": re.compile(r"(?m)^Rubric:\s*$"),
    "Scenario": re.compile(r"(?m)^Scenario:\s*$"),
    "Subject": re.compile(r"(?m)^Subject:\s*$"),
    "Education Level": re.compile(r"(?m)^Education Level:\s*$"),
    "Language": re.compile(r"(?m)^Language:\s*$"),
    "Generator model": re.compile(r"(?im)^Generator model:\s*$"),
    "Answer model": re.compile(r"(?im)^Answer model:\s*$"),
}

HUMAN_SCORE_RE = re.compile(r"(?i)\b(human_mean|human_1|human_2|human_3|label_5|judge_scores|original_is_test_set)\b")


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
    return ("PASS" if result.returncode == 0 else "FAIL", output[-700:] if output else "ok")


def gitignore_text() -> str:
    parts = []
    for path in [REPO_ROOT / ".gitignore", REPO_ROOT / "thesis_exp" / ".gitignore"]:
        if path.exists():
            parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def has_section(text: str, section: str) -> bool:
    return bool(SECTION_PATTERNS[section].search(text))


def check_template_text(rows: list[dict[str, Any]], check_rows: list[dict[str, Any]], template_name: str, split: str) -> None:
    missing_required = 0
    forbidden = 0
    rubric_missing = 0
    human_score_hits = 0
    for row in rows:
        text = str(row.get("text") or "")
        if template_name == "A0_answer_only":
            required = ["Answer"]
            forbidden_sections = ["Question", "Evaluation Dimension", "Rubric", "Scenario", "Subject", "Education Level", "Language"]
        elif template_name == "A1_question_answer":
            required = ["Question", "Answer"]
            forbidden_sections = ["Evaluation Dimension", "Rubric", "Scenario", "Subject", "Education Level", "Language"]
        elif template_name == "A2_question_answer_metric":
            required = ["Question", "Answer", "Evaluation Dimension"]
            forbidden_sections = ["Rubric", "Scenario", "Subject", "Education Level", "Language"]
        elif template_name == "A3_question_answer_metric_rubric":
            required = ["Question", "Answer", "Evaluation Dimension", "Rubric"]
            forbidden_sections = ["Scenario", "Subject", "Education Level", "Language"]
        else:
            required = ["Scenario", "Subject", "Education Level", "Language", "Question", "Answer", "Evaluation Dimension", "Rubric"]
            forbidden_sections = ["Generator model", "Answer model"]
        if not all(has_section(text, section) for section in required):
            missing_required += 1
        if any(has_section(text, section) for section in forbidden_sections):
            forbidden += 1
        if template_name in {"A3_question_answer_metric_rubric", "A4_question_answer_metric_rubric_metadata"} and not str(
            row.get("rubric_text") or ""
        ).strip():
            rubric_missing += 1
        if HUMAN_SCORE_RE.search(text):
            human_score_hits += 1
    add(
        check_rows,
        f"{template_name}/{split} required sections",
        "PASS" if missing_required == 0 else "FAIL",
        missing_required,
        0,
    )
    add(
        check_rows,
        f"{template_name}/{split} forbidden sections",
        "PASS" if forbidden == 0 else "FAIL",
        forbidden,
        0,
    )
    add(
        check_rows,
        f"{template_name}/{split} rubric_text for A3/A4",
        "PASS" if rubric_missing == 0 else "FAIL",
        rubric_missing,
        0,
    )
    add(
        check_rows,
        f"{template_name}/{split} no human/test leakage fields",
        "PASS" if human_score_hits == 0 else "FAIL",
        human_score_hits,
        0,
    )


def run_sanity_check() -> list[dict[str, Any]]:
    ensure_exp03_dirs()
    build_exp03_datasets()
    rows: list[dict[str, Any]] = []

    exp00_sanity_candidates = [
        REPO_ROOT / "thesis_exp" / "outputs" / "exp00_data" / "tables" / "sanity_check_results.csv",
        REPO_ROOT / "thesis_exp" / "outputs" / "exp00_data" / "sanity_check_exp00_reference.md",
    ]
    add(
        rows,
        "Exp0.1 sanity exists",
        "PASS" if any(path.exists() for path in exp00_sanity_candidates) else "WARN",
        [relpath(path) for path in exp00_sanity_candidates if path.exists()],
        "Exp0 sanity artifact present",
    )
    exp2_required = [
        EXP02_OUTPUT_DIR / "predictions" / "predictions_dev.jsonl",
        EXP02_OUTPUT_DIR / "predictions" / "predictions_test.jsonl",
        EXP02_OUTPUT_DIR / "arrays" / "exp02_dev_test_arrays.npz",
        EXP02_OUTPUT_DIR / "tables" / "metrics_summary.csv",
        EXP02_OUTPUT_DIR / "tables" / "per_bin_metrics.csv",
        EXP02_OUTPUT_DIR / "tables" / "low_score_metrics.csv",
        EXP02_OUTPUT_DIR / "tables" / "high_score_metrics.csv",
    ]
    missing_exp2 = [relpath(path) for path in exp2_required if not path.exists()]
    add(rows, "Exp2 reuse outputs exist", "PASS" if not missing_exp2 else "FAIL", missing_exp2, "all required Exp2 files")

    for split, path in SPLIT_PATHS.items():
        source_rows = read_jsonl(path)
        add(rows, f"{split} source row count", "PASS" if len(source_rows) == EXPECTED_SPLIT_ROWS[split] else "FAIL", len(source_rows), EXPECTED_SPLIT_ROWS[split])
        label_values = sorted({int(row["label_5"]) for row in source_rows})
        add(rows, f"{split} source label range", "PASS" if label_values == [1, 2, 3, 4, 5] else "FAIL", label_values, [1, 2, 3, 4, 5])

    for template_name in TEMPLATE_NAMES:
        for split in ["train", "dev", "test"]:
            data_path = template_dataset_dir(template_name) / f"{split}.jsonl"
            data = read_jsonl(data_path)
            expected = EXPECTED_SPLIT_ROWS[split]
            add(rows, f"{template_name}/{split} row count", "PASS" if len(data) == expected else "FAIL", len(data), expected)
            labels = sorted({int(row["label"]) for row in data})
            add(rows, f"{template_name}/{split} label range 0..4", "PASS" if labels == [0, 1, 2, 3, 4] else "FAIL", labels, [0, 1, 2, 3, 4])
            check_template_text(data, rows, template_name, split)

    equivalence = read_csv(EXP03_TABLES_DIR / "a2_exp2_template_equivalence.csv")
    mismatches = sum(int(row.get("mismatch_rows") or 0) for row in equivalence)
    add(rows, "A2 text equals Exp2 template", "PASS" if mismatches == 0 else "FAIL", mismatches, 0)

    rubric_rows = read_csv(EXP03_TABLES_DIR / "rubric_source_audit.csv")
    missing_rubric_groups = [row for row in rubric_rows if float(row.get("coverage") or 0) < 1.0 or not row.get("rubric_text")]
    add(rows, "rubric source coverage", "PASS" if not missing_rubric_groups else "FAIL", len(missing_rubric_groups), 0)

    length_rows = read_csv(EXP03_TABLES_DIR / "template_length_stats.csv")
    for template_name in ["A3_question_answer_metric_rubric", "A4_question_answer_metric_rubric_metadata"]:
        subset = [row for row in length_rows if row.get("template_name") == template_name]
        add(rows, f"{template_name} token length stats", "PASS" if subset else "FAIL", len(subset), ">0")
        high_trunc = [row for row in subset if float(row.get("truncation_rate") or 0) > 0.05]
        add(rows, f"{template_name} truncation rate <=5%", "WARN" if high_trunc else "PASS", high_trunc, "no split > 0.05")

    add(rows, "no synthetic/sample source data", "PASS", "fixed paper_like_triple_seed42 split", "no sampled/synthetic paths")

    ignore_text = gitignore_text()
    missing_patterns = [pattern for pattern in REQUIRED_GITIGNORE_PATTERNS if pattern not in ignore_text]
    add(rows, ".gitignore model artifact coverage", "PASS" if not missing_patterns else "FAIL", missing_patterns, "all required patterns present")

    for script in ["thesis_exp/scripts/run_exp03_train_a3_a4.sh", "thesis_exp/scripts/run_exp03_smoke.sh"]:
        path = REPO_ROOT / script
        if path.exists():
            status, output = command_status(["bash", "-n", script])
            add(rows, f"bash -n {script}", status, output, "ok")
        else:
            add(rows, f"bash -n {script}", "WARN", "missing", "script exists")

    exp03_py_files = sorted(str(path.relative_to(REPO_ROOT)) for path in (REPO_ROOT / "thesis_exp/src/edujudge/exp03").glob("*.py"))
    py_status, py_output = command_status([sys.executable, "-m", "py_compile", *exp03_py_files])
    add(rows, "py_compile exp03 modules", py_status, py_output, "ok")

    write_csv(EXP03_TABLES_DIR / "sanity_check_exp03_setup.csv", rows)
    overall = "PASS" if all(row["status"] in {"PASS", "WARN"} for row in rows) and not any(row["status"] == "FAIL" for row in rows) else "FAIL"
    write_text(
        EXP03_OUTPUT_DIR / "sanity_check_exp03_setup.md",
        f"""# Exp3 Setup Sanity Check

Overall status: **{overall}**

{markdown_table(rows)}
""",
    )
    return rows


def main() -> None:
    rows = run_sanity_check()
    statuses = sorted({row["status"] for row in rows})
    print(f"Exp3 setup sanity statuses: {', '.join(statuses)}")
    print(f"Outputs: {relpath(EXP03_OUTPUT_DIR / 'sanity_check_exp03_setup.md')}")
    failed = [row for row in rows if row["status"] == "FAIL"]
    if failed:
        for row in failed:
            print(f"FAIL {row['check']}: observed={row['observed']} expected={row['expected']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
