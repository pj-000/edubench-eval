"""Sanity checks for the Exp6-3 mini-batch generation audit pipeline."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp06_generation import (
    EXP06_GENERATION_PROMPTS_DIR,
    MINI_BATCH_TOTAL_TARGET,
    ensure_mini_batch_dirs,
)
from thesis_exp.src.edujudge.exp06_generation.mini_batch_common import (
    count_jsonl_lines,
    mini_filtered_path,
    mini_generated_path,
    mini_leakage_path,
    mini_prompt_path,
    mini_report_path,
    mini_spotcheck_path,
    mini_table_path,
    read_mini_table,
    split_key_cache,
    write_mini_table,
)
from thesis_exp.src.edujudge.utils.io import REPO_ROOT, relpath, write_text
from thesis_exp.src.edujudge.utils.text_norm import stringify


FIELDS = ["check_name", "status", "details"]


def add(rows: list[dict[str, Any]], name: str, status: str, details: str) -> None:
    rows.append({"check_name": name, "status": status, "details": details})


def git_output(args: list[str]) -> list[str]:
    try:
        proc = subprocess.run(["git", *args], cwd=REPO_ROOT, check=False, text=True, capture_output=True)
    except OSError as exc:
        return [f"GIT_ERROR:{exc}"]
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def tracked_weight_files() -> list[str]:
    files = git_output(["ls-files"])
    suffixes = (".pt", ".pth", ".bin", ".safetensors", ".ckpt")
    blocked_parts = {"/checkpoints/", "/weights/", "/model_weights/", "/wandb/", "/runs/"}
    return [
        path
        for path in files
        if path.endswith(suffixes) or any(part in f"/{path}" for part in blocked_parts)
    ]


def exp0_to_exp5_diffs() -> list[str]:
    pathspecs = [
        "thesis_exp/outputs/exp00_data",
        "thesis_exp/outputs/exp01*",
        "thesis_exp/outputs/exp02*",
        "thesis_exp/outputs/exp03*",
        "thesis_exp/outputs/exp04*",
        "thesis_exp/outputs/exp05*",
        "thesis_exp/src/edujudge/exp00*",
        "thesis_exp/src/edujudge/exp01*",
        "thesis_exp/src/edujudge/exp02*",
        "thesis_exp/src/edujudge/exp03*",
        "thesis_exp/src/edujudge/exp04*",
        "thesis_exp/src/edujudge/exp05*",
    ]
    return git_output(["diff", "--name-only", "--", *pathspecs])


def generated_status() -> bool:
    raw_path = mini_generated_path("raw_generations.jsonl")
    return raw_path.exists() and count_jsonl_lines(raw_path) > 0


def run_checks() -> list[dict[str, Any]]:
    ensure_mini_batch_dirs()
    rows: list[dict[str, Any]] = []
    targets = read_mini_table("mini_batch_target_matrix.csv")
    sources = read_mini_table("mini_batch_source_selection.csv")
    keys = split_key_cache()

    add(rows, "target_matrix_exists", "PASS" if targets else "FAIL", relpath(mini_table_path("mini_batch_target_matrix.csv")))
    add(rows, "source_selection_exists", "PASS" if sources else "FAIL", relpath(mini_table_path("mini_batch_source_selection.csv")))
    prompt_path = mini_prompt_path("mini_batch_prompts.jsonl")
    add(rows, "prompts_exist", "PASS" if prompt_path.exists() and count_jsonl_lines(prompt_path) else "FAIL", relpath(prompt_path))
    planned = sum(int(row.get("target_count") or 0) for row in targets)
    add(rows, "planned_count_le_24", "PASS" if planned <= MINI_BATCH_TOTAL_TARGET else "FAIL", f"planned={planned}")
    labels = {stringify(row.get("target_label_5")) for row in targets}
    add(rows, "labels_only_1_2_3", "PASS" if labels <= {"1", "2", "3"} else "FAIL", f"labels={sorted(labels)}")
    add(rows, "target_rows_equal_24", "PASS" if len(targets) == MINI_BATCH_TOTAL_TARGET else "FAIL", f"rows={len(targets)}")
    actual_sources = [row for row in sources if stringify(row.get("source_record_id")).strip()]
    add(
        rows,
        "source_selection_complete",
        "PASS" if len(actual_sources) == MINI_BATCH_TOTAL_TARGET else "FAIL",
        f"selected_sources={len(actual_sources)} required={MINI_BATCH_TOTAL_TARGET}",
    )
    source_leaks = [
        row
        for row in actual_sources
        if row.get("source_question_key") in keys["dev"]["source_question_key"]
        or row.get("source_question_key") in keys["test"]["source_question_key"]
        or row.get("source_triple_key") in keys["dev"]["source_triple_key"]
        or row.get("source_triple_key") in keys["test"]["source_triple_key"]
        or stringify(row.get("source_split")) != "train"
    ]
    add(rows, "no_dev_test_source_selected", "PASS" if not source_leaks and sources else "FAIL", f"leak_or_nontrain_rows={len(source_leaks)}")
    full_templates = [
        EXP06_GENERATION_PROMPTS_DIR / "generate_score_controlled_answer.md",
        EXP06_GENERATION_PROMPTS_DIR / "generate_score_controlled_answer_en.md",
        EXP06_GENERATION_PROMPTS_DIR / "generate_score_controlled_answer_zh.md",
    ]
    missing_templates = [relpath(path) for path in full_templates if not path.exists()]
    add(rows, "full_score_prompt_templates_exist", "PASS" if not missing_templates else "FAIL", f"missing={missing_templates}")

    raw_path = mini_generated_path("raw_generations.jsonl")
    normalized_path = mini_generated_path("normalized_synthetic_candidates.jsonl")
    if raw_path.exists():
        add(rows, "raw_has_normalized", "PASS" if normalized_path.exists() else "FAIL", relpath(normalized_path))
    else:
        add(rows, "raw_has_normalized", "DRY_RUN", "raw_generations.jsonl absent in dry-run")
    filtered_path = mini_filtered_path("filtered_synthetic_candidates.jsonl")
    filter_report = mini_table_path("filter_report.csv")
    if filtered_path.exists():
        add(rows, "filtered_has_filter_report", "PASS" if filter_report.exists() else "FAIL", relpath(filter_report))
    else:
        add(rows, "filtered_has_filter_report", "DRY_RUN", "filtered_synthetic_candidates.jsonl absent")
    generated = generated_status()
    add(
        rows,
        "leakage_report_if_generated",
        "PASS" if (not generated or mini_leakage_path("leakage_report.md").exists()) else "FAIL",
        relpath(mini_leakage_path("leakage_report.md")),
    )
    add(
        rows,
        "spotcheck_package_if_generated",
        "PASS" if (not generated or mini_spotcheck_path("spotcheck_samples.csv").exists()) else "FAIL",
        relpath(mini_spotcheck_path("spotcheck_samples.csv")),
    )
    weights = tracked_weight_files()
    add(rows, "no_checkpoint_or_weight_files_tracked", "PASS" if not weights else "FAIL", "; ".join(weights[:20]))
    exp_diffs = exp0_to_exp5_diffs()
    add(rows, "no_exp0_to_exp5_tracked_diff", "PASS" if not exp_diffs else "FAIL", "; ".join(exp_diffs[:20]))
    return rows


def write_report(rows: list[dict[str, Any]]) -> None:
    failures = [row for row in rows if row["status"] == "FAIL"]
    dry = [row for row in rows if row["status"] == "DRY_RUN"]
    status = "FAIL" if failures else "PASS_WITH_DRY_RUN_NOTES" if dry else "PASS"
    lines = [
        "# Exp6-3 Mini-batch Sanity Check",
        "",
        f"Overall status: **{status}**",
        "",
        f"- Checks: **{len(rows)}**",
        f"- Failures: **{len(failures)}**",
        f"- Dry-run notes: **{len(dry)}**",
        "",
        "| check | status | details |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        detail = stringify(row.get("details")).replace("|", "\\|")
        if len(detail) > 180:
            detail = detail[:177] + "..."
        lines.append(f"| {row['check_name']} | {row['status']} | {detail} |")
    write_text(mini_report_path("sanity_check_mini_batch_generation.md"), "\n".join(lines))


def run_sanity() -> list[dict[str, Any]]:
    rows = run_checks()
    write_mini_table("sanity_check_mini_batch_generation.csv", rows, FIELDS)
    write_report(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    rows = run_sanity()
    failures = sum(1 for row in rows if row["status"] == "FAIL")
    print(f"Wrote mini-batch sanity check rows={len(rows)} failures={failures}")


if __name__ == "__main__":
    main()
