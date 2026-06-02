"""Unified entry point for Exp1 evaluator-vs-human audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from thesis_exp.src.edujudge.audit import (
    DATASET_NAME,
    EXPECTED_TEST_ROWS,
    EXP00_OUTPUT_DIR,
    EXP01_OUTPUT_DIR,
    EXP01_TABLES_DIR,
    PROCESSED_DATASET_PATH,
    TEST_SPLIT_PATH,
    collect_candidate_paths,
    ensure_exp01_dirs,
    is_synthetic_or_sampled_path,
    markdown_table,
    read_jsonl,
    relpath,
    write_csv,
    write_text,
)
from thesis_exp.src.edujudge.audit.align_judge_scores import run_alignment
from thesis_exp.src.edujudge.audit.compute_audit_metrics import run_metrics
from thesis_exp.src.edujudge.audit.judge_inventory import run_inventory
from thesis_exp.src.edujudge.audit.plot_audit_figures import run_plots
from thesis_exp.src.edujudge.audit.sanity_check_exp01 import run_sanity_check
from thesis_exp.src.edujudge.audit.write_exp01_report import write_report


def preflight_check() -> list[dict[str, Any]]:
    ensure_exp01_dirs()
    rows: list[dict[str, Any]] = []

    def add(check: str, status: str, observed: Any, expected: Any, notes: str = "") -> None:
        rows.append({"check": check, "status": status, "observed": observed, "expected": expected, "notes": notes})

    sanity = EXP00_OUTPUT_DIR / "sanity_check_exp00_reference.md"
    add("Exp0.1 sanity check exists", "PASS" if sanity.exists() else "FAIL", relpath(sanity), "exists")
    add(
        "main processed dataset exists",
        "PASS" if PROCESSED_DATASET_PATH.exists() else "FAIL",
        relpath(PROCESSED_DATASET_PATH),
        "exists",
    )
    add("main test split exists", "PASS" if TEST_SPLIT_PATH.exists() else "FAIL", relpath(TEST_SPLIT_PATH), "exists")
    add("main dataset name", "PASS", DATASET_NAME, "edubench_audit_human_scored_subset")

    test_rows = read_jsonl(TEST_SPLIT_PATH) if TEST_SPLIT_PATH.exists() else []
    status = "PASS" if len(test_rows) == EXPECTED_TEST_ROWS else "WARN"
    add(
        "paper-like test set row count",
        status,
        len(test_rows),
        EXPECTED_TEST_ROWS,
        "Do not abort on mismatch; write warning to report.",
    )

    synthetic_candidates = [
        relpath(path) for path in collect_candidate_paths() if is_synthetic_or_sampled_path(path)
    ]
    add(
        "synthetic/sample candidates excluded",
        "PASS",
        synthetic_candidates,
        "excluded from alignment/data source",
        "Inventory may list them as excluded; Exp1 does not use them as prediction sources.",
    )

    write_csv(EXP01_TABLES_DIR / "preflight_check_exp01.csv", rows, ["check", "status", "observed", "expected", "notes"])
    overall = "PASS" if all(row["status"] == "PASS" for row in rows) else "WARN"
    write_text(
        EXP01_OUTPUT_DIR / "preflight_check_exp01.md",
        f"""# Exp1 Preflight Check

Overall status: **{overall}**

{markdown_table(rows, ["check", "status", "observed", "expected", "notes"])}
""",
    )
    return rows


def terminal_summary() -> str:
    coverage = pd.read_csv(EXP01_TABLES_DIR / "alignment_coverage.csv")
    metrics = pd.read_csv(EXP01_TABLES_DIR / "evaluator_metrics.csv")
    per_bin = pd.read_csv(EXP01_TABLES_DIR / "per_bin_metrics.csv")
    low = pd.read_csv(EXP01_TABLES_DIR / "low_score_metrics.csv")
    test_rows = int(coverage["n_test"].iloc[0]) if not coverage.empty else 0
    found = coverage[coverage["n_aligned"] > 0]["evaluator"].tolist()
    missing = coverage[coverage["n_aligned"] == 0]["evaluator"].tolist()

    lines = [
        "Exp1 completed.",
        f"test rows: {test_rows}",
        f"evaluators found: {', '.join(found) if found else 'None'}",
        f"evaluators missing: {', '.join(missing) if missing else 'None'}",
        "coverage per evaluator:",
    ]
    for _, row in coverage.iterrows():
        lines.append(f"  - {row['evaluator']}: {row['coverage']:.3f}")
    lines.append("MAE per evaluator:")
    for _, row in metrics.iterrows():
        lines.append(f"  - {row['evaluator']}: {row['MAE']:.3f}")
    lines.append("Acc@1/2 per evaluator:")
    for evaluator in coverage["evaluator"].tolist():
        pieces = []
        for label in [1, 2]:
            subset = per_bin[(per_bin["evaluator"] == evaluator) & (per_bin["label_5"] == label)]
            if not subset.empty:
                pieces.append(f"Acc@{label}={subset.iloc[0]['accuracy']:.3f}")
        lines.append(f"  - {evaluator}: {', '.join(pieces)}")
    lines.append("low-score overestimation per evaluator:")
    for _, row in low.iterrows():
        lines.append(f"  - {row['evaluator']}: low_to_high_rate={row['low_to_high_rate']:.3f}")
    lines.append(f"output directory: {relpath(EXP01_OUTPUT_DIR)}")
    return "\n".join(lines)


def run_exp01() -> str:
    preflight_check()
    run_inventory()
    run_alignment()
    run_metrics()
    run_plots()
    write_report()
    run_sanity_check()
    summary = terminal_summary()
    print(summary)
    return summary


def main() -> None:
    run_exp01()


if __name__ == "__main__":
    main()

