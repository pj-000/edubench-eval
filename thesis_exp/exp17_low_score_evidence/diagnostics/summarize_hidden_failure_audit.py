"""Summarize filled Exp17-D1 hidden failure audit annotations.

The script validates human annotation labels and writes lightweight CSV/JSON/MD
summaries. It does not read test data or train/load any model.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from failure_taxonomy import (
        PRIMARY_FAILURE_MODES,
        RECOMMENDED_TRAINING_USE,
        RUBRIC_LINK_LEVELS,
        TRAINABILITY_OPTIONS,
        normalize_label_string,
        validate_annotation_row,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from failure_taxonomy import (  # type: ignore
        PRIMARY_FAILURE_MODES,
        RECOMMENDED_TRAINING_USE,
        RUBRIC_LINK_LEVELS,
        TRAINABILITY_OPTIONS,
        normalize_label_string,
        validate_annotation_row,
    )


RUBRIC_LINKED_LEVELS = {
    "explicit_rubric_clause",
    "implicit_task_constraint",
    "inferred_from_context",
}

STRONG_OR_WEAK_TRAIN = {"strong_train_signal", "weak_train_signal"}

TRAINABILITY_ACTIONS = {
    "strong_train_signal": "Use as evidence-positive weak supervision after train-side expansion.",
    "weak_train_signal": "Use with lower weight or as auxiliary evidence signal.",
    "format_auxiliary_signal": "Use for format/task-constraint auxiliary supervision.",
    "pairwise_only": "Use in matched hard-negative or pairwise separation only.",
    "downweight_or_exclude": "Downweight or exclude from evidence-positive labels.",
    "review_only": "Keep for qualitative analysis; do not train from it.",
    "unclear": "Needs more review before training use.",
}

QUESTION_GROUP_FIELDS = [
    "question_group_id",
    "metric",
    "n_cases",
    "dominant_failure_mode",
    "dominant_failure_mode_rate",
    "rubric_linked_rate",
    "hidden_failure_rate",
    "possible_label_conflict_rate",
    "strong_or_weak_train_signal_rate",
    "recommended_group_use",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def norm(row: dict[str, Any], field: str, default: str = "unclear") -> str:
    value = normalize_label_string(row.get(field, ""))
    return value or default


def boolish(row: dict[str, Any], field: str) -> bool:
    value = normalize_label_string(row.get(field, ""))
    return value in {"1", "yes", "true", "y"}


def rate(num: int, den: int) -> float:
    return num / den if den else 0.0


def fmt(value: float) -> str:
    return f"{value:.4f}"


def validation_issues(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=2):
        for issue in validate_annotation_row(row):
            issues.append(
                {
                    "csv_line": idx,
                    "sample_id": row.get("sample_id", ""),
                    "issue": issue,
                }
            )
    return issues


def failure_mode_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    total = len(rows)
    for mode in PRIMARY_FAILURE_MODES:
        items = [row for row in rows if norm(row, "primary_failure_mode_manual") == mode]
        n = len(items)
        rubric_linked = sum(norm(row, "rubric_link_level_manual") in RUBRIC_LINKED_LEVELS for row in items)
        hidden = sum(boolish(row, "is_hidden_failure_manual") for row in items)
        conflicts = sum(boolish(row, "possible_label_conflict_manual") for row in items)
        train_counter = Counter(norm(row, "trainability_manual") for row in items)
        out.append(
            {
                "failure_mode": mode,
                "n": n,
                "rate": fmt(rate(n, total)),
                "rubric_linked_n": rubric_linked,
                "rubric_linked_rate": fmt(rate(rubric_linked, n)),
                "hidden_failure_n": hidden,
                "hidden_failure_rate": fmt(rate(hidden, n)),
                "possible_label_conflict_n": conflicts,
                "possible_label_conflict_rate": fmt(rate(conflicts, n)),
                "strong_train_signal_n": train_counter.get("strong_train_signal", 0),
                "weak_train_signal_n": train_counter.get("weak_train_signal", 0),
                "format_auxiliary_signal_n": train_counter.get("format_auxiliary_signal", 0),
                "pairwise_only_n": train_counter.get("pairwise_only", 0),
                "downweight_or_exclude_n": train_counter.get("downweight_or_exclude", 0),
                "review_only_n": train_counter.get("review_only", 0),
            }
        )
    return out


def trainability_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    total = len(rows)
    counts = Counter(norm(row, "trainability_manual") for row in rows)
    out: list[dict[str, Any]] = []
    for option in TRAINABILITY_OPTIONS:
        n = counts.get(option, 0)
        out.append(
            {
                "trainability_manual": option,
                "n": n,
                "rate": fmt(rate(n, total)),
                "recommended_action": TRAINABILITY_ACTIONS.get(option, "Review manually."),
            }
        )
    return out


def recommended_group_use(items: list[dict[str, str]]) -> str:
    conflict_rate = rate(sum(boolish(row, "possible_label_conflict_manual") for row in items), len(items))
    train_rate = rate(sum(norm(row, "trainability_manual") in STRONG_OR_WEAK_TRAIN for row in items), len(items))
    format_rate = rate(sum(norm(row, "trainability_manual") == "format_auxiliary_signal" for row in items), len(items))
    if conflict_rate >= 0.5:
        return "review_or_downweight"
    if train_rate >= 0.5:
        return "evidence_positive_candidate"
    if format_rate >= 0.5:
        return "format_auxiliary_candidate"
    return "pairwise_or_review"


def question_group_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("question_group_id", ""), row.get("metric", ""))].append(row)
    out: list[dict[str, Any]] = []
    for (group_id, metric), items in sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        mode_counts = Counter(norm(row, "primary_failure_mode_manual") for row in items)
        dominant_mode, dominant_n = mode_counts.most_common(1)[0] if mode_counts else ("unclear", 0)
        n = len(items)
        rubric_linked = sum(norm(row, "rubric_link_level_manual") in RUBRIC_LINKED_LEVELS for row in items)
        hidden = sum(boolish(row, "is_hidden_failure_manual") for row in items)
        conflicts = sum(boolish(row, "possible_label_conflict_manual") for row in items)
        train = sum(norm(row, "trainability_manual") in STRONG_OR_WEAK_TRAIN for row in items)
        out.append(
            {
                "question_group_id": group_id,
                "metric": metric,
                "n_cases": n,
                "dominant_failure_mode": dominant_mode,
                "dominant_failure_mode_rate": fmt(rate(dominant_n, n)),
                "rubric_linked_rate": fmt(rate(rubric_linked, n)),
                "hidden_failure_rate": fmt(rate(hidden, n)),
                "possible_label_conflict_rate": fmt(rate(conflicts, n)),
                "strong_or_weak_train_signal_rate": fmt(rate(train, n)),
                "recommended_group_use": recommended_group_use(items),
            }
        )
    return out


def decision(rows: list[dict[str, str]], validation_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    rubric_linked_hidden = sum(
        norm(row, "rubric_link_level_manual") in RUBRIC_LINKED_LEVELS
        and boolish(row, "is_hidden_failure_manual")
        for row in rows
    )
    conflicts = sum(boolish(row, "possible_label_conflict_manual") for row in rows)
    trainable = sum(norm(row, "trainability_manual") in STRONG_OR_WEAK_TRAIN for row in rows)
    group_counts = Counter(row.get("question_group_id", "") for row in rows)
    max_group_rate = rate(group_counts.most_common(1)[0][1], total) if total and group_counts else 0.0
    rubric_linked_hidden_rate = rate(rubric_linked_hidden, total)
    conflict_rate = rate(conflicts, total)
    trainable_rate = rate(trainable, total)
    enter = (
        rubric_linked_hidden_rate >= 0.60
        and conflict_rate <= 0.35
        and trainable_rate >= 0.50
        and not validation_rows
    )
    reasons = []
    if rubric_linked_hidden_rate < 0.60:
        reasons.append("rubric_linked_hidden_failure_rate < 0.60")
    if conflict_rate > 0.35:
        reasons.append("possible_label_conflict_rate > 0.35")
    if trainable_rate < 0.50:
        reasons.append("strong_or_weak_train_signal_rate < 0.50")
    if validation_rows:
        reasons.append("annotation validation issues exist")
    if max_group_rate >= 0.70:
        reasons.append(
            "WARNING: Failure cases are highly concentrated in one question group; Exp17-A must use train-side weak labels and should avoid question_key-specific features."
        )
    return {
        "total_cases": total,
        "rubric_linked_hidden_failure_rate": rubric_linked_hidden_rate,
        "possible_label_conflict_rate": conflict_rate,
        "strong_or_weak_train_signal_rate": trainable_rate,
        "max_question_group_rate": max_group_rate,
        "enter_exp17a_recommendation": enter,
        "recommendation_reason": "; ".join(reasons) if reasons else "All default criteria satisfied.",
    }


def case_control_notes(case_control_csv: Path | None) -> dict[str, Any]:
    if not case_control_csv or not case_control_csv.exists():
        return {"provided": False, "rows": 0, "manual_note_rows": 0}
    rows = read_csv(case_control_csv)
    note_rows = sum(1 for row in rows if row.get("manual_difference_notes", "").strip())
    return {"provided": True, "rows": len(rows), "manual_note_rows": note_rows}


def write_report(
    path: Path,
    rows: list[dict[str, str]],
    failures: list[dict[str, Any]],
    trainability: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    decision_data: dict[str, Any],
    validation_rows: list[dict[str, Any]],
    case_control: dict[str, Any],
) -> None:
    lines = [
        "# Exp17-D1 Hidden Failure Audit Summary",
        "",
        "This is a dev-only diagnostic summary. It does not train a model, read test data, or generate checkpoints.",
        "",
        "## 1. Summary of Annotated Cases",
        "",
        f"- Total annotated cases: `{len(rows)}`",
        f"- Validation issues: `{len(validation_rows)}`",
        "",
        "## 2. Failure Mode Distribution",
        "",
        "| failure_mode | n | rate | rubric linked rate | hidden failure rate | conflict rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in failures:
        if int(row["n"]) == 0:
            continue
        lines.append(
            f"| {row['failure_mode']} | {row['n']} | {row['rate']} | {row['rubric_linked_rate']} | "
            f"{row['hidden_failure_rate']} | {row['possible_label_conflict_rate']} |"
        )
    lines += [
        "",
        "## 3. Rubric Linkage Analysis",
        "",
        f"- Rubric-linked hidden failure rate: `{decision_data['rubric_linked_hidden_failure_rate']:.4f}`",
        "",
        "## 4. Hidden Failure vs Visible Defect Analysis",
        "",
        "Use `is_hidden_failure_manual` together with the primary failure mode to separate surface-fluent hidden failures from visible format or task violations.",
        "",
        "## 5. Possible Label Conflict Analysis",
        "",
        f"- Possible label conflict rate: `{decision_data['possible_label_conflict_rate']:.4f}`",
        "",
        "## 6. Question Group Concentration Analysis",
        "",
        f"- Max question group rate: `{decision_data['max_question_group_rate']:.4f}`",
        "",
        "## 7. Trainability Analysis",
        "",
        "| trainability | n | rate | recommended action |",
        "|---|---:|---:|---|",
    ]
    for row in trainability:
        if int(row["n"]) == 0:
            continue
        lines.append(f"| {row['trainability_manual']} | {row['n']} | {row['rate']} | {row['recommended_action']} |")
    lines += [
        "",
        "## 8. Case-Control Comparison Notes",
        "",
        f"- Case-control CSV provided: `{case_control['provided']}`",
        f"- Case-control rows: `{case_control['rows']}`",
        f"- Rows with manual notes: `{case_control['manual_note_rows']}`",
        "",
        "## 9. Decision",
        "",
        f"- Enter Exp17-A recommendation: `{decision_data['enter_exp17a_recommendation']}`",
        f"- Reason: {decision_data['recommendation_reason']}",
        "",
        "## 10. Recommended Exp17-A Supervision Design",
        "",
        "- Prefer a scalar hidden failure score before multi-class defect type prediction.",
        "- Do not train multi-class defect type unless enough train-side weak labels exist.",
        "- Use human-agreement-weighted weak supervision.",
        "- Ignore or downweight possible label conflict cases.",
        "- Use matched hard negatives only after D1 confirms trainable hidden failure patterns.",
        "",
        "## 11. Leakage Statement",
        "",
        "- Dev only.",
        "- No test read.",
        "- Dev annotations are not used directly as train labels.",
        "- No checkpoint generated.",
    ]
    if validation_rows:
        lines += [
            "",
            "## Annotation Validation Issues",
            "",
            "See `d1_annotation_validation_issues.csv` before making an Exp17-A decision.",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_csv(args.annotated_csv)
    validation_rows = validation_issues(rows)
    failures = failure_mode_summary(rows)
    trainability = trainability_summary(rows)
    groups = question_group_summary(rows)
    decision_data = decision(rows, validation_rows)
    case_control = case_control_notes(args.case_control_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.out_dir / "d1_failure_mode_summary.csv",
        failures,
        [
            "failure_mode",
            "n",
            "rate",
            "rubric_linked_n",
            "rubric_linked_rate",
            "hidden_failure_n",
            "hidden_failure_rate",
            "possible_label_conflict_n",
            "possible_label_conflict_rate",
            "strong_train_signal_n",
            "weak_train_signal_n",
            "format_auxiliary_signal_n",
            "pairwise_only_n",
            "downweight_or_exclude_n",
            "review_only_n",
        ],
    )
    write_csv(args.out_dir / "d1_trainability_summary.csv", trainability, ["trainability_manual", "n", "rate", "recommended_action"])
    write_csv(args.out_dir / "d1_question_group_failure_summary.csv", groups, QUESTION_GROUP_FIELDS)
    if validation_rows:
        write_csv(args.out_dir / "d1_annotation_validation_issues.csv", validation_rows, ["csv_line", "sample_id", "issue"])
    (args.out_dir / "d1_enter_exp17a_decision.json").write_text(
        json.dumps(decision_data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(
        args.out_dir / "exp17_d1_hidden_failure_report.md",
        rows,
        failures,
        trainability,
        groups,
        decision_data,
        validation_rows,
        case_control,
    )
    return {
        "total_cases": len(rows),
        "validation_issues": len(validation_rows),
        "enter_exp17a_recommendation": decision_data["enter_exp17a_recommendation"],
        "out_dir": str(args.out_dir),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize filled Exp17-D1 hidden failure annotations.")
    parser.add_argument("--annotated-csv", required=True, type=Path)
    parser.add_argument("--case-control-csv", type=Path, default=None)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--split", default="dev")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
