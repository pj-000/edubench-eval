"""Recover original human rationales for Exp17-D1 audit cases.

This is a lightweight provenance diagnostic. It joins the D1 dev-only cases
back to the original EduBench 5-grade human rationale files by
question/answer/metric. It does not train, load checkpoints, read test labels,
or write large/raw prediction artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_CASES_CSV = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/"
    "d1_hidden_failure_audit_seed42_dev/d1_exact_cases_for_manual_review.csv"
)
DEFAULT_SOURCE_ROOT = Path(".")
DEFAULT_OUT_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/"
    "d1_hidden_failure_audit_seed42_dev/human_rationale_recovery"
)
DEFAULT_REASON_FILES = [
    "5-grades/5_merge_human_metric_en.jsonl",
    "5-grades/5_merge_human_metric_zh.jsonl",
    "5-grades/5_human_1.jsonl",
    "5-grades/5_human_2.jsonl",
    "5-grades/5_human_3.jsonl",
]

METRIC_ALIASES = {
    "Content Relevance & Scope Control": {
        "Content Relevance & Scope Control",
        "内容相关性与范围控制",
    },
    "Domain Knowledge Accuracy": {
        "Domain Knowledge Accuracy",
        "领域知识准确性",
    },
    "Basic Factual Accuracy": {
        "Basic Factual Accuracy",
        "基础事实准确性",
    },
    "Reasoning Process Rigor": {
        "Reasoning Process Rigor",
        "推理过程严谨性",
    },
    "Instruction Following & Task Completion": {
        "Instruction Following & Task Completion",
        "指令遵循与任务完成",
    },
    "Scenario Element Integration": {
        "Scenario Element Integration",
        "场景要素融合度",
        "场景要素整合",
    },
    "Personalization, Adaptation & Learning Support": {
        "Personalization, Adaptation & Learning Support",
        "个性化适配与学习支持",
    },
    "Higher-Order Thinking & Skill Development": {
        "Higher-Order Thinking & Skill Development",
        "促进高阶思维与能力发展",
    },
    "Clarity, Simplicity & Inspiration": {
        "Clarity, Simplicity & Inspiration",
        "清晰易懂与表达启发",
    },
    "Role & Tone Consistency": {
        "Role & Tone Consistency",
        "角色与语气一致性",
    },
    "Motivation, Guidance & Positive Feedback": {
        "Motivation, Guidance & Positive Feedback",
        "鼓励支持与正向反馈",
        "动机引导与正向反馈",
        "激励引导与积极反馈",
    },
    "Error Identification & Correction Precision": {
        "Error Identification & Correction Precision",
        "错误识别与纠正精确性",
    },
}

RECOVERED_FIELDS = [
    "case_no",
    "sample_id",
    "question_group_id",
    "metric",
    "language",
    "subject",
    "gold_label",
    "pred_label",
    "quality_score_s",
    "g_i3",
    "human_1",
    "human_2",
    "human_3",
    "answer_model",
    "match_status",
    "qa_candidate_count",
    "metric_candidate_count",
    "recovered_reason_count",
    "recovered_eval_ids",
    "recovered_scores",
    "recovered_principles",
    "recovered_source_files",
    "human_reason_1",
    "human_reason_2",
    "human_reason_3",
    "human_reason_other",
    "human_reason_summary",
    "question",
    "answer",
    "rubric",
    "metadata",
    "boundary_key",
]

SUMMARY_FIELDS = [
    "group_key",
    "n_cases",
    "recovered_cases",
    "recovered_rate",
    "metrics",
    "case_nos",
    "dominant_reason_summary",
]


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "\n".join(line.rstrip() for line in str(value).strip().splitlines())


def norm_ws(value: Any) -> str:
    return re.sub(r"\s+", " ", clean_text(value)).strip()


def norm_alnum(value: Any) -> str:
    return re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]", "", norm_ws(value))


def short_text(value: Any, max_len: int = 220) -> str:
    text = norm_ws(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def reason_files(source_root: Path, rel_paths: list[str]) -> list[Path]:
    files = [source_root / rel for rel in rel_paths]
    return [path for path in files if path.exists()]


def load_reason_rows(source_root: Path, rel_paths: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for rel in rel_paths:
        path = source_root / rel
        if not path.exists():
            missing.append(rel)
            continue
        for row in read_jsonl(path):
            row["_source_file"] = rel
            rows.append(row)
    return rows, missing


def build_indexes(rows: list[dict[str, Any]]) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[tuple[str, str], list[dict[str, Any]]]]:
    exact: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    alnum: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        q = norm_ws(row.get("question", ""))
        a = norm_ws(row.get("response", ""))
        exact[(q, a)].append(row)
        alnum[(q, norm_alnum(row.get("response", "")))].append(row)
    return exact, alnum


def metric_aliases(metric: str) -> set[str]:
    aliases = set(METRIC_ALIASES.get(metric, {metric}))
    aliases.add(metric)
    return aliases


def preferred_metric_rows(candidates: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    aliases = metric_aliases(metric)
    metric_rows = [row for row in candidates if clean_text(row.get("principle")) in aliases]
    merged = [row for row in metric_rows if "5_merge_human_metric" in clean_text(row.get("_source_file"))]
    return merged or metric_rows


def rater_id(row: dict[str, Any]) -> str:
    eval_id = clean_text(row.get("eval"))
    match = re.search(r"human[_-]?([123])", eval_id)
    if match:
        return match.group(1)
    source = clean_text(row.get("_source_file"))
    match = re.search(r"5_human_([123])", source)
    if match:
        return match.group(1)
    return "other"


def summarize_reasons(rows: list[dict[str, Any]]) -> tuple[dict[str, str], str]:
    by_rater: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        reason = clean_text(row.get("reason"))
        if reason:
            by_rater[rater_id(row)].append(reason)

    out = {
        "human_reason_1": " | ".join(dict.fromkeys(by_rater.get("1", []))),
        "human_reason_2": " | ".join(dict.fromkeys(by_rater.get("2", []))),
        "human_reason_3": " | ".join(dict.fromkeys(by_rater.get("3", []))),
        "human_reason_other": " | ".join(dict.fromkeys(by_rater.get("other", []))),
    }
    unique_reasons = []
    for row in rows:
        reason = clean_text(row.get("reason"))
        if reason and reason not in unique_reasons:
            unique_reasons.append(reason)
    summary = " / ".join(short_text(reason, 110) for reason in unique_reasons[:3])
    return out, summary


def recover_cases(cases: list[dict[str, str]], reason_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    exact_index, alnum_index = build_indexes(reason_rows)
    recovered: list[dict[str, Any]] = []

    for case in cases:
        question = norm_ws(case.get("question", ""))
        answer = norm_ws(case.get("answer", ""))
        candidates = exact_index.get((question, answer), [])
        if not candidates:
            candidates = alnum_index.get((question, norm_alnum(answer)), [])

        metric_rows = preferred_metric_rows(candidates, clean_text(case.get("metric")))
        if metric_rows:
            match_status = "metric_rationale_recovered"
        elif candidates:
            match_status = "question_answer_matched_metric_unmatched"
        else:
            match_status = "question_answer_unmatched"

        reason_cols, reason_summary = summarize_reasons(metric_rows)
        eval_ids = [clean_text(row.get("eval")) or rater_id(row) for row in metric_rows]
        scores = [clean_text(row.get("score")) for row in metric_rows]
        principles = [clean_text(row.get("principle")) for row in metric_rows]
        source_files = [clean_text(row.get("_source_file")) for row in metric_rows]

        out: dict[str, Any] = dict(case)
        out.update(reason_cols)
        out.update(
            {
                "match_status": match_status,
                "qa_candidate_count": len(candidates),
                "metric_candidate_count": len(metric_rows),
                "recovered_reason_count": sum(1 for row in metric_rows if clean_text(row.get("reason"))),
                "recovered_eval_ids": ";".join(dict.fromkeys(eval_ids)),
                "recovered_scores": ";".join(dict.fromkeys(scores)),
                "recovered_principles": ";".join(dict.fromkeys(principles)),
                "recovered_source_files": ";".join(dict.fromkeys(source_files)),
                "human_reason_summary": reason_summary,
            }
        )
        recovered.append(out)
    return recovered


def rate(num: int, den: int) -> str:
    return f"{(num / den if den else 0.0):.4f}"


def group_summary(rows: list[dict[str, Any]], key_field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[clean_text(row.get(key_field))].append(row)
    out: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        recovered_items = [row for row in items if row.get("match_status") == "metric_rationale_recovered"]
        summaries = [clean_text(row.get("human_reason_summary")) for row in recovered_items if clean_text(row.get("human_reason_summary"))]
        dominant = Counter(summaries).most_common(1)[0][0] if summaries else ""
        out.append(
            {
                "group_key": key,
                "n_cases": len(items),
                "recovered_cases": len(recovered_items),
                "recovered_rate": rate(len(recovered_items), len(items)),
                "metrics": ";".join(sorted({clean_text(row.get("metric")) for row in items})),
                "case_nos": ";".join(clean_text(row.get("case_no")) for row in items),
                "dominant_reason_summary": dominant,
            }
        )
    return out


def write_report(
    path: Path,
    recovered: list[dict[str, Any]],
    source_root: Path,
    rel_paths: list[str],
    missing_files: list[str],
) -> None:
    total = len(recovered)
    recovered_count = sum(row.get("match_status") == "metric_rationale_recovered" for row in recovered)
    qa_matched = sum(row.get("match_status") != "question_answer_unmatched" for row in recovered)
    status_counts = Counter(clean_text(row.get("match_status")) for row in recovered)
    unmatched_cases = [clean_text(row.get("case_no")) for row in recovered if row.get("match_status") != "metric_rationale_recovered"]
    group_counts = Counter(clean_text(row.get("question_group_id")) for row in recovered)

    lines = [
        "# Exp17-D1 Human Rationale Recovery",
        "",
        "This is a dev-only provenance diagnostic. It only recovers original 5-grade human rating rationales for the existing D1 cases. It does not train, load checkpoints, read test labels, or write raw predictions.",
        "",
        "## Source",
        "",
        f"- Source root: `{source_root}`",
        "- Current fork: `https://github.com/pj-000/edubench-eval`",
        "- Upstream source lineage: `https://github.com/danieglofsmi/edubench-eval/tree/main`",
        "- Reason files:",
        *[f"  - `{rel}`" for rel in rel_paths],
        f"- Missing reason files: `{'; '.join(missing_files) if missing_files else 'none'}`",
        "",
        "## Recovery Summary",
        "",
        f"- D1 cases: `{total}`",
        f"- Question-answer matched cases: `{qa_matched}/{total}` = `{rate(qa_matched, total)}`",
        f"- Metric-level rationale recovered cases: `{recovered_count}/{total}` = `{rate(recovered_count, total)}`",
        f"- Non-recovered case numbers: `{'; '.join(unmatched_cases) if unmatched_cases else 'none'}`",
        "",
        "| match_status | n |",
        "|---|---:|",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"| {status} | {count} |")

    lines.extend(
        [
            "",
            "## Concentration",
            "",
            "| question_group_id | n |",
            "|---|---:|",
        ]
    )
    for group_id, count in group_counts.most_common():
        lines.append(f"| `{group_id}` | {count} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The earlier D1 audit used the merged modeling table, which keeps human scores but not the original human rationale text.",
            "- The original 5-grade files contain rationale text for most D1 cases. Therefore, several apparently unexplained label-2 cases should be re-interpreted with these recovered rationales before deciding whether they are label conflicts.",
            "- The Annales cases are not merely arbitrary label conflicts in the recovered rationales: the human reasons treat `Marc Bloch` as a wrong answer for the long-duration-history wording and point to `Fernand Braudel` as the intended figure.",
            "- The marketing-manager cases mostly have recoverable rubric-linked reasons: incomplete corrected answer, missing key duties, shallow error explanation, weak scenario adaptation, or weak clarity/inspiration.",
            "- Case 23 should be handled carefully: recovered human rationales emphasize task/format and scoring-design mismatch in the answer, while external expert review may additionally identify domain factual mismatch. These should be separated rather than collapsed into one label.",
            "",
            "## Next Step",
            "",
            "Before training Exp17-A, rerun the D1 annotation summary using this recovered rationale table as evidence. Do not train directly from dev annotations; use recovered rationales to design train-side weak-label expansion or pairwise evidence checks.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-csv", type=Path, default=DEFAULT_CASES_CSV)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--reason-file",
        action="append",
        dest="reason_files",
        default=None,
        help="Relative path under --source-root. Can be repeated. Defaults to 5-grade human rationale files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.cases_csv.exists():
        raise FileNotFoundError(f"Missing D1 cases CSV: {args.cases_csv}")
    if not args.source_root.exists():
        raise FileNotFoundError(
            f"Missing source root: {args.source_root}. Clone https://github.com/danieglofsmi/edubench-eval first "
            "or pass --source-root."
        )

    rel_paths = args.reason_files or DEFAULT_REASON_FILES
    if not reason_files(args.source_root, rel_paths):
        raise FileNotFoundError(f"No reason files found under {args.source_root}")

    cases = read_csv(args.cases_csv)
    reason_rows, missing_files = load_reason_rows(args.source_root, rel_paths)
    recovered = recover_cases(cases, reason_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "d1_human_rationale_recovered.csv", recovered, RECOVERED_FIELDS)
    write_csv(
        args.out_dir / "d1_human_rationale_by_question_group.csv",
        group_summary(recovered, "question_group_id"),
        SUMMARY_FIELDS,
    )
    write_csv(
        args.out_dir / "d1_human_rationale_by_metric.csv",
        group_summary(recovered, "metric"),
        SUMMARY_FIELDS,
    )
    write_report(
        args.out_dir / "d1_human_rationale_recovery_report.md",
        recovered,
        args.source_root,
        rel_paths,
        missing_files,
    )

    total = len(recovered)
    recovered_count = sum(row.get("match_status") == "metric_rationale_recovered" for row in recovered)
    print(f"Recovered metric-level rationales: {recovered_count}/{total}")
    print(f"Wrote: {args.out_dir}")


if __name__ == "__main__":
    main()
