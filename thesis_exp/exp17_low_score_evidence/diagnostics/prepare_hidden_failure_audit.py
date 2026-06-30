"""Prepare Exp17-D1 hidden failure audit templates.

The script reads Exp17-D0 lightweight CSV outputs and writes annotation-ready
CSV/MD artifacts. It does not train, load checkpoints, import transformers, or
read test data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ANNOTATION_FIELDS = [
    "primary_failure_mode_manual",
    "secondary_failure_mode_manual",
    "rubric_link_level_manual",
    "is_surface_fluent_manual",
    "is_hidden_failure_manual",
    "is_format_or_task_constraint_manual",
    "possible_label_conflict_manual",
    "llm_or_model_over_scoring_pattern_manual",
    "rubric_clause_manual",
    "evidence_span_manual",
    "defect_notes_manual",
    "confidence_manual",
    "trainability_manual",
    "recommended_training_use_manual",
]

TEMPLATE_FIELDS = [
    "sample_id",
    "question_key",
    "question_group_id",
    "metric",
    "language",
    "subject",
    "gold_label",
    "pred_label",
    "quality_score_s",
    "tau2",
    "tau3",
    "tau4",
    "scale_alpha",
    "g_i3",
    "human_1",
    "human_2",
    "human_3",
    "human_agreement_pattern",
    "llm_judge_summary",
    "question",
    "answer",
    "rubric",
    "metadata",
    "boundary_key",
    "rubric_hash",
    "answer_length",
    "matched_control_ids",
    "auto_question_group_size",
    "auto_case_rank_within_group",
    "auto_has_json_requirement",
    "auto_answer_json_parse_ok",
    "auto_answer_has_extra_explanation",
    "auto_format_requirement_keywords",
    "auto_possible_format_violation",
    "auto_train_low_count_same_metric",
    "auto_train_high_count_same_metric",
    "auto_train_low_count_same_rubric_hash",
    "auto_train_high_count_same_rubric_hash",
    "auto_train_support_note",
    *ANNOTATION_FIELDS,
]

QUESTION_GROUP_FIELDS = [
    "question_group_id",
    "metric",
    "language",
    "subject",
    "rubric_hash",
    "boundary_key",
    "n_l2h_cases",
    "sample_ids",
    "question_key_examples",
    "mean_s",
    "mean_g_i3",
    "human_agreement_patterns",
    "pred_label_distribution",
    "auto_has_json_requirement_rate",
    "auto_possible_format_violation_rate",
    "train_low_count_same_metric",
    "train_high_count_same_metric",
    "train_low_count_same_rubric_hash",
    "train_high_count_same_rubric_hash",
    "dominant_failure_mode_manual",
    "rubric_link_rate_manual",
    "possible_label_conflict_rate_manual",
    "trainability_summary_manual",
]

CASE_CONTROL_FIELDS = [
    "case_sample_id",
    "control_sample_id",
    "match_rank",
    "match_score",
    "same_metric",
    "same_language",
    "same_subject",
    "same_boundary_key",
    "same_rubric_hash",
    "case_gold_label",
    "case_pred_label",
    "control_gold_label",
    "control_pred_label",
    "case_s",
    "control_s",
    "s_gap_control_minus_case",
    "case_g_i3",
    "control_g_i3",
    "g_i3_gap_control_minus_case",
    "question",
    "rubric",
    "case_answer",
    "control_answer",
    "manual_difference_notes",
    "does_control_satisfy_missing_constraint_manual",
    "case_hidden_failure_relative_to_control_manual",
]

FORMAT_KEYWORDS = [
    "json",
    "json format",
    "in json format",
    "以json格式",
    "format",
    "输出格式",
    "return",
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "\n".join(line.rstrip() for line in str(value).strip().splitlines())


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def safe_int(value: Any) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def fmt_float(value: float, digits: int = 4) -> str:
    try:
        if value != value:
            return ""
        return f"{value:.{digits}f}"
    except Exception:
        return ""


def stable_hash(value: Any) -> str:
    text = clean_text(value)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def row_id(row: dict[str, Any]) -> str:
    return clean_text(row.get("record_id") or row.get("sample_id") or row.get("id"))


def label_value(row: dict[str, Any]) -> int:
    for key in ["label_5", "gold_label", "human_mean_5"]:
        if key in row and clean_text(row.get(key)):
            return safe_int(row.get(key))
    return 0


def merge_dev_fields(cases: list[dict[str, Any]], dev_jsonl: Path | None) -> tuple[list[dict[str, Any]], str]:
    if not dev_jsonl:
        return cases, "d0_cases_only"
    raw_rows = read_jsonl(dev_jsonl)
    raw_by_id = {row_id(row): row for row in raw_rows}
    merged: list[dict[str, Any]] = []
    missing = 0
    for idx, case in enumerate(cases):
        sample_id = clean_text(case.get("sample_id"))
        raw = raw_by_id.get(sample_id)
        join_mode = "sample_id"
        if raw is None and idx < len(raw_rows):
            raw = raw_rows[idx]
            join_mode = "row_index_fallback"
        if raw is None:
            missing += 1
            raw = {}
        out = dict(case)
        for key in [
            "question",
            "answer",
            "rubric",
            "metadata",
            "language",
            "subject",
            "human_1",
            "human_2",
            "human_3",
            "judge_scores",
            "metric_canonical",
            "metric_raw",
            "subject_canonical",
            "metadata_raw",
        ]:
            if not clean_text(out.get(key)) and key in raw:
                out[key] = raw.get(key)
        if not clean_text(out.get("rubric")):
            out["rubric"] = raw.get("rubric_text") or raw.get("rubric_canonical") or ""
        if not clean_text(out.get("metadata")):
            out["metadata"] = raw.get("metadata_raw") or raw.get("metadata") or ""
        if not clean_text(out.get("subject")):
            out["subject"] = raw.get("subject_canonical") or raw.get("subject_raw") or ""
        if not clean_text(out.get("metric")):
            out["metric"] = raw.get("metric_canonical") or raw.get("metric_raw") or ""
        out["_join_mode"] = join_mode
        merged.append(out)
    if any(row.get("_join_mode") == "row_index_fallback" for row in merged):
        return merged, f"sample_id_then_row_index_fallback; missing={missing}"
    return merged, f"sample_id; missing={missing}"


def human_agreement_pattern(row: dict[str, Any]) -> str:
    values = [clean_text(row.get(key)) for key in ["human_1", "human_2", "human_3"]]
    values = [value for value in values if value]
    return "/".join(values)


def llm_judge_summary(row: dict[str, Any]) -> str:
    judge_scores = row.get("judge_scores")
    if isinstance(judge_scores, str) and judge_scores.strip().startswith("{"):
        try:
            judge_scores = json.loads(judge_scores.replace("'", '"'))
        except Exception:
            return judge_scores
    if isinstance(judge_scores, dict):
        return "; ".join(f"{key}={value}" for key, value in sorted(judge_scores.items()))
    pieces = []
    for key, value in row.items():
        lower = key.lower()
        if "judge" in lower or key in {"EduBenchEvaluator", "gpt-4o", "deepseek-r1", "deepseek-v3", "qwq-plus"}:
            if clean_text(value):
                pieces.append(f"{key}={value}")
    return "; ".join(pieces)


def detect_format_keywords(question: str, rubric: str, metadata: str) -> list[str]:
    haystack = f"{question}\n{rubric}\n{metadata}".lower()
    return [keyword for keyword in FORMAT_KEYWORDS if keyword in haystack]


def extract_json_candidate(answer: str) -> tuple[str, str, str]:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", answer, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip(), answer[: fenced.start()].strip(), answer[fenced.end() :].strip()
    start_positions = [pos for pos in [answer.find("{"), answer.find("[")] if pos >= 0]
    if not start_positions:
        return "", answer.strip(), ""
    start = min(start_positions)
    end = max(answer.rfind("}"), answer.rfind("]"))
    if end < start:
        return "", answer.strip(), ""
    return answer[start : end + 1].strip(), answer[:start].strip(), answer[end + 1 :].strip()


def json_parse_ok(answer: str) -> tuple[str, str]:
    candidate, before, after = extract_json_candidate(answer)
    if not candidate:
        return "no", "yes" if answer.strip() else "no"
    try:
        json.loads(candidate)
        parse_ok = "yes"
    except Exception:
        parse_ok = "no"
    outside = clean_text(f"{before}\n{after}")
    has_extra = "yes" if outside else "no"
    return parse_ok, has_extra


def count_train_support(train_jsonl: Path | None) -> tuple[dict[str, Counter], dict[str, Counter], str]:
    metric_counts: dict[str, Counter] = defaultdict(Counter)
    rubric_counts: dict[str, Counter] = defaultdict(Counter)
    if not train_jsonl:
        return metric_counts, rubric_counts, "train support not computed"
    for row in read_jsonl(train_jsonl):
        label = label_value(row)
        bucket = "low" if label <= 2 else "high" if label >= 4 else "mid"
        metric = clean_text(row.get("metric_canonical") or row.get("metric") or row.get("metric_raw"))
        rubric_hash = stable_hash(row.get("rubric") or row.get("rubric_text") or row.get("rubric_canonical"))
        if metric:
            metric_counts[metric][bucket] += 1
        if rubric_hash:
            rubric_counts[rubric_hash][bucket] += 1
    return metric_counts, rubric_counts, f"computed from {train_jsonl}"


def add_auto_fields(
    rows: list[dict[str, Any]],
    controls: list[dict[str, str]],
    train_metric_counts: dict[str, Counter],
    train_rubric_counts: dict[str, Counter],
    train_note: str,
) -> list[dict[str, Any]]:
    group_counts = Counter(clean_text(row.get("question_key")) or stable_hash(row.get("question")) for row in rows)
    group_ranks: dict[str, int] = defaultdict(int)
    control_ids: dict[str, list[str]] = defaultdict(list)
    for control in controls:
        control_ids[clean_text(control.get("case_sample_id"))].append(clean_text(control.get("control_sample_id")))
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        question = clean_text(item.get("question"))
        answer = clean_text(item.get("answer"))
        rubric = clean_text(item.get("rubric"))
        metadata = clean_text(item.get("metadata"))
        question_group_id = clean_text(item.get("question_key")) or stable_hash(question)
        group_ranks[question_group_id] += 1
        keywords = detect_format_keywords(question, rubric, metadata)
        parse_ok, has_extra = json_parse_ok(answer)
        has_json = "yes" if any("json" in keyword for keyword in keywords) else "no"
        possible_format = "yes" if has_json == "yes" and (parse_ok != "yes" or has_extra == "yes") else "no"
        metric = clean_text(item.get("metric"))
        rubric_hash = clean_text(item.get("rubric_hash")) or stable_hash(rubric)
        metric_counter = train_metric_counts.get(metric, Counter())
        rubric_counter = train_rubric_counts.get(rubric_hash, Counter())
        item.update(
            {
                "sample_id": clean_text(item.get("sample_id")),
                "question_key": clean_text(item.get("question_key")),
                "question_group_id": question_group_id,
                "metric": metric,
                "language": clean_text(item.get("language")),
                "subject": clean_text(item.get("subject")),
                "question": question,
                "answer": answer,
                "rubric": rubric,
                "metadata": metadata,
                "rubric_hash": rubric_hash,
                "human_agreement_pattern": human_agreement_pattern(item),
                "llm_judge_summary": llm_judge_summary(item),
                "matched_control_ids": "|".join(control_ids.get(clean_text(item.get("sample_id")), [])),
                "auto_question_group_size": group_counts[question_group_id],
                "auto_case_rank_within_group": group_ranks[question_group_id],
                "auto_has_json_requirement": has_json,
                "auto_answer_json_parse_ok": parse_ok,
                "auto_answer_has_extra_explanation": has_extra,
                "auto_format_requirement_keywords": "|".join(keywords),
                "auto_possible_format_violation": possible_format,
                "auto_train_low_count_same_metric": metric_counter.get("low", ""),
                "auto_train_high_count_same_metric": metric_counter.get("high", ""),
                "auto_train_low_count_same_rubric_hash": rubric_counter.get("low", ""),
                "auto_train_high_count_same_rubric_hash": rubric_counter.get("high", ""),
                "auto_train_support_note": train_note,
            }
        )
        for field in ANNOTATION_FIELDS:
            item[field] = ""
        out.append(item)
    return out


def group_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            clean_text(row.get("question_group_id")),
            clean_text(row.get("metric")),
            clean_text(row.get("rubric_hash")),
        )
        grouped[key].append(row)
    out: list[dict[str, Any]] = []
    for (group_id, metric, rubric_hash), items in sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        n = len(items)
        out.append(
            {
                "question_group_id": group_id,
                "metric": metric,
                "language": items[0].get("language", ""),
                "subject": items[0].get("subject", ""),
                "rubric_hash": rubric_hash,
                "boundary_key": items[0].get("boundary_key", ""),
                "n_l2h_cases": n,
                "sample_ids": "|".join(clean_text(row.get("sample_id")) for row in items),
                "question_key_examples": "|".join(sorted({clean_text(row.get("question_key")) for row in items if row.get("question_key")})),
                "mean_s": fmt_float(mean([safe_float(row.get("quality_score_s")) for row in items])),
                "mean_g_i3": fmt_float(mean([safe_float(row.get("g_i3")) for row in items])),
                "human_agreement_patterns": "|".join(sorted({clean_text(row.get("human_agreement_pattern")) for row in items if row.get("human_agreement_pattern")})),
                "pred_label_distribution": json.dumps(Counter(clean_text(row.get("pred_label")) for row in items), sort_keys=True),
                "auto_has_json_requirement_rate": fmt_float(sum(row.get("auto_has_json_requirement") == "yes" for row in items) / n),
                "auto_possible_format_violation_rate": fmt_float(sum(row.get("auto_possible_format_violation") == "yes" for row in items) / n),
                "train_low_count_same_metric": items[0].get("auto_train_low_count_same_metric", ""),
                "train_high_count_same_metric": items[0].get("auto_train_high_count_same_metric", ""),
                "train_low_count_same_rubric_hash": items[0].get("auto_train_low_count_same_rubric_hash", ""),
                "train_high_count_same_rubric_hash": items[0].get("auto_train_high_count_same_rubric_hash", ""),
                "dominant_failure_mode_manual": "",
                "rubric_link_rate_manual": "",
                "possible_label_conflict_rate_manual": "",
                "trainability_summary_manual": "",
            }
        )
    return out


def case_control_review(controls: list[dict[str, str]], cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    case_by_id = {clean_text(row.get("sample_id")): row for row in cases}
    out: list[dict[str, Any]] = []
    for control in controls:
        case_id = clean_text(control.get("case_sample_id"))
        case = case_by_id.get(case_id, {})
        out.append(
            {
                "case_sample_id": case_id,
                "control_sample_id": clean_text(control.get("control_sample_id")),
                "match_rank": control.get("match_rank", ""),
                "match_score": control.get("match_score", ""),
                "same_metric": control.get("same_metric", ""),
                "same_language": control.get("same_language", ""),
                "same_subject": control.get("same_subject", ""),
                "same_boundary_key": control.get("same_boundary_key", ""),
                "same_rubric_hash": control.get("same_rubric_hash", ""),
                "case_gold_label": control.get("case_gold_label", case.get("gold_label", "")),
                "case_pred_label": control.get("case_pred_label", case.get("pred_label", "")),
                "control_gold_label": control.get("control_gold_label", ""),
                "control_pred_label": control.get("control_pred_label", ""),
                "case_s": control.get("case_s", case.get("quality_score_s", "")),
                "control_s": control.get("control_s", ""),
                "s_gap_control_minus_case": control.get("s_gap_control_minus_case", ""),
                "case_g_i3": control.get("case_g_i3", case.get("g_i3", "")),
                "control_g_i3": control.get("control_g_i3", ""),
                "g_i3_gap_control_minus_case": control.get("g_i3_gap_control_minus_case", ""),
                "question": control.get("question", "") or control.get("case_question", "") or case.get("question", ""),
                "rubric": control.get("rubric", "") or case.get("rubric", ""),
                "case_answer": control.get("case_answer", "") or case.get("answer", ""),
                "control_answer": control.get("control_answer", ""),
                "manual_difference_notes": "",
                "does_control_satisfy_missing_constraint_manual": "",
                "case_hidden_failure_relative_to_control_manual": "",
            }
        )
    return out


def write_report(
    path: Path,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    join_strategy: str,
    train_note: str,
) -> None:
    group_counts = Counter(clean_text(row.get("question_group_id")) for row in rows)
    max_group = group_counts.most_common(1)[0] if group_counts else ("", 0)
    human_patterns = Counter(clean_text(row.get("human_agreement_pattern")) or "missing" for row in rows)
    json_req = Counter(row.get("auto_has_json_requirement") for row in rows)
    format_flags = Counter(row.get("auto_possible_format_violation") for row in rows)
    lines = [
        "# Exp17-D1 Hidden Failure Audit Preparation Report",
        "",
        "This script prepares a manual audit template only. It does not train a model, load checkpoints, import transformers, read test data, or generate raw predictions.",
        "",
        "## Inputs",
        "",
        f"- D0 cases: `{args.d0_cases}`",
        f"- D0 controls: `{args.d0_controls}`",
        f"- dev jsonl: `{args.dev_jsonl or 'not provided'}`",
        f"- train jsonl: `{args.train_jsonl or 'not provided'}`",
        f"- split: `{args.split}`",
        f"- seed: `{args.seed}`",
        "",
        "## Join Strategy",
        "",
        f"- `{join_strategy}`",
        "",
        "## Case Counts",
        "",
        f"- Cases: `{len(rows)}`",
        f"- Question group rows: `{len(group_rows)}`",
        f"- Largest question group: `{max_group[0]}` with `{max_group[1]}` cases ({fmt_float(max_group[1] / len(rows) if rows else 0)})",
        "",
        "## Human Agreement Pattern Distribution",
        "",
    ]
    for key, value in human_patterns.most_common():
        lines.append(f"- `{key}`: `{value}`")
    lines += [
        "",
        "## Automatic Format Flags",
        "",
        f"- JSON requirement flags: `{dict(json_req)}`",
        f"- Possible format violation flags: `{dict(format_flags)}`",
        "",
        "## Train Support",
        "",
        f"- {train_note}",
        "",
        "## Outputs",
        "",
        f"- `{path.parent / 'd1_hidden_failure_annotation_template.csv'}`",
        f"- `{path.parent / 'd1_question_group_summary.csv'}`",
        f"- `{path.parent / 'd1_matched_case_control_review.csv'}`",
        f"- Annotation guide: `thesis_exp/exp17_low_score_evidence/docs/exp17_d1_annotation_guidelines.md`",
        "",
        "## Leakage Statement",
        "",
        "- Dev-only diagnostic preparation.",
        "- Test data is not read.",
        "- No checkpoint or raw prediction file is generated.",
        "- Dev annotations are for diagnosis and should not be used directly as train labels.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    cases = read_csv(args.d0_cases)
    controls = read_csv(args.d0_controls)
    if args.controls_per_case and args.controls_per_case > 0:
        controls = [
            row
            for row in controls
            if safe_int(row.get("match_rank")) <= args.controls_per_case
        ]
    if args.max_cases and args.max_cases > 0:
        cases = cases[: args.max_cases]
    cases, join_strategy = merge_dev_fields(cases, args.dev_jsonl)
    train_metric_counts, train_rubric_counts, train_note = count_train_support(args.train_jsonl)
    template_rows = add_auto_fields(cases, controls, train_metric_counts, train_rubric_counts, train_note)
    group_rows = group_summary(template_rows)
    control_rows = case_control_review(controls, template_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "d1_hidden_failure_annotation_template.csv", template_rows, TEMPLATE_FIELDS)
    write_csv(args.out_dir / "d1_question_group_summary.csv", group_rows, QUESTION_GROUP_FIELDS)
    write_csv(args.out_dir / "d1_matched_case_control_review.csv", control_rows, CASE_CONTROL_FIELDS)
    write_report(args.out_dir / "exp17_d1_prepare_report.md", args, template_rows, group_rows, join_strategy, train_note)
    return {
        "cases": len(template_rows),
        "groups": len(group_rows),
        "case_controls": len(control_rows),
        "out_dir": str(args.out_dir),
        "join_strategy": join_strategy,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Exp17-D1 hidden failure audit files.")
    parser.add_argument("--d0-cases", required=True, type=Path)
    parser.add_argument("--d0-controls", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--seed", default="42")
    parser.add_argument("--dev-jsonl", type=Path, default=None)
    parser.add_argument("--train-jsonl", type=Path, default=None)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--controls-per-case", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
