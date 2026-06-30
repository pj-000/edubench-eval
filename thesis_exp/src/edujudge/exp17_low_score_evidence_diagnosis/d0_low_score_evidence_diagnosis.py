"""Exp17-D0 diagnosis for low-score evidence failures.

This module does not train or load checkpoints. It analyzes Exp16A qmr
boundary-cache predictions and the original dev split to test whether label-2
high predictions come from quality-score separation failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from thesis_exp.src.edujudge.exp17_low_score_evidence_diagnosis import (
    EXP17_OUTPUT_DIR,
    EXP17_REPORTS_DIR,
    EXP17_TABLES_DIR,
    ensure_exp17_dirs,
)
from thesis_exp.src.edujudge.utils.io import read_jsonl, relpath, write_csv, write_text


DEFAULT_PREDICTIONS_PATH = (
    Path("thesis_exp")
    / "outputs"
    / "exp16_boundary_linking"
    / "rq1_cache_eval"
    / "scout_seed42"
    / "qmr"
    / "predictions_dev.jsonl"
)
DEFAULT_DEV_PATH = Path("thesis_exp") / "data" / "splits" / "question_seed42" / "dev.jsonl"

DEFECT_TYPES = (
    "missing_key_point",
    "rubric_violation",
    "off_task_or_irrelevant",
    "insufficient_evidence",
    "wrong_reasoning",
    "surface_fluent_but_wrong",
    "too_generic",
    "other",
    "unclear_or_ambiguous",
)


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def fmt(value: Any, digits: int = 4) -> str:
    try:
        if math.isnan(float(value)):
            return "nan"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def mean(values: list[float]) -> float:
    clean = [value for value in values if not math.isnan(value)]
    return float(np.mean(clean)) if clean else float("nan")


def med(values: list[float]) -> float:
    clean = [value for value in values if not math.isnan(value)]
    return float(median(clean)) if clean else float("nan")


def rate(num: int, den: int) -> float:
    return float(num / den) if den else float("nan")


def stable_hash(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (list, dict)) else str(value or "")
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("record_id") or row.get("id") or row.get("sample_id") or "").strip()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "\n".join(line.rstrip() for line in str(value).strip().splitlines())


def answer_length(row: dict[str, Any]) -> int:
    return len(clean_text(row.get("answer")).split())


def ordinal_class_probs(probs: list[float]) -> list[float]:
    if len(probs) != 4:
        return [float("nan")] * 5
    p_gt = [float(value) for value in probs]
    return [
        1.0 - p_gt[0],
        p_gt[0] - p_gt[1],
        p_gt[1] - p_gt[2],
        p_gt[2] - p_gt[3],
        p_gt[3],
    ]


def roc_auc(scores: list[float], labels: list[int]) -> float:
    pairs = [(score, label) for score, label in zip(scores, labels) if not math.isnan(score)]
    positives = sum(1 for _, label in pairs if label == 1)
    negatives = sum(1 for _, label in pairs if label == 0)
    if positives == 0 or negatives == 0:
        return float("nan")
    pairs.sort(key=lambda item: item[0])
    rank_sum = 0.0
    idx = 0
    while idx < len(pairs):
        end = idx + 1
        while end < len(pairs) and pairs[end][0] == pairs[idx][0]:
            end += 1
        avg_rank = (idx + 1 + end) / 2.0
        rank_sum += avg_rank * sum(1 for _, label in pairs[idx:end] if label == 1)
        idx = end
    return float((rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def merge_rows(predictions: list[dict[str, Any]], raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_by_id = {row_id(row): row for row in raw_rows}
    merged: list[dict[str, Any]] = []
    missing: list[str] = []
    for pred in predictions:
        sample_id = str(pred.get("sample_id") or "")
        raw = raw_by_id.get(sample_id)
        if raw is None:
            missing.append(sample_id)
            raw = {}
        probs = pred.get("probs") or []
        class_probs = ordinal_class_probs(probs if isinstance(probs, list) else [])
        scale_alpha = safe_float(pred.get("scale_alpha"))
        margin_tau2 = safe_float(pred.get("margin_tau2"))
        margin_tau3 = safe_float(pred.get("margin_tau3"))
        row = {
            **pred,
            "p_y1": class_probs[0],
            "p_y2": class_probs[1],
            "p_y3": class_probs[2],
            "p_y4": class_probs[3],
            "p_y5": class_probs[4],
            "g_i2": scale_alpha * margin_tau2,
            "g_i3": scale_alpha * margin_tau3,
            "question": clean_text(raw.get("question")),
            "answer": clean_text(raw.get("answer")),
            "rubric": clean_text(raw.get("rubric") or raw.get("rubric_text") or raw.get("rubric_canonical")),
            "metadata": clean_text(raw.get("metadata_raw") or raw.get("metadata")),
            "language": clean_text(raw.get("language")),
            "subject": clean_text(raw.get("subject_canonical") or raw.get("subject")),
            "scenario": clean_text(raw.get("scenario_canonical") or raw.get("scenario")),
            "education_level": clean_text(raw.get("education_level_canonical") or raw.get("education_level")),
            "metric_group": clean_text(raw.get("metric_group")),
            "answer_length": answer_length(raw),
            "rubric_hash": stable_hash(raw.get("rubric") or raw.get("rubric_text") or raw.get("rubric_canonical")),
        }
        merged.append(row)
    if missing:
        print(f"WARNING: {len(missing)} predictions had no raw dev row match; first missing={missing[:3]}")
    return merged


def score_distribution(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for label in [1, 2, 3, 4, 5]:
        items = [row for row in rows if safe_int(row.get("gold_label")) == label]
        pred_ge4 = sum(1 for row in items if safe_int(row.get("pred_label")) >= 4)
        recall = sum(1 for row in items if safe_int(row.get("pred_label")) == label)
        out.append(
            {
                "gold_label": label,
                "n": len(items),
                "mean_s": mean([safe_float(row.get("quality_score_s")) for row in items]),
                "median_s": med([safe_float(row.get("quality_score_s")) for row in items]),
                "mean_tau2": mean([safe_float(row.get("tau2")) for row in items]),
                "mean_tau3": mean([safe_float(row.get("tau3")) for row in items]),
                "mean_tau4": mean([safe_float(row.get("tau4")) for row in items]),
                "mean_margin_tau2": mean([safe_float(row.get("margin_tau2")) for row in items]),
                "mean_margin_tau3": mean([safe_float(row.get("margin_tau3")) for row in items]),
                "mean_g_i2": mean([safe_float(row.get("g_i2")) for row in items]),
                "mean_g_i3": mean([safe_float(row.get("g_i3")) for row in items]),
                "pred_ge4_count": pred_ge4,
                "pred_ge4_rate": rate(pred_ge4, len(items)),
                "recall": rate(recall, len(items)),
            }
        )
    return out


def label2_l2h_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "sample_id",
        "question_key",
        "metric",
        "gold_label",
        "pred_label",
        "quality_score_s",
        "tau2",
        "tau3",
        "tau4",
        "scale_alpha",
        "g_i2",
        "g_i3",
        "margin_tau2",
        "margin_tau3",
        "p_y1",
        "p_y2",
        "p_y3",
        "p_y4",
        "p_y5",
        "question",
        "answer",
        "rubric",
        "metadata",
        "boundary_key",
        "language",
        "subject",
        "scenario",
        "education_level",
        "answer_length",
        "rubric_hash",
    ]
    cases = [
        {field: row.get(field, "") for field in fields}
        for row in rows
        if safe_int(row.get("gold_label")) == 2 and safe_int(row.get("pred_label")) >= 4
    ]
    return sorted(cases, key=lambda row: (-safe_float(row.get("g_i3")), -safe_int(row.get("pred_label")), -safe_float(row.get("quality_score_s"))))


def match_score(case: dict[str, Any], control: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    same_metric = str(case.get("metric") or "") == str(control.get("metric") or "")
    same_language = str(case.get("language") or "") == str(control.get("language") or "")
    same_subject = str(case.get("subject") or "") == str(control.get("subject") or "")
    same_boundary_key = str(case.get("boundary_key") or "") == str(control.get("boundary_key") or "")
    same_rubric_hash = str(case.get("rubric_hash") or "") == str(control.get("rubric_hash") or "")
    len_gap = abs(safe_int(case.get("answer_length")) - safe_int(control.get("answer_length")))
    score = (
        100.0 * float(same_metric)
        + 30.0 * float(same_language)
        + 20.0 * float(same_subject)
        + 50.0 * float(same_boundary_key)
        + 40.0 * float(same_rubric_hash)
        - min(len_gap, 200) / 20.0
        + safe_float(control.get("quality_score_s")) / 1000.0
    )
    flags = {
        "same_metric": same_metric,
        "same_language": same_language,
        "same_subject": same_subject,
        "same_boundary_key": same_boundary_key,
        "same_rubric_hash": same_rubric_hash,
        "answer_length_gap": len_gap,
    }
    return score, flags


def matched_controls(rows: list[dict[str, Any]], cases: list[dict[str, Any]], controls_per_case: int) -> list[dict[str, Any]]:
    controls = [
        row
        for row in rows
        if safe_int(row.get("gold_label")) in {4, 5} and safe_int(row.get("pred_label")) in {4, 5}
    ]
    out: list[dict[str, Any]] = []
    for case in cases:
        scored: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        for control in controls:
            score, flags = match_score(case, control)
            scored.append((score, flags, control))
        scored.sort(key=lambda item: (-item[0], str(item[2].get("sample_id") or "")))
        for rank, (score, flags, control) in enumerate(scored[:controls_per_case], start=1):
            out.append(
                {
                    "case_sample_id": case.get("sample_id", ""),
                    "control_sample_id": control.get("sample_id", ""),
                    "match_rank": rank,
                    "match_score": score,
                    **flags,
                    "case_gold_label": case.get("gold_label", ""),
                    "case_pred_label": case.get("pred_label", ""),
                    "control_gold_label": control.get("gold_label", ""),
                    "control_pred_label": control.get("pred_label", ""),
                    "case_s": case.get("quality_score_s", ""),
                    "control_s": control.get("quality_score_s", ""),
                    "s_gap_control_minus_case": safe_float(control.get("quality_score_s")) - safe_float(case.get("quality_score_s")),
                    "case_g_i3": case.get("g_i3", ""),
                    "control_g_i3": control.get("g_i3", ""),
                    "g_i3_gap_control_minus_case": safe_float(control.get("g_i3")) - safe_float(case.get("g_i3")),
                    "case_question": case.get("question", ""),
                    "case_answer": case.get("answer", ""),
                    "control_answer": control.get("answer", ""),
                    "rubric": case.get("rubric", ""),
                }
            )
    return out


def metric_failure_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("metric") or "")].append(row)
    out: list[dict[str, Any]] = []
    for metric, items in sorted(groups.items()):
        label2 = [row for row in items if safe_int(row.get("gold_label")) == 2]
        label2_l2h = [row for row in label2 if safe_int(row.get("pred_label")) >= 4]
        out.append(
            {
                "metric": metric,
                "n": len(items),
                "label2_n": len(label2),
                "label2_l2h_count": len(label2_l2h),
                "r_2_to_h": rate(len(label2_l2h), len(label2)),
                "label2_mean_s": mean([safe_float(row.get("quality_score_s")) for row in label2]),
                "label2_mean_g_i3": mean([safe_float(row.get("g_i3")) for row in label2]),
            }
        )
    return sorted(out, key=lambda row: (-safe_int(row.get("label2_l2h_count")), -safe_float(row.get("r_2_to_h")), str(row.get("metric"))))


def manual_audit_template(cases: list[dict[str, Any]], audit_size: int) -> list[dict[str, Any]]:
    if audit_size <= 0:
        return []
    by_metric: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_metric[str(case.get("metric") or "")].append(case)
    selected: list[dict[str, Any]] = []
    metric_names = sorted(by_metric, key=lambda metric: (-len(by_metric[metric]), metric))
    cursor = 0
    while len(selected) < min(audit_size, len(cases)) and metric_names:
        metric = metric_names[cursor % len(metric_names)]
        bucket = by_metric[metric]
        if bucket:
            selected.append(bucket.pop(0))
        metric_names = [name for name in metric_names if by_metric[name]]
        cursor += 1
    out: list[dict[str, Any]] = []
    for case in selected:
        out.append(
            {
                **case,
                "defect_type_manual": "",
                "defect_type_options": "|".join(DEFECT_TYPES),
                "defect_notes_manual": "",
                "rubric_clause_manual": "",
                "evidence_span_manual": "",
                "confidence_manual": "",
                "is_rubric_linked_manual": "",
            }
        )
    return out


def support_level(rows: list[dict[str, Any]], cases: list[dict[str, Any]], matched: list[dict[str, Any]]) -> tuple[str, list[str]]:
    label2 = [row for row in rows if safe_int(row.get("gold_label")) == 2]
    high = [row for row in rows if safe_int(row.get("gold_label")) >= 4]
    label2_l2h_rate = rate(len(cases), len(label2))
    label2_mean_g = mean([safe_float(row.get("g_i3")) for row in label2])
    auc_high_vs_label2 = roc_auc(
        [safe_float(row.get("quality_score_s")) for row in label2 + high],
        [0] * len(label2) + [1] * len(high),
    )
    matched_gap = mean([safe_float(row.get("s_gap_control_minus_case")) for row in matched])
    criteria = [
        ("label2_l2h_rate >= 0.5", label2_l2h_rate >= 0.5),
        ("label2 mean g_i3 > 0", label2_mean_g > 0.0),
        ("AUC(s, high vs label2) <= 0.65", auc_high_vs_label2 <= 0.65),
        ("matched control-case mean s gap <= 0.25", matched_gap <= 0.25),
        ("label2 L2H case count >= 20", len(cases) >= 20),
    ]
    hits = [name for name, ok in criteria if ok]
    if len(hits) >= 4:
        return "Strong support", hits
    if len(hits) >= 2:
        return "Partial support", hits
    return "Weak support", hits


def write_report(
    report_path: Path,
    rows: list[dict[str, Any]],
    dist_rows: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    matched: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
) -> None:
    label2 = [row for row in rows if safe_int(row.get("gold_label")) == 2]
    high = [row for row in rows if safe_int(row.get("gold_label")) >= 4]
    auc_high_vs_label2 = roc_auc(
        [safe_float(row.get("quality_score_s")) for row in label2 + high],
        [0] * len(label2) + [1] * len(high),
    )
    auc_low_vs_rest = roc_auc(
        [-safe_float(row.get("quality_score_s")) for row in rows],
        [1 if safe_int(row.get("gold_label")) <= 2 else 0 for row in rows],
    )
    support, support_hits = support_level(rows, cases, matched)
    label2_mean_s = mean([safe_float(row.get("quality_score_s")) for row in label2])
    high_mean_s = mean([safe_float(row.get("quality_score_s")) for row in high])
    matched_gap = mean([safe_float(row.get("s_gap_control_minus_case")) for row in matched])
    lines = [
        "# Exp17-D0 Low-Score Evidence Diagnosis",
        "",
        "This diagnostic does not train a model. It analyzes Exp16A qmr boundary-cache dev predictions to test whether label-2 high predictions are caused by quality-score separation failure.",
        "",
        "## Key Findings",
        "",
        f"- Rows analyzed: `{len(rows)}`.",
        f"- Label-2 samples: `{len(label2)}`; label-2 predicted as 4/5: `{len(cases)}` ({fmt(rate(len(cases), len(label2)))})；label-2 recall remains `{fmt(next((row['recall'] for row in dist_rows if row['gold_label'] == 2), float('nan')))}`.",
        f"- Mean `s` for label2: `{fmt(label2_mean_s)}`; mean `s` for labels 4/5: `{fmt(high_mean_s)}`; delta high-minus-label2: `{fmt(high_mean_s - label2_mean_s)}`.",
        f"- AUC using `s` to separate labels 4/5 from label2: `{fmt(auc_high_vs_label2)}`.",
        f"- AUC using negative `s` to separate low labels <=2 from the rest: `{fmt(auc_low_vs_rest)}`.",
        f"- Matched control-case mean `s` gap: `{fmt(matched_gap)}`.",
        f"- RQ2 support level: **{support}** ({', '.join(support_hits) if support_hits else 'no criteria hit'}).",
        "",
        "## Score Distribution by Gold Label",
        "",
        "| gold | n | mean s | median s | mean tau3 | mean margin tau3 | mean g_i3 | pred>=4 | recall |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in dist_rows:
        lines.append(
            f"| {row['gold_label']} | {row['n']} | {fmt(row['mean_s'])} | {fmt(row['median_s'])} | "
            f"{fmt(row['mean_tau3'])} | {fmt(row['mean_margin_tau3'])} | {fmt(row['mean_g_i3'])} | "
            f"{row['pred_ge4_count']}/{row['n']} ({fmt(row['pred_ge4_rate'])}) | {fmt(row['recall'])} |"
        )
    lines += [
        "",
        "## Label-2 Error Concentration by Metric",
        "",
        "| metric | label2 n | label2 -> 4/5 | R2->H | label2 mean s | label2 mean g_i3 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows[:20]:
        if safe_int(row.get("label2_n")) == 0:
            continue
        lines.append(
            f"| {row['metric']} | {row['label2_n']} | {row['label2_l2h_count']} | {fmt(row['r_2_to_h'])} | "
            f"{fmt(row['label2_mean_s'])} | {fmt(row['label2_mean_g_i3'])} |"
        )
    lines += [
        "",
        "## Manual Audit Template",
        "",
        f"`manual_audit_template.csv` contains `{len(audit_rows)}` label2 high-prediction cases for human defect annotation.",
        "Use the provided defect type options to mark whether each error is linked to missing key points, rubric violation, insufficient evidence, surface fluency, or ambiguity.",
        "",
        "## Interpretation",
        "",
    ]
    if support == "Strong support":
        lines.append(
            "The diagnosis strongly supports the RQ2 hypothesis: Exp16A's label-2 failure is primarily a quality-score evidence problem. The next step should be Exp17-A/B/C rather than another boundary or threshold experiment."
        )
    elif support == "Partial support":
        lines.append(
            "The diagnosis partially supports the RQ2 hypothesis. Exp17-A/B/C is still reasonable, but the manual audit should be completed before treating low-score defect evidence as the main explanation."
        )
    else:
        lines.append(
            "The diagnosis weakly supports the RQ2 hypothesis. Before training Exp17, inspect the manual audit for label noise, ambiguous 2/3 boundaries, or rubric coverage problems."
        )
    lines.append(
        "If the manual audit finds clear rubric-linked defects in most sampled cases, Exp17 can define `D` as weakly supervised low-score defect evidence and inject it into `s = s_base - lambda * D`."
    )
    write_text(report_path, "\n".join(lines))


def run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_exp17_dirs()
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    predictions = read_jsonl(args.predictions_path)
    raw_rows = read_jsonl(args.dev_path)
    rows = merge_rows(predictions, raw_rows)
    dist_rows = score_distribution(rows)
    cases = label2_l2h_cases(rows)
    matched = matched_controls(rows, cases, controls_per_case=int(args.controls_per_case))
    metric_rows = metric_failure_summary(rows)
    audit_rows = manual_audit_template(cases, audit_size=int(args.audit_size))
    write_csv(args.tables_dir / "label2_score_distribution.csv", dist_rows)
    write_csv(args.tables_dir / "label2_l2h_cases.csv", cases)
    write_csv(args.tables_dir / "matched_high_score_controls.csv", matched)
    write_csv(args.tables_dir / "metric_label2_failure_summary.csv", metric_rows)
    write_csv(args.tables_dir / "manual_audit_template.csv", audit_rows)
    write_report(args.reports_dir / "exp17_d0_diagnosis_report.md", rows, dist_rows, cases, matched, metric_rows, audit_rows)
    support, hits = support_level(rows, cases, matched)
    return {
        "rows": len(rows),
        "label2_l2h_cases": len(cases),
        "manual_audit_rows": len(audit_rows),
        "support": support,
        "support_criteria": hits,
        "output_dir": relpath(EXP17_OUTPUT_DIR),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Exp17-D0 low-score evidence diagnosis from Exp16A predictions.")
    parser.add_argument("--predictions_path", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--dev_path", type=Path, default=DEFAULT_DEV_PATH)
    parser.add_argument("--tables_dir", type=Path, default=EXP17_TABLES_DIR)
    parser.add_argument("--reports_dir", type=Path, default=EXP17_REPORTS_DIR)
    parser.add_argument("--audit_size", type=int, default=50)
    parser.add_argument("--controls_per_case", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
