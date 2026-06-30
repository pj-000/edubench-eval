"""Re-annotate Exp17-D1 cases using recovered human rationale evidence.

The output is still a dev-only diagnostic annotation. It must not be used as
train labels directly. The purpose is to redo D1 after recovering the original
5-grade human rating reasons from the current fork's `5-grades/` files.
"""

from __future__ import annotations

import argparse
import csv
import textwrap
from pathlib import Path
from typing import Any


DEFAULT_TEMPLATE_CSV = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/"
    "d1_hidden_failure_audit_seed42_dev/d1_hidden_failure_annotation_template.csv"
)
DEFAULT_RECOVERED_CSV = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/"
    "d1_hidden_failure_audit_seed42_dev/human_rationale_recovery/"
    "d1_human_rationale_recovered.csv"
)
DEFAULT_OUT_CSV = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/"
    "d1_hidden_failure_audit_seed42_dev/"
    "d1_hidden_failure_annotation_template_filled_human_rationale_recovered.csv"
)
DEFAULT_REPORT = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/"
    "d1_hidden_failure_audit_seed42_dev/"
    "d1_human_rationale_reannotation_report.md"
)


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


BASE_ANNOTATIONS: dict[str, dict[str, str]] = {
    # Annales school.
    "1": {
        "primary_failure_mode_manual": "task_constraint_violation",
        "secondary_failure_mode_manual": "possible_label_conflict",
        "rubric_link_level_manual": "explicit_rubric_clause",
        "confidence_manual": "3",
        "trainability_manual": "pairwise_only",
        "recommended_training_use_manual": "pairwise_low",
    },
    "2": {
        "primary_failure_mode_manual": "factual_or_rubric_mismatch",
        "secondary_failure_mode_manual": "surface_fluent_but_hidden_defect",
        "rubric_link_level_manual": "explicit_rubric_clause",
        "confidence_manual": "5",
        "trainability_manual": "strong_train_signal",
        "recommended_training_use_manual": "evidence_positive",
    },
    "10": {
        "primary_failure_mode_manual": "factual_or_rubric_mismatch",
        "secondary_failure_mode_manual": "surface_fluent_but_hidden_defect",
        "rubric_link_level_manual": "explicit_rubric_clause",
        "confidence_manual": "5",
        "trainability_manual": "strong_train_signal",
        "recommended_training_use_manual": "evidence_positive",
    },
    "12": {
        "primary_failure_mode_manual": "factual_or_rubric_mismatch",
        "secondary_failure_mode_manual": "task_constraint_violation",
        "rubric_link_level_manual": "explicit_rubric_clause",
        "confidence_manual": "5",
        "trainability_manual": "strong_train_signal",
        "recommended_training_use_manual": "evidence_positive",
    },
    "15": {
        "primary_failure_mode_manual": "factual_or_rubric_mismatch",
        "secondary_failure_mode_manual": "surface_fluent_but_hidden_defect",
        "rubric_link_level_manual": "explicit_rubric_clause",
        "confidence_manual": "5",
        "trainability_manual": "strong_train_signal",
        "recommended_training_use_manual": "evidence_positive",
    },
    # Marketing manager instruction following.
    "3": {
        "primary_failure_mode_manual": "missing_key_point",
        "secondary_failure_mode_manual": "task_constraint_violation",
        "rubric_link_level_manual": "explicit_rubric_clause",
        "confidence_manual": "4",
        "trainability_manual": "weak_train_signal",
        "recommended_training_use_manual": "pairwise_low",
    },
    "4": {
        "primary_failure_mode_manual": "possible_label_conflict",
        "secondary_failure_mode_manual": "unclear",
        "rubric_link_level_manual": "not_rubric_linked",
        "confidence_manual": "3",
        "trainability_manual": "downweight_or_exclude",
        "recommended_training_use_manual": "exclude",
    },
    "6": {
        "primary_failure_mode_manual": "missing_key_point",
        "secondary_failure_mode_manual": "task_constraint_violation",
        "rubric_link_level_manual": "inferred_from_context",
        "confidence_manual": "3",
        "trainability_manual": "weak_train_signal",
        "recommended_training_use_manual": "pairwise_low",
    },
    "7": {
        "primary_failure_mode_manual": "missing_key_point",
        "secondary_failure_mode_manual": "task_constraint_violation",
        "rubric_link_level_manual": "explicit_rubric_clause",
        "confidence_manual": "4",
        "trainability_manual": "weak_train_signal",
        "recommended_training_use_manual": "pairwise_low",
    },
    "8": {
        "primary_failure_mode_manual": "missing_key_point",
        "secondary_failure_mode_manual": "task_constraint_violation",
        "rubric_link_level_manual": "explicit_rubric_clause",
        "confidence_manual": "4",
        "trainability_manual": "weak_train_signal",
        "recommended_training_use_manual": "pairwise_low",
    },
    # Marketing manager correction precision.
    "5": {
        "primary_failure_mode_manual": "missing_key_point",
        "secondary_failure_mode_manual": "insufficient_evidence",
        "rubric_link_level_manual": "explicit_rubric_clause",
        "confidence_manual": "4",
        "trainability_manual": "weak_train_signal",
        "recommended_training_use_manual": "pairwise_low",
    },
    "9": {
        "primary_failure_mode_manual": "missing_key_point",
        "secondary_failure_mode_manual": "insufficient_evidence",
        "rubric_link_level_manual": "inferred_from_context",
        "confidence_manual": "3",
        "trainability_manual": "weak_train_signal",
        "recommended_training_use_manual": "pairwise_low",
    },
    "11": {
        "primary_failure_mode_manual": "missing_key_point",
        "secondary_failure_mode_manual": "insufficient_evidence",
        "rubric_link_level_manual": "explicit_rubric_clause",
        "confidence_manual": "4",
        "trainability_manual": "weak_train_signal",
        "recommended_training_use_manual": "pairwise_low",
    },
    "13": {
        "primary_failure_mode_manual": "missing_key_point",
        "secondary_failure_mode_manual": "insufficient_evidence",
        "rubric_link_level_manual": "explicit_rubric_clause",
        "confidence_manual": "4",
        "trainability_manual": "weak_train_signal",
        "recommended_training_use_manual": "pairwise_low",
    },
    # Clarity and inspiration.
    "14": {
        "primary_failure_mode_manual": "surface_fluent_but_hidden_defect",
        "secondary_failure_mode_manual": "insufficient_evidence",
        "rubric_link_level_manual": "explicit_rubric_clause",
        "confidence_manual": "4",
        "trainability_manual": "weak_train_signal",
        "recommended_training_use_manual": "pairwise_low",
    },
    "16": {
        "primary_failure_mode_manual": "surface_fluent_but_hidden_defect",
        "secondary_failure_mode_manual": "insufficient_evidence",
        "rubric_link_level_manual": "inferred_from_context",
        "confidence_manual": "3",
        "trainability_manual": "weak_train_signal",
        "recommended_training_use_manual": "pairwise_low",
    },
    "17": {
        "primary_failure_mode_manual": "surface_fluent_but_hidden_defect",
        "secondary_failure_mode_manual": "insufficient_evidence",
        "rubric_link_level_manual": "explicit_rubric_clause",
        "confidence_manual": "4",
        "trainability_manual": "weak_train_signal",
        "recommended_training_use_manual": "pairwise_low",
    },
    "18": {
        "primary_failure_mode_manual": "surface_fluent_but_hidden_defect",
        "secondary_failure_mode_manual": "insufficient_evidence",
        "rubric_link_level_manual": "explicit_rubric_clause",
        "confidence_manual": "4",
        "trainability_manual": "weak_train_signal",
        "recommended_training_use_manual": "pairwise_low",
    },
    # Reasoning rigor.
    "19": {
        "primary_failure_mode_manual": "insufficient_evidence",
        "secondary_failure_mode_manual": "missing_key_point",
        "rubric_link_level_manual": "explicit_rubric_clause",
        "confidence_manual": "4",
        "trainability_manual": "weak_train_signal",
        "recommended_training_use_manual": "pairwise_low",
    },
    "20": {
        "primary_failure_mode_manual": "insufficient_evidence",
        "secondary_failure_mode_manual": "missing_key_point",
        "rubric_link_level_manual": "inferred_from_context",
        "confidence_manual": "3",
        "trainability_manual": "weak_train_signal",
        "recommended_training_use_manual": "pairwise_low",
    },
    "22": {
        "primary_failure_mode_manual": "insufficient_evidence",
        "secondary_failure_mode_manual": "missing_key_point",
        "rubric_link_level_manual": "explicit_rubric_clause",
        "confidence_manual": "4",
        "trainability_manual": "weak_train_signal",
        "recommended_training_use_manual": "pairwise_low",
    },
    # Scenario element integration.
    "21": {
        "primary_failure_mode_manual": "missing_key_point",
        "secondary_failure_mode_manual": "surface_fluent_but_hidden_defect",
        "rubric_link_level_manual": "explicit_rubric_clause",
        "confidence_manual": "4",
        "trainability_manual": "weak_train_signal",
        "recommended_training_use_manual": "pairwise_low",
    },
    "24": {
        "primary_failure_mode_manual": "missing_key_point",
        "secondary_failure_mode_manual": "surface_fluent_but_hidden_defect",
        "rubric_link_level_manual": "inferred_from_context",
        "confidence_manual": "3",
        "trainability_manual": "weak_train_signal",
        "recommended_training_use_manual": "pairwise_low",
    },
    "25": {
        "primary_failure_mode_manual": "missing_key_point",
        "secondary_failure_mode_manual": "surface_fluent_but_hidden_defect",
        "rubric_link_level_manual": "explicit_rubric_clause",
        "confidence_manual": "4",
        "trainability_manual": "weak_train_signal",
        "recommended_training_use_manual": "pairwise_low",
    },
    "26": {
        "primary_failure_mode_manual": "missing_key_point",
        "secondary_failure_mode_manual": "surface_fluent_but_hidden_defect",
        "rubric_link_level_manual": "explicit_rubric_clause",
        "confidence_manual": "4",
        "trainability_manual": "weak_train_signal",
        "recommended_training_use_manual": "pairwise_low",
    },
    # Chinese grading case.
    "23": {
        "primary_failure_mode_manual": "task_constraint_violation",
        "secondary_failure_mode_manual": "format_violation",
        "rubric_link_level_manual": "implicit_task_constraint",
        "confidence_manual": "5",
        "trainability_manual": "format_auxiliary_signal",
        "recommended_training_use_manual": "format_auxiliary",
    },
    # SWOT feedback case.
    "27": {
        "primary_failure_mode_manual": "missing_key_point",
        "secondary_failure_mode_manual": "insufficient_evidence",
        "rubric_link_level_manual": "explicit_rubric_clause",
        "confidence_manual": "4",
        "trainability_manual": "weak_train_signal",
        "recommended_training_use_manual": "pairwise_low",
    },
}


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


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def shorten(value: str, width: int = 260) -> str:
    return textwrap.shorten(" ".join(clean(value).split()), width=width, placeholder="...")


def evidence_span(row: dict[str, str]) -> str:
    answer = row.get("answer", "")
    metric = row.get("metric", "")
    if "Annales" in row.get("question", ""):
        if "Marc Bloch" in answer:
            return "correct_answer/Answer = Marc Bloch"
    if "marketing manager" in row.get("question", "").lower():
        if "Scenario Element" in metric:
            return "Error Explanation compares marketing vs finance duties but does not adapt to the student's misconception/context."
        if "Reasoning Process" in metric:
            return "Error Explanation states role mismatch but gives little causal/organizational reasoning."
        if "Clarity" in metric:
            return "Error Explanation is fluent but generic; no example, analogy, or guiding question."
        if "Error Identification" in metric:
            return "Corrected Answer / Error Explanation identify the role error but omit key duties and deeper correction."
        return "Corrected Answer is narrower than the expected marketing-manager duties."
    if "评分" in answer and "评分细节" in answer:
        return "评分=5 conflicts with low scoring details and uses non-task dimensions such as 解释部分/案例部分."
    if "SWOT" in row.get("question", ""):
        return "Personalized feedback is clear but does not actively encourage or guide learner thinking."
    return shorten(answer, 180)


def default_booleans(case_no: str, annotation: dict[str, str]) -> dict[str, str]:
    primary = annotation["primary_failure_mode_manual"]
    is_conflict = primary == "possible_label_conflict"
    is_format_or_task = primary in {"format_violation", "task_constraint_violation"} or annotation[
        "secondary_failure_mode_manual"
    ] in {"format_violation", "task_constraint_violation"}
    return {
        "is_surface_fluent_manual": "1",
        "is_hidden_failure_manual": "0" if is_conflict else "1",
        "is_format_or_task_constraint_manual": "1" if is_format_or_task else "0",
        "possible_label_conflict_manual": "1" if is_conflict else "0",
        "llm_or_model_over_scoring_pattern_manual": "yes",
    }


def build_rows(template_rows: list[dict[str, str]], recovered_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    recovered_by_sample = {row["sample_id"]: row for row in recovered_rows}
    out: list[dict[str, str]] = []

    for idx, row in enumerate(template_rows, start=1):
        sample_id = row["sample_id"]
        recovered = recovered_by_sample.get(sample_id)
        if not recovered:
            raise KeyError(f"Missing recovered rationale row for sample_id={sample_id}")
        case_no = recovered["case_no"]
        annotation = dict(BASE_ANNOTATIONS[case_no])
        annotation.update(default_booleans(case_no, annotation))
        reason = clean(recovered.get("human_reason_summary"))
        match_status = clean(recovered.get("match_status"))
        source_note = (
            "Recovered original human rationale."
            if match_status == "metric_rationale_recovered"
            else "No exact recovered rationale; inferred from same question group and answer pattern."
        )
        if annotation["primary_failure_mode_manual"] == "possible_label_conflict":
            source_note = "No exact recovered rationale and the answer appears to satisfy the task; keep as possible label conflict."
        annotation.update(
            {
                "rubric_clause_manual": f"{row.get('metric', '')}: {shorten(reason, 180) if reason else source_note}",
                "evidence_span_manual": evidence_span(row),
                "defect_notes_manual": (
                    f"case_no={case_no}; match_status={match_status}; {source_note} "
                    f"human_reason_summary={shorten(reason, 360) if reason else 'N/A'}"
                ),
            }
        )
        updated = dict(row)
        for field in ANNOTATION_FIELDS:
            updated[field] = annotation.get(field, "")
        out.append(updated)
    return out


def write_report(path: Path, rows: list[dict[str, str]]) -> None:
    total = len(rows)
    recovered = sum("Recovered original human rationale" in row["defect_notes_manual"] for row in rows)
    inferred = sum("inferred from same question group" in row["defect_notes_manual"] for row in rows)
    conflicts = sum(row["possible_label_conflict_manual"] == "1" for row in rows)
    trainable = sum(row["trainability_manual"] in {"strong_train_signal", "weak_train_signal"} for row in rows)
    lines = [
        "# Exp17-D1 Reannotation with Human Rationales",
        "",
        "This is a dev-only diagnostic reannotation. It uses recovered original human rating rationales where available and does not train or read test data.",
        "",
        "## Summary",
        "",
        f"- Total D1 cases: `{total}`",
        f"- Cases with exact recovered metric-level rationale: `{recovered}/{total}`",
        f"- Cases inferred from same question group/answer pattern: `{inferred}/{total}`",
        f"- Possible label conflict cases after rationale recovery: `{conflicts}/{total}`",
        f"- Strong/weak train-signal cases: `{trainable}/{total}`",
        "",
        "## Interpretation",
        "",
        "- Recovering the original human reasons changes D1 materially: most label-2 high-prediction cases are no longer unexplained label conflicts.",
        "- The dominant evidence types are missing key duties, shallow correction reasoning, weak scenario adaptation, weak clarity/inspiration, and factual mismatch in the Annales item.",
        "- The evidence is still concentrated in one marketing-manager question group, so dev annotations should not be used directly as training labels.",
        "- Exp17-A, if run, should construct train-side weak labels or matched hard-negative pairs from the same evidence pattern rather than memorizing dev question keys.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-csv", type=Path, default=DEFAULT_TEMPLATE_CSV)
    parser.add_argument("--recovered-csv", type=Path, default=DEFAULT_RECOVERED_CSV)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    template_rows = read_csv(args.template_csv)
    recovered_rows = read_csv(args.recovered_csv)
    rows = build_rows(template_rows, recovered_rows)
    write_csv(args.out_csv, rows, list(template_rows[0].keys()))
    write_report(args.report, rows)
    print(f"Wrote reannotated D1 CSV: {args.out_csv}")
    print(f"Wrote report: {args.report}")


if __name__ == "__main__":
    main()
