"""Confirm label provenance for processed Excel candidate sources."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from thesis_exp.src.edujudge.exp06.processed_excel_common import all_candidate_rows, load_source_records, source_paths, write_processed_csv
from thesis_exp.src.edujudge.utils.io import relpath
from thesis_exp.src.edujudge.utils.text_norm import normalize_text, stringify


PROVENANCE_FIELDS = [
    "source_file",
    "label_fields",
    "score_fields",
    "evidence_for_human_label",
    "evidence_for_model_label",
    "evidence_for_pseudo_label",
    "has_generation_source",
    "has_reviewer_or_annotator_field",
    "has_criteria_or_rubric",
    "has_reason_or_rationale",
    "has_excel_original_column_name",
    "provenance_status",
    "confidence",
    "notes",
]


HUMAN_TOKENS = ["human", "annotator", "reviewer", "人工", "标注员", "审核", "复核"]
MODEL_TOKENS = ["model", "deepseek", "gpt", "qwen", "judge", "llm", "模型"]


def field_contains(fields: list[str], tokens: list[str]) -> bool:
    joined = " ".join(fields).lower()
    return any(token.lower() in joined for token in tokens)


def question_has_criteria(records: list[dict[str, Any]]) -> bool:
    return any("gradingcriteria" in normalize_text(record.get("question")) or "评分" in stringify(record.get("question")) for record in records)


def build_provenance_rows() -> list[dict[str, Any]]:
    candidate_rows = all_candidate_rows()
    rows: list[dict[str, Any]] = []
    for path in source_paths():
        source = relpath(path)
        records = load_source_records(path)
        source_candidates = [row for row in candidate_rows if row["source_file"] == source]
        top_fields = sorted({key for record in records for key in record})
        score_fields = sorted(
            {
                f"scores[].{key}"
                for record in records
                for item in record.get("scores", [])
                if isinstance(item, dict)
                for key in item
            }
        )
        model_values = sorted({stringify(record.get("model")) for record in records if record.get("model")})
        human_marker = field_contains(top_fields + score_fields, HUMAN_TOKENS)
        model_marker = field_contains(top_fields + score_fields + model_values, MODEL_TOKENS)
        has_reason = "scores[].reason" in score_fields
        has_criteria = question_has_criteria(records)
        has_excel_original_col = any(re.search(r"unnamed|理由\\.|##|excel|原始列", field, flags=re.IGNORECASE) for field in top_fields + score_fields)
        if human_marker:
            status = "human_confirmed"
            confidence = "medium"
            notes = "human/reviewer marker found; still needs manual confirmation against source Excel"
        elif model_marker:
            status = "model_label_likely"
            confidence = "high"
            notes = "model field and score/reason structure indicate labels are model or pseudo labels, not human-confirmed"
        elif has_reason:
            status = "pseudo_label_likely"
            confidence = "medium"
            notes = "scores with reasons exist but no human provenance marker"
        else:
            status = "unknown"
            confidence = "low"
            notes = "no direct evidence for label source"
        rows.append(
            {
                "source_file": source,
                "label_fields": ["derived target_label_5 from scores[].score"],
                "score_fields": score_fields,
                "evidence_for_human_label": "human/reviewer/annotator field present" if human_marker else "",
                "evidence_for_model_label": f"model field values={model_values}" if model_marker else "",
                "evidence_for_pseudo_label": "scores[].score with scores[].reason; no human provenance marker" if has_reason and not human_marker else "",
                "has_generation_source": bool(model_values),
                "has_reviewer_or_annotator_field": human_marker,
                "has_criteria_or_rubric": has_criteria,
                "has_reason_or_rationale": has_reason,
                "has_excel_original_column_name": has_excel_original_col,
                "provenance_status": status,
                "confidence": confidence,
                "notes": notes,
            }
        )
    return rows


def main() -> None:
    rows = build_provenance_rows()
    write_processed_csv("processed_excel_label_provenance.csv", rows, PROVENANCE_FIELDS)
    print(f"Wrote {len(rows)} provenance rows")


if __name__ == "__main__":
    main()
