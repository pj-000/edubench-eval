"""Inventory existing judge-score sources for Exp1."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.audit import (
    EVALUATORS,
    EXP01_OUTPUT_DIR,
    EXP01_SAMPLES_DIR,
    EXP01_TABLES_DIR,
    canonical_evaluator,
    collect_candidate_paths,
    ensure_exp01_dirs,
    evaluator_from_path,
    flatten_keys,
    infer_field_candidates,
    is_synthetic_or_sampled_path,
    markdown_table,
    relpath,
    sample_records,
    truncate_for_json,
    write_csv,
    write_json,
    write_text,
)
from thesis_exp.src.edujudge.utils.text_norm import normalize_text, stringify


INVENTORY_COLUMNS = [
    "file_path",
    "exists",
    "file_type",
    "num_records",
    "top_level_keys",
    "score_like_fields",
    "judge_name_candidates",
    "question_like_fields",
    "answer_like_fields",
    "metric_like_fields",
    "record_id_like_fields",
    "triple_key_like_fields",
    "language_fields",
    "likely_evaluator",
    "likely_role",
    "usable_for_exp01",
    "notes",
]


def _discover_evaluators(path: Path, samples: list[dict[str, Any]], keys: set[str]) -> list[str]:
    evaluators: set[str] = set()
    from_path = evaluator_from_path(path)
    if from_path:
        evaluators.add(from_path)
    for key in keys:
        found = canonical_evaluator(key)
        if found:
            evaluators.add(found)
    for record in samples:
        for field in ["eval", "evaluator", "judge", "judge_name", "model_name"]:
            found = canonical_evaluator(record.get(field))
            if found:
                evaluators.add(found)
        for field in ["judge_scores", "evaluate", "scores", "score"]:
            value = record.get(field)
            if isinstance(value, dict):
                for subkey in value:
                    found = canonical_evaluator(subkey)
                    if found:
                        evaluators.add(found)
    return [spec.name for spec in EVALUATORS if spec.name in evaluators]


def _infer_role(path: Path, samples: list[dict[str, Any]], evaluators: list[str], score_fields: list[str]) -> str:
    path_norm = normalize_text(relpath(path))
    sample_text = normalize_text(" ".join(stringify(sample)[:1200] for sample in samples))
    if is_synthetic_or_sampled_path(path):
        return "synthetic_or_sampled"
    if "human_" in path_norm or "human annotation" in sample_text:
        return "human_annotation"
    if "judge_scores" in sample_text:
        return "processed_or_split_with_judge_scores"
    if "evaluate" in sample_text and any(spec.name in evaluators for spec in EVALUATORS):
        return "merged_human_and_judge_scores"
    if evaluators and score_fields:
        return "llm_judge_output"
    if score_fields:
        return "score_like_unknown_evaluator"
    return "unknown"


def _field_summary(samples: list[dict[str, Any]], csv_fields: list[str]) -> tuple[list[str], set[str]]:
    if csv_fields:
        return sorted(csv_fields), set(csv_fields)
    top_keys: set[str] = set()
    all_keys: set[str] = set()
    for record in samples:
        top_keys.update(str(k) for k in record.keys())
        all_keys.update(flatten_keys(record))
    return sorted(top_keys), all_keys


def _usable_flag(
    role: str,
    evaluators: list[str],
    score_fields: list[str],
    question_fields: list[str],
    answer_fields: list[str],
    metric_fields: list[str],
    record_id_fields: list[str],
    triple_key_fields: list[str],
) -> tuple[str, str]:
    if role == "synthetic_or_sampled":
        return "false", "synthetic/sample/augmentation source excluded"
    if role == "human_annotation":
        return "false", "human labels only; not an automatic judge source"
    has_alignment_key = bool(record_id_fields or triple_key_fields or (question_fields and answer_fields and metric_fields))
    if evaluators and score_fields and has_alignment_key:
        return "true", "known evaluator and usable alignment fields"
    if role == "processed_or_split_with_judge_scores" and has_alignment_key:
        return "true", "locked Exp0.1 row-level judge_scores available"
    if score_fields and has_alignment_key:
        return "maybe", "score-like fields found, but evaluator identity is unclear"
    if evaluators and score_fields:
        return "maybe", "known evaluator but alignment fields are incomplete"
    if score_fields:
        return "false", "score-like fields found but no usable alignment key"
    return "false", "no judge score fields detected"


def inventory_file(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    exists = path.exists()
    samples: list[dict[str, Any]] = []
    num_records = 0
    csv_fields: list[str] = []
    notes: list[str] = []
    if exists:
        try:
            samples, num_records, csv_fields = sample_records(path)
        except Exception as exc:  # noqa: BLE001 - inventory should keep going.
            notes.append(f"read_error: {type(exc).__name__}: {exc}")
    top_keys, all_keys = _field_summary(samples, csv_fields)
    score_fields = infer_field_candidates(all_keys, "score")
    question_fields = infer_field_candidates(all_keys, "question")
    answer_fields = infer_field_candidates(all_keys, "answer")
    metric_fields = infer_field_candidates(all_keys, "metric")
    record_id_fields = infer_field_candidates(all_keys, "record_id")
    triple_key_fields = infer_field_candidates(all_keys, "triple_key")
    language_fields = infer_field_candidates(all_keys, "language")
    evaluators = _discover_evaluators(path, samples, all_keys)
    role = _infer_role(path, samples, evaluators, score_fields)
    usable, usable_note = _usable_flag(
        role,
        evaluators,
        score_fields,
        question_fields,
        answer_fields,
        metric_fields,
        record_id_fields,
        triple_key_fields,
    )
    notes.append(usable_note)
    if "qvq" in normalize_text(relpath(path)):
        notes.append("qvq alias is treated as QwQ-plus only with this inventory note")

    row = {
        "file_path": relpath(path),
        "exists": exists,
        "file_type": path.suffix.lower().lstrip("."),
        "num_records": num_records,
        "top_level_keys": top_keys,
        "score_like_fields": score_fields,
        "judge_name_candidates": evaluators,
        "question_like_fields": question_fields,
        "answer_like_fields": answer_fields,
        "metric_like_fields": metric_fields,
        "record_id_like_fields": record_id_fields,
        "triple_key_like_fields": triple_key_fields,
        "language_fields": language_fields,
        "likely_evaluator": "; ".join(evaluators) if evaluators else "unknown",
        "likely_role": role,
        "usable_for_exp01": usable,
        "notes": "; ".join(notes),
    }
    return row, samples


def write_inventory_markdown(rows: list[dict[str, Any]]) -> None:
    role_counts = Counter(row["likely_role"] for row in rows)
    usable_counts = Counter(row["usable_for_exp01"] for row in rows)
    role_rows = [{"likely_role": key, "num_files": value} for key, value in sorted(role_counts.items())]
    usable_rows = [{"usable_for_exp01": key, "num_files": value} for key, value in sorted(usable_counts.items())]
    required_names = {
        "results_merge.jsonl",
        "merge_model_metric.jsonl",
        "groupby_metric_qwq_eval_en.jsonl",
        "groupby_metric_qwq_eval_zh.jsonl",
        "groupby_metric_r1_eval_en.jsonl",
        "groupby_metric_r1_eval_zh.jsonl",
        "groupby_metric_v3_eval_en.jsonl",
        "groupby_metric_v3_eval_zh.jsonl",
        "groupby_metric_4o_eval_en.jsonl",
        "groupby_metric_4o_eval_zh.jsonl",
        "sampled_merge_50_new.json",
        "sampled_merge_50_new_swift.json",
    }
    required_rows = [row for row in rows if Path(row["file_path"]).name in required_names]
    usable_rows_detail = [row for row in rows if row["usable_for_exp01"] in {"true", "maybe"}]
    text = f"""# Exp1 Judge Score Inventory

This inventory scans likely judge-score files and marks synthetic/sample sources as excluded. It is
descriptive only; missing evaluator predictions are not inferred or filled.

## Role Counts

{markdown_table(role_rows, ["likely_role", "num_files"])}

## Usability Counts

{markdown_table(usable_rows, ["usable_for_exp01", "num_files"])}

## Required Candidate Files

{markdown_table(required_rows, INVENTORY_COLUMNS, max_rows=80)}

## Usable or Maybe-Usable Sources

{markdown_table(usable_rows_detail, INVENTORY_COLUMNS, max_rows=120)}

## All Scanned Candidate Files

{markdown_table(rows, INVENTORY_COLUMNS, max_rows=240)}
"""
    write_text(EXP01_OUTPUT_DIR / "judge_score_inventory.md", text)


def run_inventory() -> list[dict[str, Any]]:
    ensure_exp01_dirs()
    rows: list[dict[str, Any]] = []
    source_samples: dict[str, list[dict[str, Any]]] = {}
    for path in collect_candidate_paths():
        row, samples = inventory_file(path)
        rows.append(row)
        if row["usable_for_exp01"] in {"true", "maybe"} and samples:
            source_samples[row["file_path"]] = [truncate_for_json(sample) for sample in samples[:3]]
    rows = sorted(rows, key=lambda row: row["file_path"])
    write_csv(EXP01_TABLES_DIR / "judge_score_inventory.csv", rows, INVENTORY_COLUMNS)
    write_json(EXP01_SAMPLES_DIR / "judge_source_samples.json", source_samples)
    write_inventory_markdown(rows)
    return rows


def main() -> None:
    rows = run_inventory()
    found = [row for row in rows if row["usable_for_exp01"] == "true"]
    print(f"Inventoried {len(rows)} candidate files; usable_for_exp01=true: {len(found)}")
    print(f"Outputs: {relpath(EXP01_TABLES_DIR / 'judge_score_inventory.csv')}")


if __name__ == "__main__":
    main()

