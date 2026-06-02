"""Align existing evaluator predictions to the locked Exp1 test split."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.audit import (
    EVALUATORS,
    EXP01_OUTPUT_DIR,
    EXP01_TABLES_DIR,
    PROCESSED_DATASET_PATH,
    TEST_SPLIT_PATH,
    canonical_evaluator,
    collect_candidate_paths,
    ensure_exp01_dirs,
    evaluator_from_path,
    is_synthetic_or_sampled_path,
    normalized_qam_key,
    parse_score_to_1_5,
    raw_qam_key,
    read_jsonl,
    relpath,
    round_label,
    write_csv,
    write_jsonl,
)
from thesis_exp.src.edujudge.utils.io import iter_json_records
from thesis_exp.src.edujudge.utils.text_norm import stringify


@dataclass(frozen=True)
class PredictionCandidate:
    evaluator: str
    raw_score: Any
    source_file: str
    source_row_index: int
    alignment_record_id: str
    alignment_triple_key: str
    qam_key: str
    raw_qam_key: str
    source_scale: str
    note: str


def _source_scale(path: Path, record: dict[str, Any]) -> str:
    if "score_mean" in record or path.name == "merge_model_metric.jsonl":
        return "1-10"
    if "groupby_metric" in path.name.lower():
        return "1-10"
    return "auto"


def _candidate_from_record(
    record: dict[str, Any],
    path: Path,
    row_idx: int,
    evaluator: str,
    raw_score: Any,
    metric_override: object = None,
    note: str = "",
) -> PredictionCandidate:
    return PredictionCandidate(
        evaluator=evaluator,
        raw_score=raw_score,
        source_file=relpath(path),
        source_row_index=row_idx,
        alignment_record_id=stringify(record.get("record_id")),
        alignment_triple_key=stringify(record.get("triple_key")),
        qam_key=normalized_qam_key(record, metric_override=metric_override, canonical_metric=True),
        raw_qam_key=raw_qam_key(record, metric_override=metric_override),
        source_scale=_source_scale(path, record),
        note=note,
    )


def extract_candidates_from_record(record: dict[str, Any], path: Path, row_idx: int) -> list[PredictionCandidate]:
    out: list[PredictionCandidate] = []
    for field in ["judge_scores", "evaluate"]:
        value = record.get(field)
        if isinstance(value, dict):
            for key, raw_score in value.items():
                evaluator = canonical_evaluator(key)
                if evaluator:
                    out.append(_candidate_from_record(record, path, row_idx, evaluator, raw_score, note=f"{field}.{key}"))

    for spec in EVALUATORS:
        for key in [spec.name, spec.field_suffix, *spec.aliases]:
            if key in record:
                out.append(_candidate_from_record(record, path, row_idx, spec.name, record[key], note=f"top_level.{key}"))

    score_obj = record.get("score")
    source_evaluator = canonical_evaluator(record.get("eval")) or canonical_evaluator(record.get("evaluator")) or evaluator_from_path(path)
    if isinstance(score_obj, dict) and source_evaluator:
        for metric, raw_score in score_obj.items():
            out.append(
                _candidate_from_record(
                    record,
                    path,
                    row_idx,
                    source_evaluator,
                    raw_score,
                    metric_override=metric,
                    note=f"score_by_metric.{metric}",
                )
            )
    elif source_evaluator and "score" in record:
        out.append(_candidate_from_record(record, path, row_idx, source_evaluator, record.get("score"), note="score"))
    elif source_evaluator and "score_mean" in record:
        out.append(_candidate_from_record(record, path, row_idx, source_evaluator, record.get("score_mean"), note="score_mean"))
    return out


def iter_source_candidates() -> list[PredictionCandidate]:
    """Return prediction candidates in preferred alignment order."""
    ordered_paths: list[Path] = []
    for path in [TEST_SPLIT_PATH, PROCESSED_DATASET_PATH]:
        if path.exists():
            ordered_paths.append(path)
    explicit_candidates = [
        "results_merge.jsonl",
        "report/results_merge_enriched.jsonl",
        "merge_model_metric.jsonl",
        "groupby_metric_qwq_eval_en.jsonl",
        "groupby_metric_qwq_eval_zh.jsonl",
        "groupby_metric_r1_eval_en.jsonl",
        "groupby_metric_r1_eval_zh.jsonl",
        "groupby_metric_v3_eval_en.jsonl",
        "groupby_metric_v3_eval_zh.jsonl",
        "download_raw/deepseek-r1_merged.jsonl",
        "download_raw/deepseek-v3_merged.jsonl",
        "download_raw/gpt-4o_merged.jsonl",
        "download_raw/qwq-plus_merged.jsonl",
        "deepseek-r1_merged.jsonl",
    ]
    repo_root = TEST_SPLIT_PATH.parents[4]
    for rel in explicit_candidates:
        path = repo_root / rel
        if path.exists() and path not in ordered_paths and not is_synthetic_or_sampled_path(path):
            ordered_paths.append(path)

    candidates: list[PredictionCandidate] = []
    for path in ordered_paths:
        if path.suffix.lower() not in {".json", ".jsonl"}:
            continue
        try:
            iterator = iter_json_records(path)
            for row_idx, record in iterator:
                if isinstance(record, dict):
                    candidates.extend(extract_candidates_from_record(record, path, row_idx))
        except Exception:
            continue
    return candidates


def build_indices(candidates: list[PredictionCandidate]) -> dict[str, dict[tuple[str, str], PredictionCandidate]]:
    indices: dict[str, dict[tuple[str, str], PredictionCandidate]] = {
        "record_id": {},
        "triple_key": {},
        "normalized_question_answer_metric": {},
        "normalized_question_answer_raw_metric": {},
    }
    for candidate in candidates:
        if candidate.alignment_record_id:
            indices["record_id"].setdefault((candidate.evaluator, candidate.alignment_record_id), candidate)
        if candidate.alignment_triple_key:
            indices["triple_key"].setdefault((candidate.evaluator, candidate.alignment_triple_key), candidate)
        if candidate.qam_key:
            indices["normalized_question_answer_metric"].setdefault((candidate.evaluator, candidate.qam_key), candidate)
        if candidate.raw_qam_key:
            indices["normalized_question_answer_raw_metric"].setdefault((candidate.evaluator, candidate.raw_qam_key), candidate)
    return indices


def select_candidate(
    record: dict[str, Any],
    evaluator: str,
    indices: dict[str, dict[tuple[str, str], PredictionCandidate]],
) -> tuple[PredictionCandidate | None, str]:
    lookups = [
        ("record_id", stringify(record.get("record_id"))),
        ("triple_key", stringify(record.get("triple_key"))),
        ("normalized_question_answer_metric", normalized_qam_key(record, canonical_metric=True)),
        ("normalized_question_answer_raw_metric", raw_qam_key(record)),
    ]
    for method, key in lookups:
        if not key:
            continue
        candidate = indices[method].get((evaluator, key))
        if candidate:
            return candidate, method
    return None, ""


BASE_FIELDS = [
    "record_id",
    "triple_key",
    "question_key",
    "answer_key",
    "question",
    "answer",
    "metric_raw",
    "metric_canonical",
    "metric_abbr",
    "metric_group",
    "scenario_canonical",
    "subject_canonical",
    "education_level_canonical",
    "language",
    "generator_model",
    "human_1_5",
    "human_2_5",
    "human_3_5",
    "human_mean_5",
    "label_5",
]


def align_predictions() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    ensure_exp01_dirs()
    test_records = read_jsonl(TEST_SPLIT_PATH)
    candidates = iter_source_candidates()
    indices = build_indices(candidates)
    aligned_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    coverage_stats: dict[str, dict[str, Any]] = {
        spec.name: {
            "evaluator": spec.name,
            "n_test": len(test_records),
            "n_aligned": 0,
            "n_valid_score": 0,
            "n_missing": 0,
            "n_invalid": 0,
            "source_files": set(),
            "alignment_methods": Counter(),
            "notes": [],
        }
        for spec in EVALUATORS
    }

    for record in test_records:
        out = {field: record.get(field) for field in BASE_FIELDS}
        for spec in EVALUATORS:
            suffix = spec.field_suffix
            candidate, method = select_candidate(record, spec.name, indices)
            raw_pred = candidate.raw_score if candidate else None
            out[f"raw_pred_{suffix}"] = raw_pred
            out[f"alignment_source_{suffix}"] = candidate.source_file if candidate else ""
            out[f"pred_{suffix}"] = None
            out[f"pred_label_{suffix}"] = None
            out[f"valid_{suffix}"] = False
            stats = coverage_stats[spec.name]
            if candidate is None:
                stats["n_missing"] += 1
                missing_rows.append(
                    {
                        "evaluator": spec.name,
                        "record_id": record.get("record_id"),
                        "triple_key": record.get("triple_key"),
                        "question_key": record.get("question_key"),
                        "answer_key": record.get("answer_key"),
                        "metric_canonical": record.get("metric_canonical"),
                        "reason": "no aligned judge score found",
                    }
                )
                continue
            stats["n_aligned"] += 1
            stats["source_files"].add(candidate.source_file)
            stats["alignment_methods"][method] += 1
            parsed = parse_score_to_1_5(raw_pred, source_scale=candidate.source_scale)
            if parsed.valid and parsed.value is not None:
                stats["n_valid_score"] += 1
                out[f"pred_{suffix}"] = parsed.value
                out[f"pred_label_{suffix}"] = round_label(parsed.value)
                out[f"valid_{suffix}"] = True
            else:
                stats["n_invalid"] += 1
                invalid_rows.append(
                    {
                        "evaluator": spec.name,
                        "record_id": record.get("record_id"),
                        "triple_key": record.get("triple_key"),
                        "metric_canonical": record.get("metric_canonical"),
                        "raw_score": stringify(raw_pred),
                        "source_file": candidate.source_file,
                        "source_row_index": candidate.source_row_index,
                        "alignment_method": method,
                        "parse_note": parsed.note,
                    }
                )
        aligned_rows.append(out)

    coverage_rows: list[dict[str, Any]] = []
    for spec in EVALUATORS:
        stats = coverage_stats[spec.name]
        n_test = stats["n_test"]
        n_aligned = stats["n_aligned"]
        n_valid = stats["n_valid_score"]
        primary_method = stats["alignment_methods"].most_common(1)[0][0] if stats["alignment_methods"] else ""
        notes = []
        if spec.name == "EduBenchEvaluator" and n_aligned == 0:
            notes.append("EduBenchEvaluator predictions not found; Exp2 should reproduce the CE baseline.")
        coverage_rows.append(
            {
                "evaluator": spec.name,
                "n_test": n_test,
                "n_aligned": n_aligned,
                "coverage": n_aligned / n_test if n_test else 0,
                "n_valid_score": n_valid,
                "valid_score_rate": n_valid / n_aligned if n_aligned else 0,
                "n_missing": stats["n_missing"],
                "n_invalid": stats["n_invalid"],
                "source_files": sorted(stats["source_files"]),
                "primary_alignment_method": primary_method,
                "notes": " ".join(notes),
            }
        )
    return aligned_rows, coverage_rows, missing_rows, invalid_rows


def run_alignment() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    aligned_rows, coverage_rows, missing_rows, invalid_rows = align_predictions()
    write_jsonl(EXP01_OUTPUT_DIR / "predictions_aligned.jsonl", aligned_rows)
    write_csv(EXP01_TABLES_DIR / "alignment_coverage.csv", coverage_rows)
    write_csv(EXP01_TABLES_DIR / "missing_judge_scores.csv", missing_rows)
    write_csv(EXP01_TABLES_DIR / "invalid_judge_outputs.csv", invalid_rows)
    return aligned_rows, coverage_rows


def main() -> None:
    aligned_rows, coverage_rows = run_alignment()
    print(f"Aligned rows: {len(aligned_rows)}")
    for row in coverage_rows:
        print(f"{row['evaluator']}: coverage={row['coverage']:.3f}, valid={row['valid_score_rate']:.3f}")
    print(f"Outputs: {relpath(EXP01_OUTPUT_DIR / 'predictions_aligned.jsonl')}")


if __name__ == "__main__":
    main()
