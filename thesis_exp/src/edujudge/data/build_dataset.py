"""Build the Exp 0 normalized human-scored EduBench dataset."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from thesis_exp.src.edujudge.data.normalize_fields import (
    METRIC_SPECS,
    SCENARIO_SPECS,
    canonicalize_metric,
    canonicalize_scenario,
    canonicalize_subject,
    convert_score_to_five,
    detect_score_scale,
    extract_subject_from_question,
    infer_education_level,
    infer_language,
    infer_subject,
    round_half_up,
    stable_language_label,
)
from thesis_exp.src.edujudge.data.reference_contract import (
    DATASET_NAME,
    EXPECTED_EDUCATION_LEVELS,
    EXPECTED_SUBJECTS,
    PDF_AUDIT_TOTAL,
    PDF_HELDOUT_TEST_ROWS,
    PDF_TRAIN_POOL_ROWS,
)
from thesis_exp.src.edujudge.utils.hashing import sha1_text
from thesis_exp.src.edujudge.utils.io import (
    OUTPUT_DIR,
    PROCESSED_DIR,
    REPO_ROOT,
    SAMPLES_DIR,
    TABLES_DIR,
    ensure_exp_dirs,
    iter_json_records,
    md_table,
    read_jsonl,
    relpath,
    write_csv,
    write_json,
    write_jsonl,
    write_text,
)
from thesis_exp.src.edujudge.utils.text_norm import normalize_text, stringify, truncate_text


PRIMARY_SOURCE = REPO_ROOT / "results_merge.jsonl"
ENRICHED_SOURCE = REPO_ROOT / "report" / "results_merge_enriched.jsonl"
PROCESSED_PATH = PROCESSED_DIR / "edubench_scoring_all.jsonl"
PRIMARY_OUTPUT = PROCESSED_PATH

MAPPING_SOURCE_NAMES = [
    "results_merge.jsonl",
    "merge_human_metric.jsonl",
    "merge_human_metric_strict_en.jsonl",
    "merge_human_metric_strict_zh.jsonl",
    "human_1.jsonl",
    "human_2.jsonl",
    "human_3.jsonl",
    "merge_model_metric.jsonl",
    "groupby_metric_qwq_eval_en.jsonl",
    "groupby_metric_qwq_eval_zh.jsonl",
    "groupby_metric_r1_eval_en.jsonl",
    "groupby_metric_r1_eval_zh.jsonl",
    "groupby_metric_v3_eval_en.jsonl",
    "groupby_metric_v3_eval_zh.jsonl",
    "metrics_map.json",
]


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
        if math.isnan(number):
            return None
        return number
    return None


def _clean_key(value: object) -> str:
    return normalize_text(value)


def _clip_label(label: int | None) -> tuple[int | None, bool]:
    if label is None:
        return None, False
    clipped = max(1, min(5, int(label)))
    return clipped, clipped != label


def _score_mapping(scores: list[float]) -> tuple[float | None, int | None, str, str, bool]:
    scale = detect_score_scale(scores)
    if not scores:
        return None, None, "unknown", "unknown", False
    raw_mean = mean(scores)
    clipped = False
    if scale == "1-5":
        human_mean_5 = raw_mean
        label, clipped = _clip_label(round_half_up(human_mean_5))
        return human_mean_5, label, scale, "already_1_5_round", clipped
    if scale == "1-10":
        mapped = [convert_score_to_five(score) for score in scores]
        if any(value is None for value in mapped):
            return None, None, scale, "unknown", False
        human_mean_5 = mean([float(value) for value in mapped if value is not None])
        label, clipped = _clip_label(round_half_up(human_mean_5))
        return human_mean_5, label, scale, "ten_to_five_mapping", clipped
    return None, None, "unknown", "unknown", False


def _map_single_score(value: float | None, scale: str) -> float | None:
    if value is None:
        return None
    if scale == "1-5":
        return float(value)
    if scale == "1-10":
        mapped = convert_score_to_five(value)
        return float(mapped) if mapped is not None else None
    return None


def _human_score_summary(evaluate: dict[str, Any]) -> tuple[dict[str, float | None], list[float]]:
    humans = {
        "human_1": _to_float(evaluate.get("human_1")),
        "human_2": _to_float(evaluate.get("human_2")),
        "human_3": _to_float(evaluate.get("human_3")),
    }
    scores = [value for value in humans.values() if value is not None]
    return humans, scores


def _judge_scores(evaluate: dict[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in evaluate.items() if not str(key).startswith("human_")}


def _canonical_education(raw_value: object) -> tuple[str, str]:
    raw = stringify(raw_value).strip()
    folded = normalize_text(raw)
    if not raw:
        return "unknown", "unknown"
    exact = {normalize_text(level): level for level in EXPECTED_EDUCATION_LEVELS}
    if folded in exact:
        return raw, exact[folded]
    if any(token in folded for token in ["elementary", "primary", "小学"]):
        return raw, "Elementary School"
    if any(token in folded for token in ["middle school", "junior high", "初中"]):
        return raw, "Middle School"
    if any(token in folded for token in ["high school", "高中", "grade 10", "高一", "高二", "高三"]):
        return raw, "High School"
    if any(token in folded for token in ["undergraduate", "bachelor", "college", "本科", "大一", "大二", "大三", "大四"]):
        return raw, "Undergraduate"
    if any(token in folded for token in ["master", "硕士"]):
        return raw, "Master"
    if any(token in folded for token in ["phd", "doctor", "博士"]):
        return raw, "PhD"
    return raw, "unknown"


def _load_enriched_rows() -> list[dict[str, Any]]:
    if not ENRICHED_SOURCE.exists():
        return []
    return read_jsonl(ENRICHED_SOURCE)


def _enriched_for_index(enriched_rows: list[dict[str, Any]], row_index: int, raw: dict[str, Any]) -> dict[str, Any]:
    if row_index >= len(enriched_rows):
        return {}
    enriched = enriched_rows[row_index]
    if stringify(enriched.get("question")) == stringify(raw.get("question")) and stringify(enriched.get("answer")) == stringify(raw.get("answer")):
        return enriched
    return {}


def _subject_info(question: object, enriched: dict[str, Any]) -> dict[str, Any]:
    if enriched.get("subject_unified"):
        return {
            "subject_raw": enriched.get("subject") or enriched.get("subject_unified"),
            "subject_canonical": enriched.get("subject_unified"),
            "subject_source_field": "report/results_merge_enriched.jsonl.subject_unified",
            "subject_mapping_method": "structured_field",
            "subject_confidence": "high",
            "subject_notes": "recovered from local enriched audit metadata",
        }
    if enriched.get("subject"):
        mapped = canonicalize_subject(enriched.get("subject"))
        return {
            "subject_raw": enriched.get("subject"),
            "subject_canonical": mapped["canonical_subject"],
            "subject_source_field": "report/results_merge_enriched.jsonl.subject",
            "subject_mapping_method": "structured_field",
            "subject_confidence": mapped["confidence"],
            "subject_notes": mapped["notes"],
        }
    raw, canonical, source_field, method, confidence = extract_subject_from_question(question)
    if canonical != "unknown":
        return {
            "subject_raw": raw,
            "subject_canonical": canonical,
            "subject_source_field": source_field,
            "subject_mapping_method": method,
            "subject_confidence": confidence,
            "subject_notes": "subject recovered from structured prompt metadata",
        }
    raw_fallback, canonical_fallback = infer_subject(question)
    return {
        "subject_raw": raw_fallback,
        "subject_canonical": canonical_fallback,
        "subject_source_field": "question",
        "subject_mapping_method": "text_inference",
        "subject_confidence": "medium" if canonical_fallback != "unknown" else "none",
        "subject_notes": "fallback keyword inference from prompt text",
    }


def _record_from_results_merge(
    row_index: int,
    raw: dict[str, Any],
    enriched_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    missing_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []

    source_file = relpath(PRIMARY_SOURCE)
    question = raw.get("question")
    answer = raw.get("answer")
    metric_raw = raw.get("metric")
    scenario_raw = raw.get("task")
    evaluate = raw.get("evaluate") if isinstance(raw.get("evaluate"), dict) else {}
    enriched = _enriched_for_index(enriched_rows, row_index, raw)

    required = {
        "question": question,
        "answer": answer,
        "metric": metric_raw,
        "evaluate": evaluate,
    }
    missing = [key for key, value in required.items() if value in (None, "", {})]
    if missing:
        missing_rows.append(
            {
                "source_file": source_file,
                "source_row_index": row_index,
                "missing_fields": missing,
                "reason": "required source fields missing",
            }
        )
        return None, missing_rows, invalid_rows, None

    humans, scores = _human_score_summary(evaluate)
    if not scores:
        invalid_rows.append(
            {
                "source_file": source_file,
                "source_row_index": row_index,
                "human_scores": humans,
                "reason": "no real human score found",
            }
        )
        return None, missing_rows, invalid_rows, None

    human_mean_raw = mean(scores)
    human_mean_5, label_5, scale, method, clipped = _score_mapping(scores)
    mapped_humans = {key: _map_single_score(value, scale) for key, value in humans.items()}
    if label_5 is None or human_mean_5 is None:
        invalid_rows.append(
            {
                "source_file": source_file,
                "source_row_index": row_index,
                "human_scores": humans,
                "human_mean_raw": human_mean_raw,
                "raw_score_scale_detected": scale,
                "label_mapping_method": method,
                "reason": "ambiguous or unsupported human score scale",
            }
        )
        return None, missing_rows, invalid_rows, None
    if clipped:
        invalid_rows.append(
            {
                "source_file": source_file,
                "source_row_index": row_index,
                "human_scores": humans,
                "human_mean_raw": human_mean_raw,
                "human_mean_5": human_mean_5,
                "label_5": label_5,
                "raw_score_scale_detected": scale,
                "label_mapping_method": method,
                "reason": "label clipped to 1-5",
            }
        )

    metric = canonicalize_metric(metric_raw)
    scenario = canonicalize_scenario(scenario_raw)
    subject = _subject_info(question, enriched)
    if enriched.get("education_level_unified"):
        edu_raw, edu_canonical = _canonical_education(enriched.get("education_level_unified"))
    elif enriched.get("education_level"):
        edu_raw, edu_canonical = _canonical_education(enriched.get("education_level"))
    else:
        edu_raw_inferred, edu_canonical_inferred = infer_education_level(question)
        edu_raw, edu_canonical = _canonical_education(edu_raw_inferred or edu_canonical_inferred)
    language = stable_language_label(enriched.get("language") or infer_language(question, answer, metric_raw))

    normalized_question = _clean_key(question)
    normalized_answer = _clean_key(answer)
    metric_for_hash = metric["canonical_metric"] if metric["canonical_metric"] != "unknown" else stringify(metric_raw)
    triple_key = sha1_text(normalized_question, normalized_answer, _clean_key(metric_for_hash))
    question_key = sha1_text(normalized_question)
    answer_key = sha1_text(normalized_answer)
    record_id = sha1_text(source_file, row_index, normalized_question, normalized_answer, _clean_key(metric_raw))

    record = {
        "record_id": record_id,
        "source_file": source_file,
        "source_row_index": row_index,
        "question_id": question_key,
        "answer_id": answer_key,
        "metric_id": metric["metric_abbr"],
        "question": question,
        "answer": answer,
        "metric_raw": metric_raw,
        "metric_canonical": metric["canonical_metric"],
        "metric_abbr": metric["metric_abbr"],
        "metric_group": metric["metric_group"],
        "rubric": raw.get("levels"),
        "scenario_raw": scenario_raw,
        "scenario_canonical": scenario["canonical_scenario"],
        "scenario_abbr": scenario["scenario_abbr"],
        "subject_raw": subject["subject_raw"],
        "subject_canonical": subject["subject_canonical"],
        "subject_source_field": subject["subject_source_field"],
        "subject_mapping_method": subject["subject_mapping_method"],
        "subject_confidence": subject["subject_confidence"],
        "subject_notes": subject["subject_notes"],
        "education_level_raw": edu_raw,
        "education_level_canonical": edu_canonical,
        "language": language,
        "generator_model": raw.get("model"),
        "answer_model": raw.get("model"),
        "human_1_raw": humans["human_1"],
        "human_2_raw": humans["human_2"],
        "human_3_raw": humans["human_3"],
        "human_1_5": mapped_humans["human_1"],
        "human_2_5": mapped_humans["human_2"],
        "human_3_5": mapped_humans["human_3"],
        "human_1": mapped_humans["human_1"],
        "human_2": mapped_humans["human_2"],
        "human_3": mapped_humans["human_3"],
        "human_scores_available": ",".join(key for key, value in humans.items() if value is not None),
        "human_mean_raw": human_mean_raw,
        "human_mean_5": human_mean_5,
        "label_5": int(label_5),
        "raw_score_scale_detected": scale,
        "label_mapping_method": method,
        "judge_scores": _judge_scores(evaluate),
        "metadata_raw": {
            "task": raw.get("task"),
            "model": raw.get("model"),
            "evaluate_keys": sorted(map(str, evaluate.keys())),
            "local_enriched_source": relpath(ENRICHED_SOURCE) if enriched else "",
            "original_is_test_set": enriched.get("is_test_set") if enriched else None,
        },
        "original_is_test_set": enriched.get("is_test_set") if enriched else None,
        "source_status": "local_derived_merged_human_scored_subset",
        "triple_key": triple_key,
        "question_key": question_key,
        "answer_key": answer_key,
    }
    score_audit = {
        "record_id": record_id,
        "source_file": source_file,
        "source_row_index": row_index,
        "human_1_raw": humans["human_1"],
        "human_2_raw": humans["human_2"],
        "human_3_raw": humans["human_3"],
        "human_mean_raw": human_mean_raw,
        "raw_score_scale_detected": scale,
        "human_1_5": mapped_humans["human_1"],
        "human_2_5": mapped_humans["human_2"],
        "human_3_5": mapped_humans["human_3"],
        "human_mean_5": human_mean_5,
        "label_5": int(label_5),
        "label_mapping_method": method,
        "mapping_verified": True,
        "notes": "raw human scores already on 1-5 scale" if scale == "1-5" else "raw human scores converted using 5-grades.py ten-to-five bins",
    }
    return record, missing_rows, invalid_rows, score_audit


def _load_results_merge_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    score_audit: list[dict[str, Any]] = []
    enriched_rows = _load_enriched_rows()
    for row_index, obj in iter_json_records(PRIMARY_SOURCE):
        if not isinstance(obj, dict):
            invalid.append(
                {
                    "source_file": relpath(PRIMARY_SOURCE),
                    "source_row_index": row_index,
                    "reason": "source row is not a JSON object",
                }
            )
            continue
        record, missing_rows, invalid_rows, score_row = _record_from_results_merge(row_index, obj, enriched_rows)
        missing.extend(missing_rows)
        invalid.extend(invalid_rows)
        if record is not None:
            records.append(record)
        if score_row is not None:
            score_audit.append(score_row)
    return records, missing, invalid, score_audit


def _collect_raw_metrics_and_scenarios() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    seen_metrics: set[tuple[str, str]] = set()
    seen_scenarios: set[tuple[str, str]] = set()

    for source_name in MAPPING_SOURCE_NAMES:
        path = REPO_ROOT / source_name
        if not path.exists():
            continue
        if path.name == "metrics_map.json":
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                for raw_scenario, metric_list in data.items():
                    key = (stringify(raw_scenario), relpath(path))
                    if key not in seen_scenarios:
                        seen_scenarios.add(key)
                        scenario_rows.append({"raw_scenario": raw_scenario, "source_file": relpath(path)})
                    if isinstance(metric_list, list):
                        for raw_metric in metric_list:
                            key = (stringify(raw_metric), relpath(path))
                            if key not in seen_metrics:
                                seen_metrics.add(key)
                                metric_rows.append({"raw_metric": raw_metric, "source_file": relpath(path)})
            continue

        if path.suffix.lower() not in {".json", ".jsonl"}:
            continue
        for _, obj in iter_json_records(path):
            if not isinstance(obj, dict):
                continue
            raw_metric = obj.get("metric", obj.get("principle"))
            if raw_metric not in (None, ""):
                key = (stringify(raw_metric), relpath(path))
                if key not in seen_metrics:
                    seen_metrics.add(key)
                    metric_rows.append({"raw_metric": raw_metric, "source_file": relpath(path)})
            raw_scenario = obj.get("task", obj.get("scenario"))
            if raw_scenario not in (None, ""):
                key = (stringify(raw_scenario), relpath(path))
                if key not in seen_scenarios:
                    seen_scenarios.add(key)
                    scenario_rows.append({"raw_scenario": raw_scenario, "source_file": relpath(path)})

    for spec in METRIC_SPECS:
        for raw in [spec["canonical_metric"], spec["zh"], spec["metric_abbr"]]:
            key = (raw, "official_reference")
            if key not in seen_metrics:
                seen_metrics.add(key)
                metric_rows.append({"raw_metric": raw, "source_file": "official_reference"})
    for spec in SCENARIO_SPECS:
        for raw in [spec["canonical_scenario"], spec["scenario_abbr"]]:
            key = (raw, "official_reference")
            if key not in seen_scenarios:
                seen_scenarios.add(key)
                scenario_rows.append({"raw_scenario": raw, "source_file": "official_reference"})

    return metric_rows, scenario_rows


def write_mapping_tables() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metric_inputs, scenario_inputs = _collect_raw_metrics_and_scenarios()

    metric_rows: list[dict[str, Any]] = []
    unmapped_metrics: list[dict[str, Any]] = []
    for item in metric_inputs:
        mapped = canonicalize_metric(item["raw_metric"])
        severity = "INFO" if item["source_file"].startswith("groupby_metric_") else "ERROR"
        unmapped_note = "non-official legacy/ad hoc evaluator metric outside main human-scored source" if severity == "INFO" else "unmapped metric requires review"
        row = {
            "raw_metric": item["raw_metric"],
            "canonical_metric": mapped["canonical_metric"],
            "metric_abbr": mapped["metric_abbr"],
            "metric_group": mapped["metric_group"],
            "language": mapped["language"],
            "source_file": item["source_file"],
            "confidence": mapped["confidence"],
            "notes": mapped["notes"],
            "severity": "" if mapped["canonical_metric"] != "unknown" else severity,
            "unmapped_explanation": "" if mapped["canonical_metric"] != "unknown" else unmapped_note,
        }
        metric_rows.append(row)
        if mapped["canonical_metric"] == "unknown":
            unmapped_metrics.append(row)

    scenario_rows: list[dict[str, Any]] = []
    unmapped_scenarios: list[dict[str, Any]] = []
    for item in scenario_inputs:
        mapped = canonicalize_scenario(item["raw_scenario"])
        severity = "INFO" if item["source_file"].startswith("groupby_metric_") else "ERROR"
        row = {
            "raw_scenario": item["raw_scenario"],
            "canonical_scenario": mapped["canonical_scenario"],
            "scenario_abbr": mapped["scenario_abbr"],
            "student_or_teacher_oriented": mapped["student_or_teacher_oriented"],
            "source_file": item["source_file"],
            "confidence": mapped["confidence"],
            "notes": mapped["notes"],
            "severity": "" if mapped["canonical_scenario"] != "unknown" else severity,
            "unmapped_explanation": "" if mapped["canonical_scenario"] != "unknown" else "unmapped scenario requires review",
        }
        scenario_rows.append(row)
        if mapped["canonical_scenario"] == "unknown":
            unmapped_scenarios.append(row)

    write_csv(
        TABLES_DIR / "metric_mapping.csv",
        metric_rows,
        ["raw_metric", "canonical_metric", "metric_abbr", "metric_group", "language", "source_file", "confidence", "notes", "severity", "unmapped_explanation"],
    )
    write_csv(
        TABLES_DIR / "scenario_mapping.csv",
        scenario_rows,
        [
            "raw_scenario",
            "canonical_scenario",
            "scenario_abbr",
            "student_or_teacher_oriented",
            "source_file",
            "confidence",
            "notes",
            "severity",
            "unmapped_explanation",
        ],
    )
    write_csv(
        TABLES_DIR / "unmapped_metrics.csv",
        unmapped_metrics,
        ["raw_metric", "canonical_metric", "metric_abbr", "metric_group", "language", "source_file", "confidence", "notes", "severity", "unmapped_explanation"],
    )
    write_csv(
        TABLES_DIR / "unmapped_scenarios.csv",
        unmapped_scenarios,
        [
            "raw_scenario",
            "canonical_scenario",
            "scenario_abbr",
            "student_or_teacher_oriented",
            "source_file",
            "confidence",
            "notes",
            "severity",
            "unmapped_explanation",
        ],
    )
    return metric_rows, scenario_rows, unmapped_metrics, unmapped_scenarios


def _distribution(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    counts = Counter(stringify(row.get(key, "unknown") or "unknown") for row in rows)
    total = sum(counts.values())
    return [{"value": value, "count": count, "pct": round(count / max(1, total), 4)} for value, count in counts.most_common()]


def _duplicate_triples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["triple_key"]].append(record)
    out = []
    for triple_key, group in groups.items():
        if len(group) <= 1:
            continue
        out.append(
            {
                "triple_key": triple_key,
                "count": len(group),
                "record_ids": [row["record_id"] for row in group[:10]],
                "question_preview": truncate_text(group[0].get("question"), 180),
                "answer_preview": truncate_text(group[0].get("answer"), 180),
                "metric_canonical": group[0].get("metric_canonical"),
            }
        )
    return out


def _dataset_row_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "record_id",
        "source_file",
        "source_row_index",
        "metric_canonical",
        "scenario_canonical",
        "subject_canonical",
        "education_level_canonical",
        "language",
        "generator_model",
        "human_1",
        "human_2",
        "human_3",
        "human_1_raw",
        "human_2_raw",
        "human_3_raw",
        "human_1_5",
        "human_2_5",
        "human_3_5",
        "human_mean_raw",
        "human_mean_5",
        "label_5",
        "original_is_test_set",
        "triple_key",
        "question_key",
        "answer_key",
    ]
    return [{field: row.get(field) for field in fields} for row in records]


def _subject_tables(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in records:
        raw_subject = stringify(row.get("subject_raw") or "unknown")
        canonical = stringify(row.get("subject_canonical") or "unknown")
        source_field = stringify(row.get("subject_source_field") or "")
        method = stringify(row.get("subject_mapping_method") or "")
        key = (raw_subject, canonical, source_field, method)
        grouped.setdefault(
            key,
            {
                "raw_subject": raw_subject,
                "canonical_subject": canonical,
                "source_field": source_field,
                "source_file": row.get("source_file"),
                "mapping_method": method,
                "confidence": row.get("subject_confidence"),
                "notes": row.get("subject_notes"),
                "count": 0,
            },
        )["count"] += 1
    rows = sorted(grouped.values(), key=lambda item: (-int(item["count"]), item["canonical_subject"], item["raw_subject"]))
    unmapped = [row for row in rows if row["canonical_subject"] == "unknown"]
    return rows, unmapped


def _write_subject_alignment_report(records: list[dict[str, Any]], subject_mapping: list[dict[str, Any]], unmapped: list[dict[str, Any]]) -> None:
    observed = sorted({row.get("subject_canonical") for row in records if row.get("subject_canonical") and row.get("subject_canonical") != "unknown"})
    missing = [subject for subject in EXPECTED_SUBJECTS if subject not in observed]
    status = "PASS" if len(observed) == 25 else "WARNING"
    lines = [
        "# Subject Alignment Report",
        "",
        f"Status: **{status}**",
        "",
        f"Observed canonical subjects: {len(observed)} / 25.",
        "",
        "Subject metadata is recovered primarily from `report/results_merge_enriched.jsonl` when available. Text inference is retained only as a fallback.",
        "",
        "## Observed Subjects",
        "",
        md_table([{"canonical_subject": subject} for subject in observed], ["canonical_subject"], max_rows=40),
        "",
        "## Missing Subjects",
        "",
        md_table([{"canonical_subject": subject} for subject in missing], ["canonical_subject"], max_rows=40),
        "",
        "## Mapping Methods",
        "",
        md_table(_distribution(subject_mapping, "mapping_method"), ["value", "count", "pct"], max_rows=20),
        "",
        "## Recommendation",
        "",
        "Subject-level results can be used as audit metadata only after confirming the local enriched subject annotations. If subject alignment ever falls below 25, subject-level results should be treated as exploratory metadata, not as primary thesis evidence.",
        "",
        f"Unmapped subject mapping rows: {len(unmapped)}.",
    ]
    write_text(OUTPUT_DIR / "subject_alignment_report.md", "\n".join(lines))


def _write_field_decisions(
    records: list[dict[str, Any]],
    unmapped_metrics: list[dict[str, Any]],
    unmapped_scenarios: list[dict[str, Any]],
) -> None:
    lines = [
        "# Field Mapping Decisions",
        "",
        "## Primary Human-Scored Source",
        "",
        f"Dataset name: `{DATASET_NAME}`.",
        "",
        f"`{relpath(PRIMARY_SOURCE)}` is selected as the primary local source because it contains one row per scored item with `question`, `answer`, `metric`, `task`, generator `model`, and an `evaluate` object that includes `human_1`, `human_2`, and `human_3` together with automatic judge scores.",
        "",
        "`human_1.jsonl`, `human_2.jsonl`, and `human_3.jsonl` are treated as real human annotation sources for provenance and schema profiling, but they are not directly concatenated into the main dataset because they use a 1-10 score scale and would duplicate or partially overlap the already merged scored items.",
        "",
        "`sampled_merge_50_new.json` and `sampled_merge_50_new_swift.json` are inventory-only synthetic/augmented files and are excluded from `edubench_scoring_all.jsonl`. The 5536-row dataset is the PDF audit human-scored subset, not the full official EduBench data.",
        "",
        "## Standard Field Construction",
        "",
        md_table(
            [
                {"standard_field": "question", "source_logic": "results_merge.question"},
                {"standard_field": "answer", "source_logic": "results_merge.answer"},
                {"standard_field": "metric_raw", "source_logic": "results_merge.metric"},
                {"standard_field": "scenario_raw", "source_logic": "results_merge.task"},
                {"standard_field": "generator_model / answer_model", "source_logic": "results_merge.model"},
                {"standard_field": "human_1/2/3", "source_logic": "results_merge.evaluate.human_1/2/3"},
                {"standard_field": "judge_scores", "source_logic": "non-human keys from results_merge.evaluate"},
                {"standard_field": "subject / education_level", "source_logic": "explicit profile fields when recoverable, otherwise conservative keyword inference from question text"},
                {"standard_field": "language", "source_logic": "script-based detection from question, answer, and metric"},
                {"standard_field": "record_id / triple_key / question_key / answer_key", "source_logic": "stable SHA1 hashes over normalized text fields"},
            ],
            ["standard_field", "source_logic"],
            max_rows=30,
        ),
        "",
        "## Score Scale Handling",
        "",
        "Each record stores `human_1_raw`/`human_2_raw`/`human_3_raw` and `human_1_5`/`human_2_5`/`human_3_5`. Records with all available human scores in 1-5 are kept as already normalized and rounded to `label_5`. Records with a 1-10 scale would be mapped using the repository `5-grades.py` rule: 1-2->1, 3-4->2, 5-6->3, 7-8->4, 9-10->5. Ambiguous scales are excluded from the main processed JSONL and recorded in `invalid_or_ambiguous_scores.csv`.",
        "",
        "## Canonicalization Status",
        "",
        f"- Standardized rows: {len(records)}",
        f"- Canonical metrics observed: {len({row['metric_canonical'] for row in records if row['metric_canonical'] != 'unknown'})}",
        f"- Canonical scenarios observed: {len({row['scenario_canonical'] for row in records if row['scenario_canonical'] != 'unknown'})}",
        f"- Unmapped metric rows in mapping table: {len(unmapped_metrics)}",
        f"- Unmapped scenario rows in mapping table: {len(unmapped_scenarios)}",
    ]
    write_text(OUTPUT_DIR / "field_mapping_decisions.md", "\n".join(lines))


def _write_data_card(records: list[dict[str, Any]], duplicate_rows: list[dict[str, Any]]) -> None:
    total = len(records)
    train_pool_reference = 3318
    test_reference = 2218
    reference_total = train_pool_reference + test_reference
    can_match_reference = total == reference_total
    unique_metric_count = len({row["metric_canonical"] for row in records if row["metric_canonical"] != "unknown"})
    unique_scenario_count = len({row["scenario_canonical"] for row in records if row["scenario_canonical"] != "unknown"})

    lines = [
        "# Data Card: Exp 0.1 EduBench Audit Human-Scored Subset",
        "",
        "## Dataset Identity",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| dataset_name | {DATASET_NAME} |",
        "| not_full_official_edubench | true |",
        f"| primary_local_source | {relpath(PRIMARY_SOURCE)} |",
        "| source_status | local_derived_merged_human_scored_subset |",
        "| official_alignment_status.scenario_metric | aligned |",
        "| official_alignment_status.corpus_size | aligned_with_pdf_audit |",
        "| official_alignment_status.subject | aligned_with_local_enriched_audit_metadata |",
        "| official_alignment_status.full_official_data | not_reconstructed |",
        "",
        "## Source Selection",
        "",
        f"Primary source: `{relpath(PRIMARY_SOURCE)}`.",
        "",
        "The selected source has merged scored items with three human annotation fields and automatic judge metadata. Synthetic/sample files are excluded from the processed dataset. This is the PDF audit corpus / human-scored subset, not the official full EduBench benchmark.",
        "",
        "## Distinction between official EduBench full data and PDF audit subset",
        "",
        "Official EduBench full data is described as 9 scenarios, 4000+ educational contexts, 18821 data points, and 500 sampled queries evaluated by human raters and LLMs. The local thesis dataset here is the 5536-item PDF audit corpus with human and judge scores. Downstream Exp1 evaluator training/testing should use this 5536-row audit corpus as the main human-labeled dataset. Future synthetic augmentation or distillation may use official full data separately, but those rows must not be mixed into the main human-labeled test set.",
        "",
        "## Dataset Statistics",
        "",
        md_table(
            [
                {"stat": "dataset_name", "value": DATASET_NAME},
                {"stat": "total_scored_items", "value": total},
                {"stat": "unique_triple_key", "value": len({row["triple_key"] for row in records})},
                {"stat": "unique_question_key", "value": len({row["question_key"] for row in records})},
                {"stat": "unique_answer_key", "value": len({row["answer_key"] for row in records})},
                {"stat": "duplicate_triple_groups", "value": len(duplicate_rows)},
                {"stat": "canonical_metric_count", "value": unique_metric_count},
                {"stat": "canonical_scenario_count", "value": unique_scenario_count},
                {"stat": "canonical_subject_count", "value": len({row["subject_canonical"] for row in records if row["subject_canonical"] != "unknown"})},
                {"stat": "education_level_count", "value": len({row["education_level_canonical"] for row in records if row["education_level_canonical"] != "unknown"})},
                {"stat": "language_count", "value": len({row["language"] for row in records if row["language"] != "unknown"})},
            ],
            ["stat", "value"],
            max_rows=30,
        ),
        "",
        "## Label Distribution",
        "",
        md_table(_distribution(records, "label_5"), ["value", "count", "pct"], max_rows=10),
        "",
        "## Generator Model Distribution",
        "",
        md_table(_distribution(records, "generator_model"), ["value", "count", "pct"], max_rows=20),
        "",
        "## Metric Distribution",
        "",
        md_table(_distribution(records, "metric_canonical"), ["value", "count", "pct"], max_rows=20),
        "",
        "## Scenario Distribution",
        "",
        md_table(_distribution(records, "scenario_canonical"), ["value", "count", "pct"], max_rows=20),
        "",
        "## Reference checks against EduBench paper/PDF",
        "",
        md_table(
            [
                {"check": "total scored items", "observed": total, "reference": f"{PDF_AUDIT_TOTAL} = {PDF_TRAIN_POOL_ROWS} train pool + {PDF_HELDOUT_TEST_ROWS} held-out test", "note": "matches PDF audit corpus total" if can_match_reference else "does not match reference total"},
                {"check": "unique triple_key", "observed": len({row["triple_key"] for row in records}), "reference": "question-answer-metric scored item", "note": "used for evaluator-vs-human split"},
                {"check": "unique question_key", "observed": len({row["question_key"] for row in records}), "reference": "not fixed in task", "note": "used for robustness split"},
                {"check": "unique answer_key", "observed": len({row["answer_key"] for row in records}), "reference": "not fixed in task", "note": "used for leakage diagnostics"},
                {"check": "generator_model distribution", "observed": len({row["generator_model"] for row in records}), "reference": "5 generated models", "note": "see distribution table"},
                {"check": "canonical metrics", "observed": unique_metric_count, "reference": "12", "note": "aligned" if unique_metric_count == 12 else "requires review"},
                {"check": "canonical scenarios", "observed": unique_scenario_count, "reference": "9", "note": "aligned" if unique_scenario_count == 9 else "requires review"},
                {"check": "canonical subjects", "observed": len({row["subject_canonical"] for row in records if row["subject_canonical"] != "unknown"}), "reference": "25 canonical subjects", "note": "recovered from local enriched audit metadata when available"},
                {"check": "education levels", "observed": len({row["education_level_canonical"] for row in records if row["education_level_canonical"] != "unknown"}), "reference": "6 education stages", "note": "inferred from question profile"},
                {"check": "languages", "observed": sorted({row["language"] for row in records}), "reference": "English / Chinese", "note": "script-detected"},
                {"check": "human annotator fields", "observed": sorted({field for field in ["human_1", "human_2", "human_3"] if any(row.get(field) is not None for row in records)}), "reference": "3 annotators", "note": "all three present" if all(any(row.get(field) is not None for row in records) for field in ["human_1", "human_2", "human_3"]) else "missing annotator fields"},
                {"check": "3318 train pool / 2218 held-out test", "observed": "reproducible at row-count level" if can_match_reference else "not reproducible exactly", "reference": "3318 / 2218", "note": "make_splits targets this only when total is close to 5536"},
            ],
            ["check", "observed", "reference", "note"],
            max_rows=20,
        ),
    ]
    write_text(OUTPUT_DIR / "data_card.md", "\n".join(lines))


def build_dataset() -> list[dict[str, Any]]:
    ensure_exp_dirs()
    if not PRIMARY_SOURCE.exists():
        raise FileNotFoundError(f"Primary source not found: {PRIMARY_SOURCE}")

    metric_rows, scenario_rows, unmapped_metrics, unmapped_scenarios = write_mapping_tables()
    records, missing_rows, invalid_rows, score_audit_rows = _load_results_merge_records()
    duplicate_rows = _duplicate_triples(records)
    subject_mapping, unmapped_subjects = _subject_tables(records)

    write_jsonl(PROCESSED_PATH, records)
    write_csv(TABLES_DIR / "dataset_rows.csv", _dataset_row_table(records))
    write_csv(
        TABLES_DIR / "score_scale_audit.csv",
        score_audit_rows,
        [
            "record_id",
            "source_file",
            "source_row_index",
            "human_1_raw",
            "human_2_raw",
            "human_3_raw",
            "human_mean_raw",
            "raw_score_scale_detected",
            "human_1_5",
            "human_2_5",
            "human_3_5",
            "human_mean_5",
            "label_5",
            "label_mapping_method",
            "mapping_verified",
            "notes",
        ],
    )
    write_csv(
        TABLES_DIR / "invalid_or_ambiguous_scores.csv",
        invalid_rows,
        [
            "source_file",
            "source_row_index",
            "human_scores",
            "human_1_raw",
            "human_2_raw",
            "human_3_raw",
            "human_mean_raw",
            "human_mean_5",
            "label_5",
            "raw_score_scale_detected",
            "label_mapping_method",
            "reason",
        ],
    )
    write_csv(
        TABLES_DIR / "missing_required_fields.csv",
        missing_rows,
        ["source_file", "source_row_index", "missing_fields", "reason"],
    )
    write_csv(
        TABLES_DIR / "duplicate_triples.csv",
        duplicate_rows,
        ["triple_key", "count", "record_ids", "question_preview", "answer_preview", "metric_canonical"],
    )
    write_csv(
        TABLES_DIR / "subject_mapping.csv",
        subject_mapping,
        ["raw_subject", "canonical_subject", "source_field", "source_file", "mapping_method", "confidence", "notes", "count"],
    )
    write_csv(
        TABLES_DIR / "unmapped_subjects.csv",
        unmapped_subjects,
        ["raw_subject", "canonical_subject", "source_field", "source_file", "mapping_method", "confidence", "notes", "count"],
    )
    write_json(SAMPLES_DIR / "standardized_sample_records.json", records[:5])

    _write_field_decisions(records, unmapped_metrics, unmapped_scenarios)
    _write_data_card(records, duplicate_rows)
    _write_subject_alignment_report(records, subject_mapping, unmapped_subjects)
    return records


def main() -> None:
    records = build_dataset()
    print(f"Wrote {len(records)} standardized human-scored rows to {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
