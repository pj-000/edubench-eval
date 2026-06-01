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
    convert_score_to_five,
    detect_score_scale,
    infer_education_level,
    infer_language,
    infer_subject,
    round_half_up,
    stable_language_label,
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


def _record_from_results_merge(row_index: int, raw: dict[str, Any]) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
    missing_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []

    source_file = relpath(PRIMARY_SOURCE)
    question = raw.get("question")
    answer = raw.get("answer")
    metric_raw = raw.get("metric")
    scenario_raw = raw.get("task")
    evaluate = raw.get("evaluate") if isinstance(raw.get("evaluate"), dict) else {}

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
        return None, missing_rows, invalid_rows

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
        return None, missing_rows, invalid_rows

    human_mean_raw = mean(scores)
    human_mean_5, label_5, scale, method, clipped = _score_mapping(scores)
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
        return None, missing_rows, invalid_rows
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
    subject_raw, subject_canonical = infer_subject(question)
    edu_raw, edu_canonical = infer_education_level(question)
    language = stable_language_label(infer_language(question, answer, metric_raw))

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
        "subject_raw": subject_raw,
        "subject_canonical": subject_canonical,
        "education_level_raw": edu_raw,
        "education_level_canonical": edu_canonical,
        "language": language,
        "generator_model": raw.get("model"),
        "answer_model": raw.get("model"),
        "human_1": humans["human_1"],
        "human_2": humans["human_2"],
        "human_3": humans["human_3"],
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
        },
        "triple_key": triple_key,
        "question_key": question_key,
        "answer_key": answer_key,
    }
    return record, missing_rows, invalid_rows


def _load_results_merge_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
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
        record, missing_rows, invalid_rows = _record_from_results_merge(row_index, obj)
        missing.extend(missing_rows)
        invalid.extend(invalid_rows)
        if record is not None:
            records.append(record)
    return records, missing, invalid


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
        row = {
            "raw_metric": item["raw_metric"],
            "canonical_metric": mapped["canonical_metric"],
            "metric_abbr": mapped["metric_abbr"],
            "metric_group": mapped["metric_group"],
            "language": mapped["language"],
            "source_file": item["source_file"],
            "confidence": mapped["confidence"],
            "notes": mapped["notes"],
        }
        metric_rows.append(row)
        if mapped["canonical_metric"] == "unknown":
            unmapped_metrics.append(row)

    scenario_rows: list[dict[str, Any]] = []
    unmapped_scenarios: list[dict[str, Any]] = []
    for item in scenario_inputs:
        mapped = canonicalize_scenario(item["raw_scenario"])
        row = {
            "raw_scenario": item["raw_scenario"],
            "canonical_scenario": mapped["canonical_scenario"],
            "scenario_abbr": mapped["scenario_abbr"],
            "student_or_teacher_oriented": mapped["student_or_teacher_oriented"],
            "source_file": item["source_file"],
            "confidence": mapped["confidence"],
            "notes": mapped["notes"],
        }
        scenario_rows.append(row)
        if mapped["canonical_scenario"] == "unknown":
            unmapped_scenarios.append(row)

    write_csv(
        TABLES_DIR / "metric_mapping.csv",
        metric_rows,
        ["raw_metric", "canonical_metric", "metric_abbr", "metric_group", "language", "source_file", "confidence", "notes"],
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
        ],
    )
    write_csv(
        TABLES_DIR / "unmapped_metrics.csv",
        unmapped_metrics,
        ["raw_metric", "canonical_metric", "metric_abbr", "metric_group", "language", "source_file", "confidence", "notes"],
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
        "human_mean_raw",
        "human_mean_5",
        "label_5",
        "triple_key",
        "question_key",
        "answer_key",
    ]
    return [{field: row.get(field) for field in fields} for row in records]


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
        f"`{relpath(PRIMARY_SOURCE)}` is selected as the primary main dataset source because it contains one row per scored item with `question`, `answer`, `metric`, `task`, generator `model`, and an `evaluate` object that includes `human_1`, `human_2`, and `human_3` together with automatic judge scores.",
        "",
        "`human_1.jsonl`, `human_2.jsonl`, and `human_3.jsonl` are treated as real human annotation sources for provenance and schema profiling, but they are not directly concatenated into the main dataset because they use a 1-10 score scale and would duplicate or partially overlap the already merged scored items.",
        "",
        "`sampled_merge_50_new.json` and `sampled_merge_50_new_swift.json` are inventory-only synthetic/augmented files and are excluded from `edubench_scoring_all.jsonl`.",
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
        "Records with all available human scores in 1-5 are kept as already normalized and rounded to `label_5`. Records with a 1-10 scale would be mapped using the repository `5-grades.py` rule: 1-2->1, 3-4->2, 5-6->3, 7-8->4, 9-10->5. Ambiguous scales are excluded from the main processed JSONL and recorded in `invalid_or_ambiguous_scores.csv`.",
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
        "# Data Card: Exp 0 EduBench Human-Scored Dataset",
        "",
        "## Source Selection",
        "",
        f"Primary source: `{relpath(PRIMARY_SOURCE)}`.",
        "",
        "The selected source has merged scored items with three human annotation fields and automatic judge metadata. Synthetic/sample files are excluded from the processed dataset.",
        "",
        "## Dataset Statistics",
        "",
        md_table(
            [
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
                {"check": "total scored items", "observed": total, "reference": "5536 = 3318 train pool + 2218 held-out test", "note": "matches reference total" if can_match_reference else "does not match reference total"},
                {"check": "unique triple_key", "observed": len({row["triple_key"] for row in records}), "reference": "question-answer-metric scored item", "note": "used for evaluator-vs-human split"},
                {"check": "unique question_key", "observed": len({row["question_key"] for row in records}), "reference": "not fixed in task", "note": "used for robustness split"},
                {"check": "unique answer_key", "observed": len({row["answer_key"] for row in records}), "reference": "not fixed in task", "note": "used for leakage diagnostics"},
                {"check": "generator_model distribution", "observed": len({row["generator_model"] for row in records}), "reference": "5 generated models", "note": "see distribution table"},
                {"check": "canonical metrics", "observed": unique_metric_count, "reference": "12", "note": "aligned" if unique_metric_count == 12 else "requires review"},
                {"check": "canonical scenarios", "observed": unique_scenario_count, "reference": "9", "note": "aligned" if unique_scenario_count == 9 else "requires review"},
                {"check": "canonical subjects", "observed": len({row["subject_canonical"] for row in records if row["subject_canonical"] != "unknown"}), "reference": "25 canonical subjects", "note": "inferred from text where explicit fields are absent"},
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
    records, missing_rows, invalid_rows = _load_results_merge_records()
    duplicate_rows = _duplicate_triples(records)

    write_jsonl(PROCESSED_PATH, records)
    write_csv(TABLES_DIR / "dataset_rows.csv", _dataset_row_table(records))
    write_csv(
        TABLES_DIR / "invalid_or_ambiguous_scores.csv",
        invalid_rows,
        [
            "source_file",
            "source_row_index",
            "human_scores",
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
    write_json(SAMPLES_DIR / "standardized_sample_records.json", records[:5])

    _write_field_decisions(records, unmapped_metrics, unmapped_scenarios)
    _write_data_card(records, duplicate_rows)
    return records


def main() -> None:
    records = build_dataset()
    print(f"Wrote {len(records)} standardized human-scored rows to {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
