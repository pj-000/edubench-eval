"""Schema profiling for candidate JSON/JSONL sources."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.data.inventory_sources import likely_role_for
from thesis_exp.src.edujudge.data.normalize_fields import infer_field_candidates
from thesis_exp.src.edujudge.utils.io import (
    OUTPUT_DIR,
    REPO_ROOT,
    SAMPLES_DIR,
    TABLES_DIR,
    candidate_paths,
    ensure_exp_dirs,
    flatten_keys,
    iter_json_records,
    md_table,
    relpath,
    write_csv,
    write_json,
    write_text,
)
from thesis_exp.src.edujudge.utils.text_norm import truncate_text


def _json_source_paths() -> list[Path]:
    return [path for path in candidate_paths() if path.suffix.lower() in {".json", ".jsonl"}]


def _safe_sample_obj(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _safe_sample_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_sample_obj(v) for v in obj[:5]]
    if isinstance(obj, str):
        return truncate_text(obj, 1000)
    return obj


def profile_path(path: Path, random_seed: int = 42) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rng = random.Random(random_seed)
    all_keys: Counter[str] = Counter()
    nested_keys: set[str] = set()
    first_samples: list[dict[str, Any]] = []
    reservoir: list[dict[str, Any]] = []
    num_records = 0

    try:
        for row_index, obj in iter_json_records(path):
            num_records += 1
            if isinstance(obj, dict):
                keys = list(map(str, obj.keys()))
                all_keys.update(keys)
                nested_keys.update(flatten_keys(obj))
                sample = {"source_row_index": row_index, **_safe_sample_obj(obj)}
            else:
                sample = {"source_row_index": row_index, "__value__": _safe_sample_obj(obj)}
            if len(first_samples) < 5:
                first_samples.append(sample)
            if len(reservoir) < 20:
                reservoir.append(sample)
            else:
                j = rng.randint(0, num_records - 1)
                if j < 20:
                    reservoir[j] = sample
    except Exception as exc:  # noqa: BLE001 - profile should report failures
        row = {
            "file_path": relpath(path),
            "num_records": num_records,
            "all_keys": [],
            "key_presence_rate": {},
            "nested_keys": [],
            "score_like_fields": [],
            "question_like_fields": [],
            "answer_like_fields": [],
            "metric_like_fields": [],
            "subject_like_fields": [],
            "scenario_like_fields": [],
            "language_like_fields": [],
            "model_like_fields": [],
            "human_score_like_fields": [],
            "judge_score_like_fields": [],
            "profile_error": f"{type(exc).__name__}: {exc}",
            "likely_role": likely_role_for(path),
        }
        return row, first_samples + reservoir

    keys_sorted = sorted(all_keys)
    presence = {key: round(all_keys[key] / max(1, num_records), 4) for key in keys_sorted}
    row = {
        "file_path": relpath(path),
        "num_records": num_records,
        "all_keys": keys_sorted,
        "key_presence_rate": presence,
        "nested_keys": sorted(nested_keys),
        "score_like_fields": infer_field_candidates(keys_sorted, "score"),
        "question_like_fields": infer_field_candidates(keys_sorted, "question"),
        "answer_like_fields": infer_field_candidates(keys_sorted, "answer"),
        "metric_like_fields": infer_field_candidates(keys_sorted, "metric"),
        "subject_like_fields": infer_field_candidates(keys_sorted, "subject"),
        "scenario_like_fields": infer_field_candidates(keys_sorted, "scenario"),
        "language_like_fields": infer_field_candidates(keys_sorted, "language"),
        "model_like_fields": infer_field_candidates(keys_sorted, "model"),
        "human_score_like_fields": infer_field_candidates(keys_sorted + sorted(nested_keys), "human_score"),
        "judge_score_like_fields": infer_field_candidates(keys_sorted + sorted(nested_keys), "judge_score"),
        "profile_error": "",
        "likely_role": likely_role_for(path),
    }
    samples = first_samples + reservoir
    return row, samples


def write_schema_report(rows: list[dict[str, Any]]) -> None:
    role_rows = []
    for role, group in _group_by(rows, "likely_role").items():
        role_rows.append(
            {
                "likely_role": role,
                "num_files": len(group),
                "total_records": sum(int(r.get("num_records") or 0) for r in group),
                "example_files": [r["file_path"] for r in group[:5]],
            }
        )

    candidate_notes = []
    for row in rows:
        role = row["likely_role"]
        note = "profiled only"
        if role == "merged_human_metric" and row["file_path"].endswith("results_merge.jsonl"):
            note = "primary main-dataset candidate: has question/answer/metric/task/model and nested human scores"
        elif role == "human_annotation":
            note = "real human annotation candidate; useful for corroboration but may use a different 10-point scale"
        elif role == "merged_human_metric":
            note = "merged human annotation candidate; profile before using because row multiplicity may differ"
        elif role == "llm_judge_output":
            note = "automatic judge output; not a human-label source"
        elif role == "synthetic_or_augmented":
            note = "synthetic/augmented inventory only; excluded from main human-scored dataset"
        elif role == "official_raw":
            note = "official/raw reference or downloaded model data"
        candidate_notes.append({"file_path": row["file_path"], "likely_role": role, "decision_note": note})

    lines = [
        "# Schema Profile",
        "",
        "Each JSON/JSONL file was inspected using the first five records and a deterministic random sample of up to twenty records.",
        "",
        "## Role-Level Summary",
        "",
        md_table(role_rows, ["likely_role", "num_files", "total_records", "example_files"], max_rows=30),
        "",
        "## Candidate Role Decisions",
        "",
        md_table(candidate_notes, ["file_path", "likely_role", "decision_note"], max_rows=160),
        "",
        "## Profile Rows",
        "",
        md_table(
            rows,
            [
                "file_path",
                "num_records",
                "likely_role",
                "score_like_fields",
                "question_like_fields",
                "answer_like_fields",
                "metric_like_fields",
                "scenario_like_fields",
                "human_score_like_fields",
            ],
            max_rows=160,
        ),
    ]
    write_text(OUTPUT_DIR / "schema_profile.md", "\n".join(lines))


def _group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, ""))].append(row)
    return dict(groups)


def main() -> None:
    ensure_exp_dirs()
    rows: list[dict[str, Any]] = []
    sample_rows: dict[str, list[dict[str, Any]]] = {}
    for path in _json_source_paths():
        row, samples = profile_path(path)
        rows.append(row)
        sample_rows[row["file_path"]] = samples

    fieldnames = [
        "file_path",
        "num_records",
        "all_keys",
        "key_presence_rate",
        "nested_keys",
        "score_like_fields",
        "question_like_fields",
        "answer_like_fields",
        "metric_like_fields",
        "subject_like_fields",
        "scenario_like_fields",
        "language_like_fields",
        "model_like_fields",
        "human_score_like_fields",
        "judge_score_like_fields",
        "likely_role",
        "profile_error",
    ]
    write_csv(TABLES_DIR / "schema_profile.csv", rows, fieldnames)
    write_json(OUTPUT_DIR / "samples" / "sample_rows.json", sample_rows)

    main_candidates = [
        "results_merge.jsonl",
        "human_1.jsonl",
        "human_2.jsonl",
        "human_3.jsonl",
        "merge_human_metric.jsonl",
        "merge_human_metric_strict_en.jsonl",
        "merge_human_metric_strict_zh.jsonl",
    ]
    for name in main_candidates:
        path = REPO_ROOT / name
        if path.exists() and path.suffix.lower() in {".json", ".jsonl"}:
            _, samples = profile_path(path)
            write_json(SAMPLES_DIR / f"{path.stem}_sample_records.json", samples[:3])

    write_schema_report(rows)
    print(f"Wrote schema profile for {len(rows)} JSON/JSONL files")


if __name__ == "__main__":
    main()
