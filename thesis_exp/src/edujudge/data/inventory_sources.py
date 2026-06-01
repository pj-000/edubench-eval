"""Inventory candidate EduBench source files."""

from __future__ import annotations

import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.utils.io import (
    OUTPUT_DIR,
    REPO_ROOT,
    TABLES_DIR,
    candidate_paths,
    count_json_records,
    ensure_exp_dirs,
    iter_json_records,
    md_table,
    relpath,
    write_csv,
    write_text,
)


IMPORTANT_NAMES = {
    "EduBench.zip",
    "results_merge.jsonl",
    "merge_human_metric.jsonl",
    "merge_human_metric_strict_en.jsonl",
    "merge_human_metric_strict_zh.jsonl",
    "human_1.jsonl",
    "human_2.jsonl",
    "human_3.jsonl",
    "metrics_map.json",
    "merge_model_metric.jsonl",
    "groupby_metric_qwq_eval_en.jsonl",
    "groupby_metric_qwq_eval_zh.jsonl",
    "groupby_metric_r1_eval_en.jsonl",
    "groupby_metric_r1_eval_zh.jsonl",
    "groupby_metric_v3_eval_en.jsonl",
    "groupby_metric_v3_eval_zh.jsonl",
    "sampled_merge_50_new.json",
    "sampled_merge_50_new_swift.json",
    "5-grades.py",
}


def likely_role_for(path: Path) -> str:
    name = path.name.lower()
    rel = relpath(path).lower()
    if name in {"sampled_merge_50_new.json", "sampled_merge_50_new_swift.json"} or "sampled" in name:
        return "synthetic_or_augmented"
    if name == "edubench.zip" or name == "edubench.pdf":
        return "official_raw"
    if name in {"metrics_map.json", "5-grades.py"} or name.endswith(".py"):
        return "script_or_config"
    if name in {"human_1.jsonl", "human_2.jsonl", "human_3.jsonl"} or name.startswith("5_human_"):
        return "human_annotation"
    if "merge_human_metric" in name or name == "results_merge.jsonl":
        return "merged_human_metric"
    if "groupby_metric" in name and "_eval_" in name:
        return "llm_judge_output"
    if "merge_model_metric" in name or "judge" in rel or "evaluator" in rel:
        return "llm_judge_output"
    if "download_raw" in rel:
        return "official_raw"
    return "unknown"


def _json_keys(path: Path) -> tuple[list[str], list[str], str]:
    sample_keys: set[str] = set()
    top_level_keys: set[str] = set()
    sample_preview = ""
    try:
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                top_level_keys.update(map(str, data.keys()))
            elif isinstance(data, list) and data:
                top_level_keys.add("[]")
        for _, record in zip(range(5), iter_json_records(path)):
            _, obj = record
            if isinstance(obj, dict):
                sample_keys.update(map(str, obj.keys()))
                if not sample_preview:
                    sample_preview = json.dumps(obj, ensure_ascii=False)[:500]
            elif not sample_preview:
                sample_preview = json.dumps(obj, ensure_ascii=False)[:500]
    except Exception as exc:  # noqa: BLE001 - inventory should keep going
        sample_preview = f"ERROR: {type(exc).__name__}: {exc}"
    return sorted(top_level_keys), sorted(sample_keys), sample_preview


def _zip_keys(path: Path) -> tuple[list[str], list[str], str]:
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
        suffixes = sorted({Path(name).suffix.lower() or "<no_ext>" for name in names})
        return names[:20], suffixes, "; ".join(names[:20])
    except Exception as exc:  # noqa: BLE001
        return [], [], f"ERROR: {type(exc).__name__}: {exc}"


def inspect_path(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower().lstrip(".") or "no_ext"
    exists = path.exists()
    top_level_keys: list[str] = []
    sample_keys: list[str] = []
    preview = ""
    num_records: int | str = ""
    if exists:
        if suffix in {"json", "jsonl", "zip"}:
            num_records = count_json_records(path)
        if suffix in {"json", "jsonl"}:
            top_level_keys, sample_keys, preview = _json_keys(path)
        elif suffix == "zip":
            top_level_keys, sample_keys, preview = _zip_keys(path)
        else:
            num_records = ""
    return {
        "file_path": relpath(path),
        "file_type": suffix,
        "exists": exists,
        "num_rows_or_records": num_records,
        "top_level_keys": top_level_keys,
        "sample_keys": sample_keys,
        "likely_role": likely_role_for(path),
        "size_bytes": path.stat().st_size if exists else "",
        "sample_preview": preview,
    }


def build_inventory() -> list[dict[str, Any]]:
    paths = candidate_paths()
    by_name = {path.name: path for path in paths}
    for name in IMPORTANT_NAMES:
        path = REPO_ROOT / name
        if name not in by_name and path.exists():
            paths.append(path)
    rows = [inspect_path(path) for path in sorted(set(paths), key=lambda p: relpath(p))]
    return rows


def write_inventory_report(rows: list[dict[str, Any]]) -> None:
    role_counts = Counter(row["likely_role"] for row in rows)
    important = [row for row in rows if Path(row["file_path"]).name in IMPORTANT_NAMES]
    lines = [
        "# Source Inventory",
        "",
        "This inventory scans source-like files in the repository while excluding generated `thesis_exp` outputs.",
        "",
        "## Role Counts",
        "",
        md_table(
            [{"likely_role": role, "num_files": count} for role, count in role_counts.most_common()],
            ["likely_role", "num_files"],
            max_rows=30,
        ),
        "",
        "## Required Candidate Files",
        "",
        md_table(
            important,
            ["file_path", "file_type", "exists", "num_rows_or_records", "likely_role", "sample_keys"],
            max_rows=80,
        ),
        "",
        "## All Inventoried Files",
        "",
        md_table(
            rows,
            ["file_path", "file_type", "exists", "num_rows_or_records", "likely_role", "sample_keys"],
            max_rows=120,
        ),
    ]
    write_text(OUTPUT_DIR / "source_inventory.md", "\n".join(lines))


def main() -> None:
    ensure_exp_dirs()
    rows = build_inventory()
    fieldnames = [
        "file_path",
        "file_type",
        "exists",
        "num_rows_or_records",
        "top_level_keys",
        "sample_keys",
        "likely_role",
        "size_bytes",
        "sample_preview",
    ]
    write_csv(TABLES_DIR / "source_inventory.csv", rows, fieldnames)
    write_inventory_report(rows)
    print(f"Wrote {len(rows)} inventory rows to {TABLES_DIR / 'source_inventory.csv'}")


if __name__ == "__main__":
    main()
