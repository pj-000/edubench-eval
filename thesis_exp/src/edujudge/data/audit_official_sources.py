"""Audit local and official EduBench source availability for Exp 0.1."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.data.build_dataset import PRIMARY_SOURCE, PROCESSED_PATH
from thesis_exp.src.edujudge.utils.hashing import sha1_text
from thesis_exp.src.edujudge.utils.io import CACHE_DIR, OUTPUT_DIR, REPO_ROOT, TABLES_DIR, ensure_exp_dirs, iter_json_records, md_table, read_jsonl, relpath, write_csv, write_text
from thesis_exp.src.edujudge.utils.text_norm import normalize_text, stringify


OFFICIAL_REPO_URL = "https://github.com/ybai-nlp/EduBench.git"
OFFICIAL_CACHE = CACHE_DIR / "official_edubench"


def _clone_official_repo() -> Path | None:
    if OFFICIAL_CACHE.exists() and (OFFICIAL_CACHE / ".git").exists():
        return OFFICIAL_CACHE
    if OFFICIAL_CACHE.exists():
        shutil.rmtree(OFFICIAL_CACHE)
    OFFICIAL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", OFFICIAL_REPO_URL, str(OFFICIAL_CACHE)],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
    except Exception:
        return None
    return OFFICIAL_CACHE


def _role_for(path: Path) -> str:
    rel = str(path).lower()
    if "en_data" in rel:
        return "en_data"
    if "zh_data" in rel:
        return "zh_data"
    if "sampled_data" in rel:
        return "sampled_data"
    if "model_eval_score" in rel:
        return "model_eval_score"
    if path.name.lower().startswith("readme"):
        return "readme"
    return "unknown"


def _sample_records(path: Path) -> tuple[int, set[str], list[dict[str, Any]]]:
    if path.suffix.lower() not in {".json", ".jsonl"}:
        return 0, set(), []
    count = 0
    keys: set[str] = set()
    samples: list[dict[str, Any]] = []
    try:
        for _, obj in iter_json_records(path):
            count += 1
            if isinstance(obj, dict):
                keys.update(map(str, obj.keys()))
                if len(samples) < 50:
                    samples.append(obj)
    except Exception:
        return count, keys, samples
    return count, keys, samples


def _contains(keys: set[str], *tokens: str) -> bool:
    lowered = {key.lower() for key in keys}
    return any(any(token in key for token in tokens) for key in lowered)


def _inspect_file(path: Path, origin: str, display_root: Path | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    count, keys, samples = _sample_records(path)
    source_path = relpath(path) if display_root is None else str(path.relative_to(display_root))
    row = {
        "source_path": source_path,
        "source_origin": origin,
        "file_role": _role_for(path),
        "num_records": count,
        "top_level_keys": sorted(keys),
        "contains_question": _contains(keys, "question", "prompt", "instruction", "query", "题目", "问题"),
        "contains_answer": _contains(keys, "answer", "response", "completion", "output", "回答", "答案"),
        "contains_metric": _contains(keys, "metric", "principle", "criterion", "criteria", "维度", "指标"),
        "contains_task_or_scenario": _contains(keys, "task", "scenario", "scene", "场景", "任务"),
        "contains_subject": _contains(keys, "subject", "discipline", "course", "学科", "专业"),
        "contains_education_level": _contains(keys, "education", "grade", "level", "年级", "学段"),
        "contains_human_score": _contains(keys, "human", "annotator", "人工"),
        "contains_model_eval_score": _contains(keys, "eval", "judge", "model", "score", "评分"),
    }
    return row, samples


def _official_roots(local_arg: str | None) -> list[tuple[Path, str]]:
    roots: list[tuple[Path, str]] = []
    if local_arg:
        path = Path(local_arg).expanduser().resolve()
        if path.exists():
            roots.append((path, "local_repo"))
    for candidate in [
        REPO_ROOT / "data" / "all_data",
        REPO_ROOT / "EduBench",
        REPO_ROOT / "edu-data-synthesis-main" / "data" / "all_data",
    ]:
        if candidate.exists():
            roots.append((candidate, "local_repo"))
    cloned = _clone_official_repo()
    if cloned is not None:
        roots.append((cloned, "official_github_clone"))
    return roots


def _zip_rows() -> list[dict[str, Any]]:
    path = REPO_ROOT / "EduBench.zip"
    if not path.exists():
        return []
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
    except zipfile.BadZipFile:
        names = []
    return [
        {
            "source_path": relpath(path),
            "source_origin": "zip",
            "file_role": "unknown",
            "num_records": len(names),
            "top_level_keys": names[:20],
            "contains_question": False,
            "contains_answer": False,
            "contains_metric": False,
            "contains_task_or_scenario": False,
            "contains_subject": False,
            "contains_education_level": False,
            "contains_human_score": False,
            "contains_model_eval_score": False,
        }
    ]


def _record_keys(records: list[dict[str, Any]]) -> set[str]:
    keys = set()
    for row in records:
        question = row.get("question") or row.get("prompt") or row.get("instruction") or row.get("query")
        answer = row.get("answer") or row.get("response") or row.get("completion") or row.get("output")
        metric = row.get("metric") or row.get("principle") or row.get("criterion")
        if question and answer and metric:
            keys.add(sha1_text(normalize_text(question), normalize_text(answer), normalize_text(metric)))
    return keys


def audit(official_repo_path: str | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ensure_exp_dirs()
    rows: list[dict[str, Any]] = []
    official_samples: list[dict[str, Any]] = []
    rows.extend(_zip_rows())
    roots = _official_roots(official_repo_path)
    for root, origin in roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".json", ".jsonl", ".md"}:
                continue
            if "data/all_data" not in str(path).replace("\\", "/") and path.name.lower() != "readme.md":
                continue
            row, samples = _inspect_file(path, origin, root)
            rows.append(row)
            official_samples.extend(samples[:20])

    local_status = "local_derived_from_edubench"
    official_names = {Path(row["source_path"]).name for row in rows}
    if PRIMARY_SOURCE.name not in official_names:
        local_status = "local_derived_from_edubench_not_official_filename"

    processed = read_jsonl(PROCESSED_PATH) if PROCESSED_PATH.exists() else []
    official_keys = _record_keys(official_samples)
    processed_keys = _record_keys(processed)
    exact_matches = len(official_keys & processed_keys) if official_keys and processed_keys else 0
    fuzzy_matches = 0
    if official_samples and processed:
        official_question_keys = {sha1_text(normalize_text(row.get("question") or row.get("prompt") or row.get("instruction") or row.get("query"))) for row in official_samples}
        processed_question_keys = {row.get("question_key") for row in processed}
        fuzzy_matches = len(official_question_keys & processed_question_keys)

    summary = {
        "primary_local_source": relpath(PRIMARY_SOURCE),
        "results_merge_status": local_status,
        "official_roots": [str(root) for root, _ in roots],
        "official_inventory_rows": len(rows),
        "official_exact_triple_matches_sampled": exact_matches,
        "official_question_fuzzy_matches_sampled": fuzzy_matches,
    }
    return rows, summary


def write_outputs(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    write_csv(
        TABLES_DIR / "official_source_inventory.csv",
        rows,
        [
            "source_path",
            "source_origin",
            "file_role",
            "num_records",
            "top_level_keys",
            "contains_question",
            "contains_answer",
            "contains_metric",
            "contains_task_or_scenario",
            "contains_subject",
            "contains_education_level",
            "contains_human_score",
            "contains_model_eval_score",
        ],
    )
    lines = [
        "# Official Source Audit",
        "",
        "This audit checks local files, `EduBench.zip`, and a read-only clone of the official EduBench GitHub repository when network access is available.",
        "",
        "## Summary",
        "",
        md_table([{"field": key, "value": value} for key, value in summary.items()], ["field", "value"], max_rows=30),
        "",
        "## Inventory",
        "",
        md_table(rows, ["source_path", "source_origin", "file_role", "num_records", "contains_question", "contains_answer", "contains_metric", "contains_task_or_scenario", "contains_human_score", "contains_model_eval_score"], max_rows=120),
        "",
        "## Local Source Status",
        "",
        f"`results_merge.jsonl` is treated as `{summary['results_merge_status']}`. It is not labeled as official full EduBench raw data unless an official repository file with the same role/name is found.",
    ]
    write_text(OUTPUT_DIR / "official_source_audit.md", "\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official_repo_path", default=None)
    args = parser.parse_args(argv)
    rows, summary = audit(args.official_repo_path)
    write_outputs(rows, summary)
    print(f"Wrote official source audit with {len(rows)} rows")


if __name__ == "__main__":
    main()
