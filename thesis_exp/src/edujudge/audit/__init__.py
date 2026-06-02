"""Exp 1 evaluator-vs-human audit helpers.

The audit pipeline is intentionally data-only: it reads the locked Exp0.1
dataset/split and existing judge predictions, then writes all artifacts under
``thesis_exp/outputs/exp01_audit``.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from thesis_exp.src.edujudge.data.normalize_fields import convert_score_to_five
from thesis_exp.src.edujudge.utils.io import (
    REPO_ROOT,
    THESIS_DIR,
    count_json_records,
    flatten_keys,
    iter_json_records,
    read_jsonl,
    relpath,
    write_csv,
    write_json,
    write_jsonl,
    write_text,
)
from thesis_exp.src.edujudge.utils.text_norm import normalize_text, stringify


DATASET_NAME = "edubench_audit_human_scored_subset"
EXP00_OUTPUT_DIR = THESIS_DIR / "outputs" / "exp00_data"
EXP01_OUTPUT_DIR = THESIS_DIR / "outputs" / "exp01_audit"
EXP01_TABLES_DIR = EXP01_OUTPUT_DIR / "tables"
EXP01_FIGURES_DIR = EXP01_OUTPUT_DIR / "figures"
EXP01_SAMPLES_DIR = EXP01_OUTPUT_DIR / "samples"
PROCESSED_DATASET_PATH = THESIS_DIR / "data" / "processed" / "edubench_scoring_all.jsonl"
TEST_SPLIT_PATH = THESIS_DIR / "data" / "splits" / "paper_like_triple_seed42" / "test.jsonl"
EXPECTED_TEST_ROWS = 2218


@dataclass(frozen=True)
class EvaluatorSpec:
    name: str
    field_suffix: str
    aliases: tuple[str, ...]


EVALUATORS: tuple[EvaluatorSpec, ...] = (
    EvaluatorSpec(
        "EduBenchEvaluator",
        "EduBenchEvaluator",
        (
            "EduBenchEvaluator",
            "edubench_evaluator",
            "edubenchevaluator",
            "qwen3_reranker",
            "qwen3-reranker",
            "qwen3_0.6b",
            "qwen3-0.6b",
            "fine_tuned_judge",
            "evaluator_0.6b",
        ),
    ),
    EvaluatorSpec("GPT-4o", "GPT4o", ("GPT-4o", "gpt4o", "gpt-4o", "4o")),
    EvaluatorSpec("DeepSeek-R1", "DeepSeekR1", ("DeepSeek-R1", "deepseek-r1", "r1")),
    EvaluatorSpec("DeepSeek-V3", "DeepSeekV3", ("DeepSeek-V3", "deepseek-v3", "v3")),
    EvaluatorSpec("QwQ-plus", "QwQPlus", ("QwQ-plus", "qwq-plus", "qwq", "qvq")),
)

EVALUATOR_BY_NAME = {spec.name: spec for spec in EVALUATORS}
EVALUATOR_BY_SUFFIX = {spec.field_suffix: spec for spec in EVALUATORS}


def ensure_exp01_dirs() -> None:
    for path in [EXP01_OUTPUT_DIR, EXP01_TABLES_DIR, EXP01_FIGURES_DIR, EXP01_SAMPLES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def compact_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", stringify(value).lower())


def canonical_evaluator(value: object) -> str | None:
    token = compact_token(value)
    if not token:
        return None
    for spec in EVALUATORS:
        for alias in spec.aliases:
            alias_token = compact_token(alias)
            if token == alias_token:
                return spec.name
    return None


def evaluator_from_path(path: Path | str) -> str | None:
    path_text = str(path).lower()
    candidates = [
        ("EduBenchEvaluator", ["edubenchevaluator", "edubench_evaluator", "qwen3", "reranker"]),
        ("GPT-4o", ["gpt-4o", "gpt4o", "4o"]),
        ("DeepSeek-R1", ["deepseek-r1", "deepseek_r1", "r1_eval", "r1-"]),
        ("DeepSeek-V3", ["deepseek-v3", "deepseek_v3", "v3_eval", "v3-"]),
        ("QwQ-plus", ["qwq-plus", "qwq_eval", "qwq"]),
    ]
    for evaluator, aliases in candidates:
        if any(alias in path_text for alias in aliases):
            return evaluator
    return None


def is_synthetic_or_sampled_path(path: Path | str) -> bool:
    text = str(path).lower()
    blocked = ["sampled_merge_50_new", "synthetic", "augmentation", "augmented"]
    if any(token in text for token in blocked):
        return True
    name = Path(path).name.lower()
    return "_sampled" in name or name.endswith("_sampled.jsonl") or "sampled." in name


def collect_candidate_paths() -> list[Path]:
    """Collect likely judge-source files without reading generated Exp1 outputs."""
    explicit_names = {
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
    interesting = [
        "judge",
        "eval",
        "evaluator",
        "edubench",
        "reranker",
        "qwen",
        "metric",
        "score",
        "result",
        "merge",
        "merged",
        "gpt-4o",
        "deepseek",
        "qwq",
    ]
    paths: set[Path] = {PROCESSED_DATASET_PATH, TEST_SPLIT_PATH}
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".csv"}:
            continue
        parts = set(path.parts)
        if ".git" in parts or "__pycache__" in parts:
            continue
        if EXP01_OUTPUT_DIR in path.parents:
            continue
        name = path.name.lower()
        text = str(path.relative_to(REPO_ROOT)).lower()
        if name in explicit_names or any(token in text for token in interesting):
            paths.add(path)
    return sorted(paths, key=lambda p: relpath(p))


def read_csv_records(path: Path) -> tuple[list[dict[str, Any]], int, list[str]]:
    rows: list[dict[str, Any]] = []
    count = 0
    fieldnames: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            count += 1
            if len(rows) < 3:
                rows.append(dict(row))
    return rows, count, fieldnames


def sample_records(path: Path, limit: int = 3) -> tuple[list[dict[str, Any]], int, list[str]]:
    if path.suffix.lower() == ".csv":
        return read_csv_records(path)
    samples: list[dict[str, Any]] = []
    for _, record in iter_json_records(path):
        if isinstance(record, dict):
            samples.append(record)
        else:
            samples.append({"__value__": record})
        if len(samples) >= limit:
            break
    return samples, count_json_records(path), []


def infer_field_candidates(keys: Iterable[str], category: str) -> list[str]:
    patterns = {
        "score": ["score", "rating", "grade", "评分", "分数", "judge", "evaluate"],
        "question": ["question", "prompt", "instruction", "query", "题目", "问题"],
        "answer": ["answer", "response", "completion", "output", "回答", "答案"],
        "metric": ["metric", "principle", "criterion", "criteria", "rubric", "维度", "指标"],
        "record_id": ["record_id", "row_id", "id", "answer_id", "question_id"],
        "triple_key": ["triple_key", "triple", "question_key", "answer_key"],
        "language": ["language", "lang", "语言"],
    }
    wanted = patterns[category]
    out = []
    for key in keys:
        key_norm = normalize_text(key)
        if any(token in key_norm for token in wanted):
            out.append(key)
    return sorted(set(out))


def deep_get(record: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in record and record.get(name) not in (None, ""):
            return record.get(name)
    return None


def get_alignment_text_fields(record: dict[str, Any], metric_override: object = None) -> tuple[str, str, str]:
    question = deep_get(record, ["question", "prompt", "instruction", "query", "input"])
    answer = deep_get(record, ["answer", "response", "completion", "output"])
    metric = metric_override if metric_override not in (None, "") else deep_get(
        record,
        ["metric_canonical", "metric_raw", "metric", "principle", "criterion", "criteria"],
    )
    return stringify(question), stringify(answer), stringify(metric)


def normalized_qam_key(record: dict[str, Any], metric_override: object = None, canonical_metric: bool = True) -> str:
    from thesis_exp.src.edujudge.data.normalize_fields import canonicalize_metric

    question, answer, metric = get_alignment_text_fields(record, metric_override)
    metric_value = metric
    if canonical_metric:
        metric_value = canonicalize_metric(metric).get("canonical_metric", metric)
    return "\u241f".join([normalize_text(question), normalize_text(answer), normalize_text(metric_value)])


def raw_qam_key(record: dict[str, Any], metric_override: object = None) -> str:
    question, answer, metric = get_alignment_text_fields(record, metric_override)
    return "\u241f".join([normalize_text(question), normalize_text(answer), normalize_text(metric)])


def round_label(value: float | int | None) -> int | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    rounded = int(float(value) + 0.5)
    return max(1, min(5, rounded))


@dataclass(frozen=True)
class ParsedScore:
    value: float | None
    valid: bool
    note: str


def _numeric_from_score_like(value: Any) -> tuple[float | None, int | None, str]:
    if value is None:
        return None, None, "empty"
    if isinstance(value, bool):
        return None, None, "boolean is not a score"
    if isinstance(value, (int, float)):
        if math.isnan(float(value)):
            return None, None, "nan"
        return float(value), None, "numeric"
    if isinstance(value, dict):
        for key in ["score", "final_score", "rating", "value", "分数", "评分"]:
            if key in value:
                return _numeric_from_score_like(value[key])
        numeric_values = [v for v in value.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if len(numeric_values) == 1:
            return float(numeric_values[0]), None, "single numeric value in dict"
        return None, None, "dict has no unambiguous score"
    if isinstance(value, list):
        if len(value) == 1:
            return _numeric_from_score_like(value[0])
        return None, None, "list has no unambiguous score"

    text = stringify(value).strip()
    if not text:
        return None, None, "empty string"
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        loaded = None
    if loaded is not None and not isinstance(loaded, str):
        return _numeric_from_score_like(loaded)

    ratio = re.search(r"(-?\d+(?:\.\d+)?)\s*/\s*(5|10)\b", text)
    if ratio:
        return float(ratio.group(1)), int(ratio.group(2)), "ratio string"
    keyword = re.search(r"(?:score|rating|grade|评分|分数)\D{0,12}(-?\d+(?:\.\d+)?)", text, flags=re.I)
    if keyword:
        return float(keyword.group(1)), None, "keyword score string"
    numbers = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])", text)
    if len(numbers) == 1:
        return float(numbers[0]), None, "single number string"
    return None, None, "no unambiguous number"


def parse_score_to_1_5(value: Any, source_scale: str = "auto") -> ParsedScore:
    raw_value, denominator, note = _numeric_from_score_like(value)
    if raw_value is None:
        return ParsedScore(None, False, note)
    scale = source_scale
    if denominator == 5:
        scale = "1-5"
    elif denominator == 10:
        scale = "1-10"
    if scale == "1-10" or (scale == "auto" and raw_value > 5 and raw_value <= 10):
        mapped = convert_score_to_five(raw_value)
        if mapped is None:
            return ParsedScore(None, False, f"{note}; value outside 1-10")
        return ParsedScore(float(mapped), True, f"{note}; mapped from 1-10")
    if 1 <= raw_value <= 5:
        return ParsedScore(float(raw_value), True, f"{note}; kept as 1-5")
    return ParsedScore(None, False, f"{note}; value outside 1-5")


def truncate_for_json(value: Any, max_len: int = 600) -> Any:
    if isinstance(value, str):
        return value if len(value) <= max_len else value[: max_len - 3] + "..."
    if isinstance(value, dict):
        return {str(k): truncate_for_json(v, max_len=max_len) for k, v in value.items()}
    if isinstance(value, list):
        return [truncate_for_json(v, max_len=max_len) for v in value[:20]]
    return value


def markdown_table(rows: list[dict[str, Any]], columns: list[str], max_rows: int | None = None) -> str:
    shown = rows if max_rows is None else rows[:max_rows]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in shown:
        cells = []
        for col in columns:
            text = stringify(row.get(col, ""))
            text = text.replace("\n", " ").replace("|", "\\|")
            if len(text) > 96:
                text = text[:93] + "..."
            cells.append(text)
        lines.append("| " + " | ".join(cells) + " |")
    if max_rows is not None and len(rows) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(rows)} rows._")
    return "\n".join(lines)

