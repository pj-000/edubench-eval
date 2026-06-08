"""Shared helpers for Exp6 synthetic data inventory.

The Exp6 audit is intentionally read-only with respect to Exp0-Exp5 data and
model outputs. Helpers here inspect existing local files and write only under
``thesis_exp/outputs/exp06_synthetic_low_score``.
"""

from __future__ import annotations

import ast
import csv
import json
import math
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

from thesis_exp.src.edujudge.data.normalize_fields import canonicalize_metric, infer_language
from thesis_exp.src.edujudge.exp06 import requested_source_paths
from thesis_exp.src.edujudge.utils.hashing import sha1_text
from thesis_exp.src.edujudge.utils.io import flatten_keys, iter_json_records, relpath, write_csv
from thesis_exp.src.edujudge.utils.text_norm import normalize_text, stringify, truncate_text


PROFILE_SUFFIXES = {
    ".json",
    ".jsonl",
    ".xlsx",
    ".xls",
    ".csv",
    ".py",
    ".md",
    ".yaml",
    ".yml",
    ".zip",
    ".pkl",
}

QUESTION_PATTERNS = ["question", "query", "prompt", "instruction", "message", "题目", "问题", "对话"]
ANSWER_PATTERNS = ["answer", "response", "output", "completion", "solution", "studentanswer", "回答", "答案"]
METRIC_PATTERNS = ["metric", "principle", "criterion", "criteria", "eval", "评分指标", "评估指标", "维度"]
LABEL_PATTERNS = ["label", "target", "human_mean", "human_1", "human_2", "human_3", "标签"]
SCORE_PATTERNS = ["score", "scores", "分数", "评分"]
RUBRIC_PATTERNS = ["rubric", "criteria", "gradingcriteria", "grading_criteria", "评分细则", "评分标准"]
REASON_PATTERNS = ["reason", "rationale", "analysis", "scoringdetails", "details", "原因", "说明", "解析"]
ERROR_TYPE_PATTERNS = ["error_type", "error", "mistake", "错误类型", "错误"]
SOURCE_ID_PATTERNS = ["id", "question_id", "question_key", "source", "source_question", "source_row"]
LANGUAGE_PATTERNS = ["language", "lang", "语言"]
GENERATION_PATTERNS = ["generation", "generated", "generator", "gen", "model", "method", "合成", "生成"]

OFFICIAL_METRIC_SET = {
    "Instruction Following & Task Completion",
    "Role & Tone Consistency",
    "Content Relevance & Scope Control",
    "Scenario Element Integration",
    "Basic Factual Accuracy",
    "Domain Knowledge Accuracy",
    "Reasoning Process Rigor",
    "Error Identification & Correction Precision",
    "Clarity, Simplicity & Inspiration",
    "Motivation, Guidance & Positive Feedback",
    "Personalization, Adaptation & Learning Support",
    "Higher-Order Thinking & Skill Development",
}


def source_rel(path: Path | str) -> str:
    return relpath(Path(path))


def iter_candidate_paths() -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for root in requested_source_paths():
        if root not in seen:
            paths.append(root)
            seen.add(root)
        if not root.exists() or not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if "__pycache__" in path.parts or ".git" in path.parts:
                continue
            if path.is_file() and path.suffix.lower() in PROFILE_SUFFIXES and path not in seen:
                paths.append(path)
                seen.add(path)
    return paths


def file_type(path: Path) -> str:
    if not path.exists():
        return "missing"
    if path.is_dir():
        return "dir"
    suffix = path.suffix.lower().lstrip(".")
    return suffix or "no_ext"


def likely_role_for_path(path: Path) -> str:
    name = path.name.lower()
    rel = source_rel(path).lower()
    suffix = path.suffix.lower()
    if not path.exists():
        return "unknown"
    if path.is_dir():
        if name in {"deepseek_output", "qwen_output"}:
            return "model_judge_output"
        if "synthesis" in name:
            return "generation_script"
        return "unknown"
    if name in {"sampled_merge_50_new.json", "sampled_merge_50_new_swift.json"}:
        return "sampled_augmented"
    if name == "human_sampled_eval_sft_criteria_test.json":
        return "sampled_augmented"
    if name.startswith("groupby_metric_") and "_eval_" in name:
        return "model_judge_output"
    if name in {"merge_model_metric.jsonl", "deepseek-r1_merged.jsonl"}:
        return "model_judge_output"
    if suffix in {".py", ".yaml", ".yml", ".md", ".pkl", ".zip"}:
        if "data/eval_data" not in rel:
            return "generation_script" if "synthesis" in rel or suffix in {".py", ".yaml", ".yml", ".pkl"} else "unknown"
    if "judge" in rel or "_eval_" in name:
        return "model_judge_output"
    if "processed_excel_data_2" in name:
        return "model_judge_output"
    if "data/eval_data/" in rel:
        return "synthetic_candidate"
    if "deepseek_generated" in name or "generated" in name:
        return "synthetic_candidate"
    if "data/zh/" in rel and "edu-data-synthesis-main" in rel:
        return "sampled_augmented"
    if "synthesis" in rel and suffix in {".json", ".jsonl"}:
        return "synthetic_candidate"
    if suffix in {".xlsx", ".xls"} and "judge" in rel:
        return "model_judge_output"
    return "unknown"


def risk_and_usable(path: Path, role: str, contains: dict[str, bool]) -> tuple[str, str, str]:
    name = path.name.lower()
    rel = source_rel(path)
    if not path.exists():
        return "BLOCKED", "NO_MISSING", "missing requested source"
    if name in {"sampled_merge_50_new.json", "sampled_merge_50_new_swift.json"}:
        return (
            "HIGH",
            "REVIEW_ONLY_HIGH_RISK",
            "required default HIGH risk; SFT/sampled data needs leakage and label provenance confirmation",
        )
    if name == "human_sampled_eval_sft_criteria_test.json":
        return (
            "HIGH",
            "NO_TEST_STYLE_SAMPLE",
            "human-sampled/test-style SFT file; do not use as dev/test or direct train label without provenance review",
        )
    if role == "model_judge_output":
        return "BLOCKED", "NO_JUDGE_OUTPUT_ONLY", "model/judge output is not a human-label source"
    if role == "generation_script":
        return "BLOCKED", "NO_SCRIPT_OR_ARCHIVE", "generation/config/archive artifact, not a normalized training source"
    if role == "synthetic_candidate":
        if contains.get("question") and contains.get("answer") and (contains.get("metric") or contains.get("score")):
            return "MEDIUM", "POSSIBLE_FILTERED_TRAIN_ONLY", "synthetic candidate pending leakage and label reliability checks"
        return "HIGH", "REVIEW_ONLY_INCOMPLETE_SCHEMA", "synthetic-like source lacks complete question/answer/metric/score schema"
    if role == "sampled_augmented":
        return "HIGH", "REVIEW_ONLY", "sampled/augmented source; train-only use requires strict leakage and label provenance review"
    return "HIGH", "REVIEW_ONLY_UNKNOWN", f"unknown role for {rel}"


def _fold_key(value: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", value.lower())


def fields_like(keys: Iterable[str], patterns: list[str]) -> list[str]:
    folded_patterns = [_fold_key(pattern) for pattern in patterns]
    out = []
    for key in keys:
        folded = _fold_key(str(key))
        if any(pattern and pattern in folded for pattern in folded_patterns):
            out.append(str(key))
    return sorted(set(out))


def contains_from_keys(keys: Iterable[str], patterns: list[str]) -> bool:
    return bool(fields_like(keys, patterns))


def record_count(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_dir():
        return sum(1 for item in path.rglob("*") if item.is_file())
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        try:
            return sum(1 for _ in iter_json_records(path))
        except Exception:
            return 0
    if suffix == ".zip":
        try:
            with zipfile.ZipFile(path) as zf:
                return len(zf.infolist())
        except Exception:
            return 0
    if suffix == ".csv":
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                return max(0, sum(1 for _ in handle) - 1)
        except Exception:
            return 0
    if suffix in {".xlsx", ".xls"}:
        try:
            return max(0, sum(1 for _ in iter_xlsx_records(path, limit=None)))
        except Exception:
            return 0
    return 0


def iter_tabular_records(path: Path, limit: int | None = None) -> Iterator[tuple[int, dict[str, Any]]]:
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        for row_index, obj in iter_json_records(path):
            if isinstance(obj, dict):
                yield row_index, obj
            else:
                yield row_index, {"__value__": obj}
            if limit is not None and row_index + 1 >= limit:
                return
        return
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for idx, row in enumerate(reader):
                yield idx, dict(row)
                if limit is not None and idx + 1 >= limit:
                    return
        return
    if suffix in {".xlsx", ".xls"}:
        yield from iter_xlsx_records(path, limit=limit)


def iter_xlsx_records(path: Path, limit: int | None = None) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield rows from the first XLSX sheet without requiring openpyxl.

    Server experiment environments often omit optional spreadsheet packages. The
    Exp6 audit only needs light schema profiling, so a small OOXML reader is
    enough and avoids crashing normalization on `.xlsx` artifacts.
    """
    if path.suffix.lower() != ".xlsx":
        return
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rel_ns = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
    try:
        with zipfile.ZipFile(path) as zf:
            shared = _xlsx_shared_strings(zf, ns)
            sheet_path = _first_sheet_path(zf, ns, rel_ns)
            if not sheet_path:
                return
            rows = []
            root = ET.fromstring(zf.read(sheet_path))
            for row_node in root.findall(".//a:sheetData/a:row", ns):
                values: dict[int, Any] = {}
                for cell in row_node.findall("a:c", ns):
                    ref = cell.attrib.get("r", "")
                    col_idx = _excel_col_index(ref)
                    if col_idx is None:
                        continue
                    values[col_idx] = _xlsx_cell_value(cell, shared, ns)
                if values:
                    max_col = max(values)
                    rows.append([values.get(idx, "") for idx in range(max_col + 1)])
            if not rows:
                return
            header = [stringify(cell).strip() or f"col_{idx}" for idx, cell in enumerate(rows[0])]
            for idx, values in enumerate(rows[1:]):
                yield idx, {header[col]: values[col] if col < len(values) else "" for col in range(len(header))}
                if limit is not None and idx + 1 >= limit:
                    return
    except Exception:
        return


def _xlsx_shared_strings(zf: zipfile.ZipFile, ns: dict[str, str]) -> list[str]:
    try:
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    out = []
    for item in root.findall("a:si", ns):
        parts = [node.text or "" for node in item.findall(".//a:t", ns)]
        out.append("".join(parts))
    return out


def _first_sheet_path(zf: zipfile.ZipFile, ns: dict[str, str], rel_ns: dict[str, str]) -> str:
    try:
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    except KeyError:
        return ""
    first_sheet = workbook.find(".//a:sheets/a:sheet", ns)
    if first_sheet is None:
        return ""
    rel_id = first_sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
    if not rel_id:
        return "xl/worksheets/sheet1.xml"
    for rel in rels.findall("r:Relationship", rel_ns):
        if rel.attrib.get("Id") == rel_id:
            target = rel.attrib.get("Target", "")
            if target.startswith("/"):
                return target.lstrip("/")
            return "xl/" + target.lstrip("/")
    return "xl/worksheets/sheet1.xml"


def _excel_col_index(ref: str) -> int | None:
    match = re.match(r"([A-Z]+)", ref.upper())
    if not match:
        return None
    value = 0
    for ch in match.group(1):
        value = value * 26 + (ord(ch) - ord("A") + 1)
    return value - 1


def _xlsx_cell_value(cell: ET.Element, shared: list[str], ns: dict[str, str]) -> Any:
    cell_type = cell.attrib.get("t", "")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(".//a:t", ns))
    value_node = cell.find("a:v", ns)
    if value_node is None or value_node.text is None:
        return ""
    raw = value_node.text
    if cell_type == "s":
        try:
            return shared[int(raw)]
        except (ValueError, IndexError):
            return raw
    return raw


def sample_records(path: Path, limit: int = 5) -> list[dict[str, Any]]:
    if not path.exists() or path.is_dir() or path.suffix.lower() not in {".json", ".jsonl", ".csv", ".xlsx", ".xls"}:
        return []
    out = []
    try:
        for row_index, row in iter_tabular_records(path, limit=limit):
            out.append({"source_row_index": row_index, **truncate_record(row)})
    except Exception as exc:  # noqa: BLE001 - profiling should keep going
        out.append({"profile_error": f"{type(exc).__name__}: {exc}"})
    return out


def truncate_record(row: dict[str, Any], max_len: int = 300) -> dict[str, Any]:
    out = {}
    for key, value in row.items():
        if isinstance(value, str):
            out[key] = truncate_text(value, max_len)
        elif isinstance(value, (dict, list)):
            out[key] = truncate_text(value, max_len)
        else:
            out[key] = value
    return out


def key_profile(path: Path, sample_limit: int = 200) -> tuple[list[str], list[str], str]:
    keys: Counter[str] = Counter()
    nested: set[str] = set()
    error = ""
    try:
        for _, row in iter_tabular_records(path, limit=sample_limit):
            keys.update(map(str, row.keys()))
            nested.update(flatten_keys(row))
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    return sorted(keys), sorted(nested), error


def parse_jsonish(text: object) -> Any:
    raw = stringify(text).strip()
    if not raw:
        return None
    raw = re.sub(r"^```(?:json)?", "", raw.strip(), flags=re.IGNORECASE).strip()
    raw = re.sub(r"```$", "", raw.strip()).strip()
    if raw.startswith("json"):
        raw = raw[4:].strip()
    for candidate in [raw, _extract_first_json_blob(raw)]:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        try:
            return ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            pass
    return None


def _extract_first_json_blob(text: str) -> str:
    starts = [idx for idx in [text.find("["), text.find("{")] if idx >= 0]
    if not starts:
        return ""
    start = min(starts)
    opener = text[start]
    closer = "]" if opener == "[" else "}"
    depth = 0
    in_string = False
    quote = ""
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if in_string:
            if ch == quote:
                in_string = False
            continue
        if ch in {"'", '"'}:
            in_string = True
            quote = ch
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return ""


def extract_messages_qa(messages: Any) -> tuple[str, str]:
    if not isinstance(messages, list):
        return "", ""
    question = ""
    answer = ""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = normalize_text(msg.get("role"))
        content = stringify(msg.get("content"))
        if role == "user" and not question:
            question = content
        elif role == "assistant" and not answer:
            answer = content
    return question, answer


def extract_dialogue_qa(prompt: object) -> tuple[str, str]:
    text = stringify(prompt)
    marker_pos = max(text.find("对话："), text.find("对话:"))
    search_from = marker_pos if marker_pos >= 0 else 0
    bracket_start = text.find("[", search_from)
    if bracket_start < 0:
        return "", ""
    blob = _balanced_bracket(text, bracket_start, "[", "]")
    value = parse_jsonish(blob)
    if isinstance(value, list):
        return extract_messages_qa(value)
    return "", ""


def _balanced_bracket(text: str, start: int, opener: str, closer: str) -> str:
    depth = 0
    in_string = False
    quote = ""
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if in_string:
            if ch == quote:
                in_string = False
            continue
        if ch in {"'", '"'}:
            in_string = True
            quote = ch
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return ""


def parse_sft_scores(record: dict[str, Any]) -> list[dict[str, Any]]:
    output = record.get("output", "")
    if not output and isinstance(record.get("solution"), str):
        output = record.get("solution")
    if not output and isinstance(record.get("messages"), list):
        _, output = extract_messages_qa(record.get("messages"))
    parsed = parse_jsonish(output)
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []
    rows = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "metric_raw": item.get("criterion") or item.get("principle") or item.get("metric"),
                "score_raw": item.get("score") or item.get("Score"),
                "reasoning": item.get("reason") or item.get("rationale") or item.get("理由"),
                "error_type": item.get("error_type") or item.get("错误类型"),
            }
        )
    return rows


def numeric_score(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isnan(float(value)):
            return None
        return float(value)
    text = stringify(value).strip()
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    return float(match.group(0))


def score_scale_for_path(path: Path, score: Any, metric_raw: Any = "") -> str:
    value = numeric_score(score)
    if value is None:
        return "unknown"
    rel = source_rel(path).lower()
    metric = canonicalize_metric(metric_raw).get("canonical_metric", "unknown")
    if "data/eval_data/" in rel or "groupby_metric" in rel or path.name in {"merge_model_metric.jsonl", "deepseek-r1_merged.jsonl"}:
        return "1-10" if value > 5 else "1-10_or_1-5_ambiguous_model_score"
    if path.name in {"sampled_merge_50_new.json", "sampled_merge_50_new_swift.json", "human_sampled_eval_sft_criteria_test.json"}:
        return "1-5"
    if "processed_excel_data" in rel and metric in OFFICIAL_METRIC_SET:
        return "1-5"
    if 1 <= value <= 5:
        return "1-5"
    if 0 <= value <= 10:
        return "0-10_or_task_score"
    return "unknown"


def label_5_from_score(path: Path, score: Any, metric_raw: Any = "") -> int | str:
    value = numeric_score(score)
    if value is None:
        return ""
    metric = canonicalize_metric(metric_raw).get("canonical_metric", "unknown")
    if metric not in OFFICIAL_METRIC_SET:
        return ""
    scale = score_scale_for_path(path, score, metric_raw)
    if scale.startswith("1-10"):
        if value < 1:
            return ""
        return min(5, max(1, int((value + 1) // 2)))
    if scale == "1-5":
        return min(5, max(1, int(round(value))))
    return ""


def normalize_metric(raw: Any) -> str:
    mapped = canonicalize_metric(raw)
    return mapped.get("canonical_metric") or "unknown"


def synthetic_marker_for_path(path: Path, row: dict[str, Any] | None = None) -> bool:
    rel = source_rel(path).lower()
    if any(token in rel for token in ["synthetic", "synthesis", "sampled", "generated", "augmented", "model_metric"]):
        return True
    if row:
        keys = " ".join(map(str, row.keys())).lower()
        if any(token in keys for token in ["generation", "generated", "gen", "is_synthetic"]):
            return True
    return False


def stable_question_hash(question: Any) -> str:
    return sha1_text(normalize_text(question))


def stable_qa_hash(question: Any, answer: Any) -> str:
    return sha1_text(normalize_text(question), normalize_text(answer))


def stable_triple_hash(question: Any, answer: Any, metric: Any) -> str:
    return sha1_text(normalize_text(question), normalize_text(answer), normalize_text(metric))


def make_synthetic_id(path: Path, row_index: int, metric_raw: Any, ordinal: int = 0) -> str:
    return sha1_text(source_rel(path), row_index, normalize_text(metric_raw), ordinal)[:16]


def base_language(record_language: Any, question: Any, answer: Any) -> str:
    lang = normalize_text(record_language)
    if lang in {"en", "english"}:
        return "en"
    if lang in {"zh", "cn", "chinese", "中文", "汉语"}:
        return "zh"
    return infer_language(question, answer)


def get_qa_from_record(record: dict[str, Any]) -> tuple[str, str]:
    question = stringify(record.get("question") or record.get("Question") or record.get("prompt") or record.get("query"))
    answer = stringify(
        record.get("answer")
        or record.get("response")
        or record.get("StandardAnswer")
        or record.get("StudentAnswer")
        or record.get("output")
        or record.get("completion")
    )
    if not question and isinstance(record.get("message"), list):
        question, answer = extract_messages_qa(record.get("message"))
    if not question and isinstance(record.get("messages"), list):
        question, answer = extract_messages_qa(record.get("messages"))
    if not question:
        question, answer_from_prompt = extract_dialogue_qa(record.get("instruction", ""))
        answer = answer or answer_from_prompt
    return question, answer


def source_keys(record: dict[str, Any], question: str, answer: str, metric_raw: Any) -> tuple[str, str]:
    question_key = stringify(
        record.get("source_question_key")
        or record.get("question_key")
        or record.get("question_id")
        or record.get("id")
        or record.get("__key__")
    )
    if not question_key:
        question_key = stable_question_hash(question)
    triple_key = stringify(record.get("source_triple_key") or record.get("triple_key"))
    if not triple_key:
        triple_key = stable_triple_hash(question, answer, metric_raw)
    return question_key, triple_key


def normalized_row(
    path: Path,
    row_index: int,
    record: dict[str, Any],
    question: str,
    answer: str,
    metric_raw: Any,
    score_raw: Any,
    reasoning: Any = "",
    error_type: Any = "",
    status: str = "normalized_review_only",
    notes: str = "",
    ordinal: int = 0,
) -> dict[str, Any]:
    metric_canonical = normalize_metric(metric_raw)
    question_key, triple_key = source_keys(record, question, answer, metric_raw)
    generation_method = (
        record.get("generation_method")
        or record.get("gen")
        or record.get("model")
        or record.get("generator_model")
        or record.get("GenerationTime")
        or ""
    )
    rubric_text = (
        record.get("rubric")
        or record.get("GradingCriteria")
        or record.get("criteria")
        or record.get("input")
        or ""
    )
    language = base_language(record.get("language") or record.get("Language"), question, answer)
    target_label = label_5_from_score(path, score_raw, metric_raw)
    scale = score_scale_for_path(path, score_raw, metric_raw)
    if metric_canonical == "unknown" and score_raw not in (None, ""):
        status = "non_edubench_task_score_review_only"
        notes = (notes + "; " if notes else "") + "score is not tied to an official EduBench metric"
    if not question or not answer:
        status = "not_normalized_missing_question_or_answer"
        notes = (notes + "; " if notes else "") + "missing question or answer"
    return {
        "synthetic_id": make_synthetic_id(path, row_index, metric_raw, ordinal),
        "source_file": source_rel(path),
        "source_row_index": row_index,
        "question": question,
        "answer": answer,
        "metric_raw": stringify(metric_raw),
        "metric_canonical": metric_canonical,
        "rubric_text": stringify(rubric_text),
        "language": language,
        "target_label_5": target_label,
        "score_raw": stringify(score_raw),
        "score_scale_detected": scale,
        "reasoning": stringify(reasoning),
        "error_type": stringify(error_type),
        "generation_method": stringify(generation_method),
        "source_question_key": question_key,
        "source_triple_key": triple_key,
        "is_synthetic": synthetic_marker_for_path(path, record),
        "normalization_status": status,
        "normalization_notes": notes,
    }


def normalize_records_from_file(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.is_dir() or path.suffix.lower() not in {".json", ".jsonl", ".csv", ".xlsx", ".xls"}:
        return []
    rows: list[dict[str, Any]] = []
    role = likely_role_for_path(path)
    for row_index, record in iter_tabular_records(path):
        question, answer = get_qa_from_record(record)
        if isinstance(record.get("scores"), list):
            for ordinal, score_item in enumerate(record["scores"]):
                if not isinstance(score_item, dict):
                    continue
                rows.append(
                    normalized_row(
                        path,
                        row_index,
                        record,
                        question,
                        answer,
                        score_item.get("criterion") or score_item.get("principle") or score_item.get("metric"),
                        score_item.get("score") or score_item.get("Score"),
                        score_item.get("reason") or score_item.get("rationale"),
                        score_item.get("error_type") or score_item.get("错误类型"),
                        status="synthetic_metric_label_review_only",
                        notes=f"source role={role}; labels require provenance review",
                        ordinal=ordinal,
                    )
                )
            continue
        if record.get("principle") or record.get("criterion") or record.get("metric"):
            rows.append(
                normalized_row(
                    path,
                    row_index,
                    record,
                    question,
                    answer,
                    record.get("principle") or record.get("criterion") or record.get("metric"),
                    record.get("score") or record.get("Score"),
                    record.get("reason") or record.get("rationale"),
                    record.get("error_type") or record.get("错误类型"),
                    status="model_judge_output_review_only" if role == "model_judge_output" else "synthetic_metric_label_review_only",
                    notes=f"source role={role}; not a primary human-label row",
                )
            )
            continue
        sft_scores = parse_sft_scores(record)
        if sft_scores:
            for ordinal, item in enumerate(sft_scores):
                rows.append(
                    normalized_row(
                        path,
                        row_index,
                        record,
                        question,
                        answer,
                        item.get("metric_raw"),
                        item.get("score_raw"),
                        item.get("reasoning"),
                        item.get("error_type"),
                        status="sft_sample_review_only",
                        notes=f"source role={role}; SFT prompt/output wrapper",
                        ordinal=ordinal,
                    )
                )
            continue
        if question or answer:
            rows.append(
                normalized_row(
                    path,
                    row_index,
                    record,
                    question,
                    answer,
                    record.get("QuestionType") or record.get("task") or record.get("type") or "",
                    record.get("Score") or record.get("score") or "",
                    record.get("ScoringDetails") or record.get("analysis") or "",
                    record.get("error_type") or "",
                    status="non_edubench_task_score_review_only",
                    notes=f"source role={role}; no official EduBench metric label found",
                )
            )
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    write_csv(path, rows, fields)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def count_by(rows: Iterable[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    counter: Counter[tuple[Any, ...]] = Counter()
    for row in rows:
        counter[tuple(row.get(key, "") for key in keys)] += 1
    out = []
    total = sum(counter.values())
    for values, count in sorted(counter.items(), key=lambda item: (item[0], item[1])):
        item = {key: values[idx] for idx, key in enumerate(keys)}
        item["count"] = count
        item["pct"] = round(count / total, 6) if total else 0
        out.append(item)
    return out
