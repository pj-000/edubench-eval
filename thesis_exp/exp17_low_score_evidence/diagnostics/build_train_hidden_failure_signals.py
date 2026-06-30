"""Build Exp17-A0 train-side hidden-failure weak signals.

This script constructs train-only weak labels and matched pairs from the
question-disjoint train split plus original 5-grade human rationale files. It
does not train a model, read test data, load checkpoints, or use dev D1
annotations as labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_TRAIN_JSONL = Path("thesis_exp/data/splits/question_seed42/train.jsonl")
DEFAULT_REASON_ROOT = Path("5-grades")
DEFAULT_D1_TAXONOMY_SUMMARY = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/"
    "d1_hidden_failure_audit_seed42_dev/summary_human_rationale_recovered/"
    "d1_failure_mode_summary.csv"
)
DEFAULT_OUT_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/"
    "exp17_a0_train_hidden_failure_signals_seed42"
)

REASON_FILES = [
    "5_merge_human_metric_en.jsonl",
    "5_merge_human_metric_zh.jsonl",
    "5_human_1.jsonl",
    "5_human_2.jsonl",
    "5_human_3.jsonl",
]

METRIC_ALIASES = {
    "Content Relevance & Scope Control": {
        "Content Relevance & Scope Control",
        "内容相关性与范围控制",
    },
    "Domain Knowledge Accuracy": {
        "Domain Knowledge Accuracy",
        "领域知识准确性",
    },
    "Basic Factual Accuracy": {
        "Basic Factual Accuracy",
        "基础事实准确性",
    },
    "Reasoning Process Rigor": {
        "Reasoning Process Rigor",
        "推理过程严谨性",
    },
    "Instruction Following & Task Completion": {
        "Instruction Following & Task Completion",
        "指令遵循与任务完成",
    },
    "Scenario Element Integration": {
        "Scenario Element Integration",
        "场景要素融合度",
        "场景要素整合",
    },
    "Personalization, Adaptation & Learning Support": {
        "Personalization, Adaptation & Learning Support",
        "个性化适配与学习支持",
        "个性化适应与学习支持",
    },
    "Higher-Order Thinking & Skill Development": {
        "Higher-Order Thinking & Skill Development",
        "促进高阶思维与能力发展",
    },
    "Clarity, Simplicity & Inspiration": {
        "Clarity, Simplicity & Inspiration",
        "清晰易懂与表达启发",
    },
    "Role & Tone Consistency": {
        "Role & Tone Consistency",
        "角色与语气一致性",
    },
    "Motivation, Guidance & Positive Feedback": {
        "Motivation, Guidance & Positive Feedback",
        "鼓励支持与正向反馈",
        "动机引导与正向反馈",
        "激励引导与积极反馈",
    },
    "Error Identification & Correction Precision": {
        "Error Identification & Correction Precision",
        "错误识别与纠正精确性",
    },
}

CANDIDATE_FIELDS = [
    "sample_id",
    "question_key",
    "question_group_id",
    "metric",
    "language",
    "subject",
    "gold_label",
    "human_1",
    "human_2",
    "human_3",
    "human_agreement_pattern",
    "question",
    "answer",
    "rubric",
    "metadata",
    "recovered_reason_summary",
    "rationale_match_status",
    "rationale_score_phrase",
    "rationale_score_phrase_consistency",
    "score_phrase_conflict_flag",
    "hidden_failure_candidate_type",
    "failure_mode_auto",
    "confidence_weight",
    "recommended_training_use",
]

CONTROL_FIELDS = [
    "sample_id",
    "question_key",
    "question_group_id",
    "metric",
    "language",
    "subject",
    "gold_label",
    "human_agreement_pattern",
    "question",
    "answer",
    "rubric",
    "metadata",
    "recovered_reason_summary",
    "clean_high_confidence_weight",
]

PAIR_FIELDS = [
    "low_sample_id",
    "high_sample_id",
    "match_score",
    "same_metric",
    "same_language",
    "same_subject",
    "same_rubric_hash",
    "same_boundary_key",
    "same_question_group",
    "low_candidate_type",
    "low_failure_mode_auto",
    "pair_weight",
    "recommended_pair_use",
]


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "\n".join(line.rstrip() for line in str(value).strip().splitlines())


def norm_ws(value: Any) -> str:
    return re.sub(r"\s+", " ", clean_text(value)).strip()


def norm_alnum(value: Any) -> str:
    return re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]", "", norm_ws(value))


def stable_hash(value: Any) -> str:
    return hashlib.sha1(clean_text(value).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def row_id(row: dict[str, Any]) -> str:
    return clean_text(row.get("record_id") or row.get("sample_id") or row.get("id"))


def question_key(row: dict[str, Any]) -> str:
    explicit = clean_text(row.get("question_key") or row.get("source_question_key"))
    if explicit:
        return explicit
    question = clean_text(row.get("question"))
    return stable_hash(question) if question else row_id(row)


def metric_name(row: dict[str, Any]) -> str:
    return clean_text(row.get("metric_canonical") or row.get("metric") or row.get("metric_raw"))


def rubric_text(row: dict[str, Any]) -> str:
    return clean_text(row.get("rubric") or row.get("rubric_text") or row.get("rubric_canonical"))


def metadata_text(row: dict[str, Any]) -> str:
    parts = [
        ("Scenario", row.get("scenario_canonical") or row.get("scenario")),
        ("Subject", row.get("subject_canonical") or row.get("subject")),
        ("Education Level", row.get("education_level_canonical") or row.get("education_level")),
        ("Language", row.get("language")),
        ("Metric Group", row.get("metric_group")),
    ]
    lines = [f"{name}: {clean_text(value)}" for name, value in parts if clean_text(value)]
    return "\n".join(lines)


def boundary_text(row: dict[str, Any]) -> str:
    chunks = [
        f"Question:\n{clean_text(row.get('question'))}",
        f"Evaluation Dimension:\n{metric_name(row)}",
        f"Rubric:\n{rubric_text(row)}",
        f"Metadata:\n{metadata_text(row)}",
    ]
    return "\n\n".join(chunk for chunk in chunks if chunk.strip())


def boundary_key(row: dict[str, Any]) -> str:
    return stable_hash(boundary_text(row))


def human_scores(row: dict[str, Any]) -> list[float]:
    out: list[float] = []
    for key in ["human_1", "human_2", "human_3"]:
        try:
            out.append(float(row.get(key, "")))
        except Exception:
            pass
    return out


def label_5(row: dict[str, Any]) -> int:
    return int(float(row.get("label_5") or row.get("label") or row.get("human_mean_5") or 0))


def human_agreement_pattern(row: dict[str, Any]) -> str:
    return "/".join(clean_text(row.get(key)) for key in ["human_1", "human_2", "human_3"])


def load_reason_rows(reason_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for rel in REASON_FILES:
        path = reason_root / rel
        if not path.exists():
            missing.append(str(path))
            continue
        for row in read_jsonl(path):
            row["_source_file"] = rel
            rows.append(row)
    return rows, missing


def build_reason_indexes(
    rows: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[tuple[str, str], list[dict[str, Any]]]]:
    exact: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    alnum: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        q = norm_ws(row.get("question", ""))
        a = norm_ws(row.get("response", ""))
        exact[(q, a)].append(row)
        alnum[(q, norm_alnum(a))].append(row)
    return exact, alnum


def metric_aliases(metric: str) -> set[str]:
    aliases = set(METRIC_ALIASES.get(metric, {metric}))
    aliases.add(metric)
    return aliases


def preferred_metric_rows(candidates: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    aliases = metric_aliases(metric)
    metric_rows = [row for row in candidates if clean_text(row.get("principle")) in aliases]
    merged = [row for row in metric_rows if "5_merge_human_metric" in clean_text(row.get("_source_file"))]
    return merged or metric_rows


def recover_rationales_for_row(
    row: dict[str, Any],
    exact_index: dict[tuple[str, str], list[dict[str, Any]]],
    alnum_index: dict[tuple[str, str], list[dict[str, Any]]],
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    q = norm_ws(row.get("question", ""))
    a = norm_ws(row.get("answer", ""))
    candidates = exact_index.get((q, a), [])
    if not candidates:
        candidates = alnum_index.get((q, norm_alnum(a)), [])
    metric_rows = preferred_metric_rows(candidates, metric_name(row))
    if metric_rows:
        return "metric_rationale_recovered", candidates, metric_rows
    if candidates:
        return "question_answer_matched_metric_unmatched", candidates, []
    return "question_answer_unmatched", [], []


def reason_summary(reason_rows: list[dict[str, Any]]) -> str:
    seen: list[str] = []
    for row in reason_rows:
        reason = norm_ws(row.get("reason"))
        if reason and reason not in seen:
            seen.append(reason)
    return " / ".join(seen[:3])


def recover_score_values(reason_rows: list[dict[str, Any]]) -> list[int]:
    values: list[int] = []
    for row in reason_rows:
        try:
            value = int(float(row.get("score")))
        except Exception:
            continue
        if value not in values:
            values.append(value)
    return values


SCORE_PHRASE_PATTERNS = [
    re.compile(r"(?:评分|评|给(?:到|出)?|对应(?:高|低)?分)\s*[:：=]?\s*(\d+(?:\.\d+)?)(?:\s*[-—~到至]\s*(\d+(?:\.\d+)?))?\s*分?"),
    re.compile(r"(\d+(?:\.\d+)?)(?:\s*[-—~到至]\s*(\d+(?:\.\d+)?))?\s*分"),
]


def extract_score_phrases(text: str) -> list[tuple[str, list[float]]]:
    out: list[tuple[str, list[float]]] = []
    seen: set[str] = set()
    for pattern in SCORE_PHRASE_PATTERNS:
        for match in pattern.finditer(text):
            phrase = match.group(0)
            if phrase in seen:
                continue
            seen.add(phrase)
            nums = [float(part) for part in match.groups() if part]
            if nums:
                out.append((phrase, nums))
    return out


def ten_to_five(value: float) -> int | None:
    value_int = int(round(value))
    if value_int < 1:
        return None
    if value_int <= 2:
        return 1
    if value_int <= 4:
        return 2
    if value_int <= 6:
        return 3
    if value_int <= 8:
        return 4
    if value_int <= 10:
        return 5
    return None


def phrase_matches_score(nums: list[float], score: int) -> bool:
    for value in nums:
        direct = int(round(value))
        if direct == score:
            return True
        mapped = ten_to_five(value)
        if mapped == score:
            return True
    return False


def score_phrase_diagnosis(reason_rows: list[dict[str, Any]]) -> tuple[str, str, str]:
    text = "\n".join(clean_text(row.get("reason")) for row in reason_rows)
    phrases = extract_score_phrases(text)
    phrase_text = "; ".join(phrase for phrase, _ in phrases)
    if not phrases:
        return "", "no_score_phrase", "0"
    scores = recover_score_values(reason_rows)
    if not scores:
        return phrase_text, "no_recovered_score", "0"
    checks = []
    for _phrase, nums in phrases:
        checks.append(any(phrase_matches_score(nums, score) for score in scores))
    if all(checks):
        return phrase_text, "consistent_with_recovered_score_or_10pt_mapping", "0"
    if any(checks):
        return phrase_text, "mixed_score_phrase", "1"
    return phrase_text, "score_phrase_conflict", "1"


def has_any(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)


def infer_failure_mode(row: dict[str, Any], reason: str, match_status: str) -> str:
    metric = metric_name(row)
    question = clean_text(row.get("question"))
    reason_zh = reason
    text_zh = f"{reason}\n{metric}\n{question}"
    if not reason and match_status != "metric_rationale_recovered":
        return "unclear"
    if has_any(reason_zh, ["评分5与", "评分 5", "评分维度", "字段", "非纯JSON", "json格式", "JSON格式", "格式错误", "格式不符合"]):
        if has_any(reason_zh, ["评分维度", "完全偏离评分指令", "偏离评分指令"]):
            return "task_constraint_violation"
        return "format_violation"
    if has_any(text_zh, ["答案错误", "关键事实错误", "事实错误", "知识性偏差", "不正确", "混淆人物", "标准答案"]):
        if has_any(question, ["A)", "B)", "C)", "D)", "Original Answer", "学生的答案"]):
            return "answer_key_or_reference_mismatch"
        return "factual_or_rubric_mismatch"
    if has_any(text_zh, ["未完整", "缺少", "缺失", "遗漏", "未涵盖", "不够完整", "关键职责", "核心工作", "核心内容"]):
        return "missing_key_point"
    if has_any(text_zh, ["推理", "逻辑链", "论证", "缺乏深入", "不够深入", "根源", "原因", "依据"]):
        return "insufficient_evidence"
    if has_any(text_zh, ["平淡", "启发", "引导", "鼓励", "通俗", "生动", "案例", "思考"]):
        return "surface_fluent_but_hidden_defect"
    if has_any(text_zh, ["指令", "任务", "要求", "约束"]):
        return "task_constraint_violation"
    return "surface_fluent_but_hidden_defect" if reason else "unclear"


def low_confidence_from_agreement(row: dict[str, Any]) -> float:
    scores = human_scores(row)
    if len(scores) != 3:
        return 0.5
    if all(score <= 2 for score in scores):
        return 1.0
    if max(scores) <= 3 and sum(scores) / len(scores) <= 2.34:
        return 0.85
    return 0.65


def classify_candidate(row: dict[str, Any], match_status: str, reason: str, score_conflict: str) -> tuple[str, str, float, str]:
    mode = infer_failure_mode(row, reason, match_status)
    base_conf = low_confidence_from_agreement(row)
    recovered = match_status == "metric_rationale_recovered"
    if not recovered and not reason:
        return "unclear", "unclear", 0.20, "review_only"
    if score_conflict == "1" and mode not in {"format_violation", "task_constraint_violation"}:
        return "conflict_or_exclude", mode, 0.20, "exclude"
    if mode == "format_violation":
        return "format_auxiliary", mode, min(base_conf, 0.90), "format_auxiliary"
    if mode == "task_constraint_violation" and has_any(reason, ["评分维度", "偏离评分指令", "格式"]):
        return "format_auxiliary", mode, min(base_conf, 0.85), "format_auxiliary"
    if mode == "answer_key_or_reference_mismatch":
        return "answer_key_dependent", mode, min(base_conf, 0.75), "pairwise_low"
    if mode == "factual_or_rubric_mismatch" and recovered:
        return "strong_evidence_positive", mode, min(base_conf, 1.00), "evidence_positive"
    if recovered and mode in {
        "missing_key_point",
        "surface_fluent_but_hidden_defect",
        "insufficient_evidence",
        "task_constraint_violation",
    }:
        return "weak_evidence_positive", mode, min(base_conf, 0.85), "weak_evidence_positive"
    if mode != "unclear":
        return "pairwise_low", mode, min(base_conf, 0.60), "pairwise_low"
    return "unclear", "unclear", 0.20, "review_only"


def base_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": row_id(row),
        "question_key": question_key(row),
        "question_group_id": question_key(row),
        "metric": metric_name(row),
        "language": clean_text(row.get("language")),
        "subject": clean_text(row.get("subject_canonical") or row.get("subject")),
        "gold_label": label_5(row),
        "human_1": clean_text(row.get("human_1")),
        "human_2": clean_text(row.get("human_2")),
        "human_3": clean_text(row.get("human_3")),
        "human_agreement_pattern": human_agreement_pattern(row),
        "question": clean_text(row.get("question")),
        "answer": clean_text(row.get("answer")),
        "rubric": rubric_text(row),
        "metadata": metadata_text(row),
    }


def build_candidates(
    train_rows: list[dict[str, Any]],
    exact_index: dict[tuple[str, str], list[dict[str, Any]]],
    alnum_index: dict[tuple[str, str], list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    enriched_by_id: dict[str, dict[str, Any]] = {}
    for row in train_rows:
        sample_id = row_id(row)
        match_status, _qa_rows, reason_rows = recover_rationales_for_row(row, exact_index, alnum_index)
        summary = reason_summary(reason_rows)
        phrase, consistency, conflict = score_phrase_diagnosis(reason_rows)
        enriched_by_id[sample_id] = {
            "row": row,
            "rationale_match_status": match_status,
            "recovered_reason_summary": summary,
            "rationale_score_phrase": phrase,
            "rationale_score_phrase_consistency": consistency,
            "score_phrase_conflict_flag": conflict,
            "rubric_hash": stable_hash(rubric_text(row)),
            "boundary_key": boundary_key(row),
            "answer_length": len(clean_text(row.get("answer"))),
        }
        if label_5(row) > 2:
            continue
        candidate_type, mode, weight, use = classify_candidate(row, match_status, summary, conflict)
        candidate = base_fields(row)
        candidate.update(
            {
                "recovered_reason_summary": summary,
                "rationale_match_status": match_status,
                "rationale_score_phrase": phrase,
                "rationale_score_phrase_consistency": consistency,
                "score_phrase_conflict_flag": conflict,
                "hidden_failure_candidate_type": candidate_type,
                "failure_mode_auto": mode,
                "confidence_weight": f"{weight:.4f}",
                "recommended_training_use": use,
            }
        )
        candidates.append(candidate)
    return candidates, enriched_by_id


def high_control_weight(row: dict[str, Any], conflict: str) -> float:
    if conflict == "1":
        return 0.0
    scores = human_scores(row)
    if len(scores) != 3:
        return 0.0
    if all(score == 5 for score in scores):
        return 1.0
    if all(score >= 4 for score in scores):
        return 0.85
    return 0.0


def build_controls(train_rows: list[dict[str, Any]], enriched_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    controls: list[dict[str, Any]] = []
    for row in train_rows:
        if label_5(row) < 4:
            continue
        enriched = enriched_by_id[row_id(row)]
        weight = high_control_weight(row, enriched["score_phrase_conflict_flag"])
        if weight <= 0:
            continue
        control = base_fields(row)
        control.update(
            {
                "recovered_reason_summary": enriched["recovered_reason_summary"],
                "clean_high_confidence_weight": f"{weight:.4f}",
            }
        )
        controls.append(control)
    return controls


def bool_int(value: bool) -> str:
    return "1" if value else "0"


def length_similarity(low_len: int, high_len: int) -> float:
    if max(low_len, high_len) == 0:
        return 0.0
    ratio = abs(low_len - high_len) / max(low_len, high_len)
    return max(0.0, 1.0 - ratio)


def pair_score(low: dict[str, Any], high: dict[str, Any], enriched_by_id: dict[str, dict[str, Any]]) -> float:
    low_meta = enriched_by_id[low["sample_id"]]
    high_meta = enriched_by_id[high["sample_id"]]
    score = 0.0
    score += 6.0 if low["metric"] == high["metric"] else 0.0
    score += 3.0 if low["language"] == high["language"] else 0.0
    score += 2.0 if low["subject"] == high["subject"] else 0.0
    score += 2.0 if low_meta["rubric_hash"] == high_meta["rubric_hash"] else 0.0
    score += 2.0 if low_meta["boundary_key"] == high_meta["boundary_key"] else 0.0
    score += 2.0 * length_similarity(low_meta["answer_length"], high_meta["answer_length"])
    score -= 5.0 if low["question_group_id"] == high["question_group_id"] else 0.0
    return score


def recommended_pair_use(candidate_type: str) -> str:
    if candidate_type in {"strong_evidence_positive", "weak_evidence_positive"}:
        return "evidence_pair"
    if candidate_type == "format_auxiliary":
        return "format_auxiliary_pair"
    return "pairwise_low"


def build_pairs(
    candidates: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    enriched_by_id: dict[str, dict[str, Any]],
    controls_per_low: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    usable_types = {
        "strong_evidence_positive",
        "weak_evidence_positive",
        "pairwise_low",
        "format_auxiliary",
        "answer_key_dependent",
    }
    pairs: list[dict[str, Any]] = []
    shuffled_controls = list(controls)
    rng.shuffle(shuffled_controls)
    for low in candidates:
        candidate_type = clean_text(low.get("hidden_failure_candidate_type"))
        if candidate_type not in usable_types:
            continue
        different_group = [control for control in shuffled_controls if control["question_group_id"] != low["question_group_id"]]
        pool = different_group if different_group else shuffled_controls
        ranked = sorted(pool, key=lambda control: pair_score(low, control, enriched_by_id), reverse=True)
        for high in ranked[:controls_per_low]:
            low_meta = enriched_by_id[low["sample_id"]]
            high_meta = enriched_by_id[high["sample_id"]]
            score = pair_score(low, high, enriched_by_id)
            low_weight = float(low.get("confidence_weight") or 0.0)
            high_weight = float(high.get("clean_high_confidence_weight") or 0.0)
            pair_weight = max(0.0, min(1.0, (score / 17.0) * low_weight * high_weight))
            pairs.append(
                {
                    "low_sample_id": low["sample_id"],
                    "high_sample_id": high["sample_id"],
                    "match_score": f"{score:.4f}",
                    "same_metric": bool_int(low["metric"] == high["metric"]),
                    "same_language": bool_int(low["language"] == high["language"]),
                    "same_subject": bool_int(low["subject"] == high["subject"]),
                    "same_rubric_hash": bool_int(low_meta["rubric_hash"] == high_meta["rubric_hash"]),
                    "same_boundary_key": bool_int(low_meta["boundary_key"] == high_meta["boundary_key"]),
                    "same_question_group": bool_int(low["question_group_id"] == high["question_group_id"]),
                    "low_candidate_type": candidate_type,
                    "low_failure_mode_auto": low["failure_mode_auto"],
                    "pair_weight": f"{pair_weight:.4f}",
                    "recommended_pair_use": recommended_pair_use(candidate_type),
                }
            )
    return pairs


def count_by(rows: list[dict[str, Any]], field: str) -> Counter:
    return Counter(clean_text(row.get(field)) for row in rows)


def pct(num: int, den: int) -> float:
    return num / den if den else 0.0


def recommendation(candidates: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> tuple[bool, list[str], dict[str, Any]]:
    clean_positive_types = {"strong_evidence_positive", "weak_evidence_positive"}
    positives = [row for row in candidates if row["hidden_failure_candidate_type"] in clean_positive_types]
    positive_count = len(positives)
    question_group_count = len({row["question_group_id"] for row in positives})
    metric_count = len({row["metric"] for row in positives})
    group_counts = Counter(row["question_group_id"] for row in positives)
    max_group_rate = max((count / positive_count for count in group_counts.values()), default=0.0)
    score_conflicts = sum(row["score_phrase_conflict_flag"] == "1" for row in positives)
    score_conflict_rate = pct(score_conflicts, positive_count)
    pair_count = len(pairs)
    checks = {
        "positive_count": positive_count,
        "question_group_count": question_group_count,
        "metric_count": metric_count,
        "max_question_group_rate": max_group_rate,
        "score_phrase_conflict_rate_among_positives": score_conflict_rate,
        "matched_pair_count": pair_count,
    }
    reasons = []
    if positive_count < 50:
        reasons.append("evidence_positive + weak_evidence_positive < 50")
    if question_group_count < 5:
        reasons.append("question_group_count < 5")
    if metric_count < 3:
        reasons.append("metric_count < 3")
    if max_group_rate > 0.50:
        reasons.append("max_question_group_rate > 0.50")
    if score_conflict_rate > 0.20:
        reasons.append("score_phrase_conflict_rate among positives > 0.20")
    if pair_count < 100:
        reasons.append("matched pair count < 100")
    return not reasons, reasons, checks


def write_report(
    out_dir: Path,
    train_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    missing_reason_files: list[str],
    d1_taxonomy_rows: list[dict[str, str]] | None,
) -> None:
    low_count = sum(label_5(row) <= 2 for row in train_rows)
    recovered_low = sum(row["rationale_match_status"] == "metric_rationale_recovered" for row in candidates)
    candidate_counts = count_by(candidates, "hidden_failure_candidate_type")
    positives = [
        row
        for row in candidates
        if row["hidden_failure_candidate_type"] in {"strong_evidence_positive", "weak_evidence_positive"}
    ]
    positive_group_count = len({row["question_group_id"] for row in positives})
    positive_metric_count = len({row["metric"] for row in positives})
    positive_group_counts = Counter(row["question_group_id"] for row in positives)
    max_group_rate = max((count / len(positives) for count in positive_group_counts.values()), default=0.0)
    score_phrase_conflict_count = sum(row["score_phrase_conflict_flag"] == "1" for row in candidates)
    exp17a1_ok, reasons, checks = recommendation(candidates, pairs)
    taxonomy_note = (
        f"Loaded optional D1 taxonomy summary rows: `{len(d1_taxonomy_rows)}`."
        if d1_taxonomy_rows is not None
        else "Optional D1 taxonomy summary not provided/read."
    )
    lines = [
        "# Exp17-A0 Train Hidden Failure Signal Quality Report",
        "",
        "This is a train-only signal construction diagnostic. It does not train a model, read test data, or use dev D1 annotations as train labels.",
        "",
        "## Inputs",
        "",
        "- Train split: `thesis_exp/data/splits/question_seed42/train.jsonl`",
        "- Human rationale files under `5-grades/`.",
        f"- Missing reason files: `{'; '.join(missing_reason_files) if missing_reason_files else 'none'}`",
        f"- {taxonomy_note}",
        "",
        "## Summary",
        "",
        f"- total train samples: `{len(train_rows)}`",
        f"- train low-label samples: `{low_count}`",
        f"- recovered rationale coverage among low-label samples: `{recovered_low}/{len(candidates)}` = `{pct(recovered_low, len(candidates)):.4f}`",
        f"- clean evidence-positive count: `{candidate_counts.get('strong_evidence_positive', 0)}`",
        f"- weak evidence-positive count: `{candidate_counts.get('weak_evidence_positive', 0)}`",
        f"- pairwise_low count: `{candidate_counts.get('pairwise_low', 0)}`",
        f"- format_auxiliary count: `{candidate_counts.get('format_auxiliary', 0)}`",
        f"- answer_key_dependent count: `{candidate_counts.get('answer_key_dependent', 0)}`",
        f"- conflict_or_exclude count: `{candidate_counts.get('conflict_or_exclude', 0)}`",
        f"- unclear count: `{candidate_counts.get('unclear', 0)}`",
        f"- score_phrase_conflict count: `{score_phrase_conflict_count}`",
        f"- number of question groups covered by clean positives: `{positive_group_count}`",
        f"- number of metrics covered by clean positives: `{positive_metric_count}`",
        f"- max question_group rate among positives: `{max_group_rate:.4f}`",
        f"- clean high controls: `{len(controls)}`",
        f"- matched pair count: `{len(pairs)}`",
        f"- Exp17-A1 recommended: `{exp17a1_ok}`",
        f"- Recommendation reason: `{'; '.join(reasons) if reasons else 'all A1 entry rules satisfied'}`",
        "",
        "## Candidate Type Counts",
        "",
        "| hidden_failure_candidate_type | n |",
        "|---|---:|",
    ]
    for key in [
        "strong_evidence_positive",
        "weak_evidence_positive",
        "pairwise_low",
        "format_auxiliary",
        "answer_key_dependent",
        "conflict_or_exclude",
        "unclear",
    ]:
        lines.append(f"| {key} | {candidate_counts.get(key, 0)} |")
    lines.extend(
        [
            "",
            "## Failure Mode Counts",
            "",
            "| failure_mode_auto | n |",
            "|---|---:|",
        ]
    )
    for key, value in count_by(candidates, "failure_mode_auto").most_common():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Exp17-A1 Entry Rule Check",
            "",
            "| check | value | required | pass |",
            "|---|---:|---:|---|",
            f"| evidence_positive + weak_evidence_positive | {checks['positive_count']} | >= 50 | {checks['positive_count'] >= 50} |",
            f"| question_group_count | {checks['question_group_count']} | >= 5 | {checks['question_group_count'] >= 5} |",
            f"| metric_count | {checks['metric_count']} | >= 3 | {checks['metric_count'] >= 3} |",
            f"| max_question_group_rate | {checks['max_question_group_rate']:.4f} | <= 0.50 | {checks['max_question_group_rate'] <= 0.50} |",
            f"| score_phrase_conflict_rate_among_positives | {checks['score_phrase_conflict_rate_among_positives']:.4f} | <= 0.20 | {checks['score_phrase_conflict_rate_among_positives'] <= 0.20} |",
            f"| matched_pair_count | {checks['matched_pair_count']} | >= 100 | {checks['matched_pair_count'] >= 100} |",
            "",
            "## Redaction Notice",
            "",
            "`train_hidden_failure_candidates.csv` and `train_clean_high_controls.csv` intentionally include raw `question`, `answer`, and recovered human rationale text for auditability. If these artifacts will be shared outside the project, create a redacted copy first.",
            "",
        ]
    )
    (out_dir / "exp17_a0_train_signal_quality_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--reason-root", type=Path, default=DEFAULT_REASON_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--controls-per-low", type=int, default=5)
    parser.add_argument("--d1-taxonomy-summary", type=Path, default=DEFAULT_D1_TAXONOMY_SUMMARY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.train_jsonl.exists():
        raise FileNotFoundError(f"Missing train JSONL: {args.train_jsonl}")
    if not args.reason_root.exists():
        raise FileNotFoundError(f"Missing reason root: {args.reason_root}")
    if args.controls_per_low < 1:
        raise ValueError("--controls-per-low must be >= 1")

    rng = random.Random(args.seed)
    train_rows = read_jsonl(args.train_jsonl)
    reason_rows, missing_reason_files = load_reason_rows(args.reason_root)
    exact_index, alnum_index = build_reason_indexes(reason_rows)
    d1_taxonomy_rows = read_csv(args.d1_taxonomy_summary) if args.d1_taxonomy_summary.exists() else None

    candidates, enriched_by_id = build_candidates(train_rows, exact_index, alnum_index)
    controls = build_controls(train_rows, enriched_by_id)
    pairs = build_pairs(candidates, controls, enriched_by_id, args.controls_per_low, rng)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "train_hidden_failure_candidates.csv", candidates, CANDIDATE_FIELDS)
    write_csv(args.out_dir / "train_clean_high_controls.csv", controls, CONTROL_FIELDS)
    write_csv(args.out_dir / "train_hidden_failure_pairs.csv", pairs, PAIR_FIELDS)
    write_report(args.out_dir, train_rows, candidates, controls, pairs, missing_reason_files, d1_taxonomy_rows)

    exp17a1_ok, reasons, _checks = recommendation(candidates, pairs)
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "train_rows": len(train_rows),
                "candidates": len(candidates),
                "controls": len(controls),
                "pairs": len(pairs),
                "exp17a1_recommended": exp17a1_ok,
                "reasons": reasons,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
