"""Shared constants and deterministic helpers for Exp41A."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path("thesis_exp/exp41_rubric_bridge/outputs/exp41a_rubric_bridge_groupcv_seed42")
TRAIN_PATH = Path("thesis_exp/data/splits/paper_like_triple_seed42/train.jsonl")
PROCESSED_PATH = Path("thesis_exp/data/processed/edubench_scoring_all.jsonl")
PROMPT_PATH = Path("thesis_exp/exp41_rubric_bridge/prompts/exp41a_answer_blind_rubric_compiler.md")
SCHEMA_PATH = Path("thesis_exp/exp41_rubric_bridge/schemas/exp41a_compiled_rubric_schema.json")

VARIANTS = (
    "v0_original_hard",
    "v0h_human_soft",
    "v1_raw_rubric",
    "v2_deterministic_checklist",
    "v3_rubric_bridge",
    "v4_shuffled_compiled_rubric",
    "v5_human_soft_logit_adjustment",
)
FORMAL_VARIANTS = VARIANTS[1:]
CRITERION_TYPES = (
    "key_content", "factual_accuracy", "reasoning_support", "relevance_scope",
    "instruction_format", "scenario_integration", "pedagogical_support", "clarity", "other",
)
FORBIDDEN_UNIT_FIELDS = {
    "answer", "answer_id", "answer_key", "human_1", "human_2", "human_3", "human_mean_5",
    "label_5", "judge_scores", "teacher_score", "predicted_score", "model_prediction",
}


def reject_eval_path(path: Path) -> None:
    normalized = "/" + str(path).replace("\\", "/").lower().strip("/") + "/"
    if path.name.lower() in {"dev.jsonl", "test.jsonl", "dev.json", "test.json"}:
        raise ValueError(f"Exp41A forbids evaluation path: {path}")
    if "/paper_like_triple_seed42/dev." in normalized or "/paper_like_triple_seed42/test." in normalized:
        raise ValueError(f"Exp41A forbids paper-like dev/test: {path}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    reject_eval_path(path)
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    materialized = list(rows)
    fields = list(fieldnames or [])
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(value: Any) -> str:
    if isinstance(value, (list, dict)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value or ""))).strip()


def raw_rubric_text(row: dict[str, Any]) -> str:
    rubric = row.get("rubric", [])
    if isinstance(rubric, list):
        return "\n".join(str(item).strip() for item in rubric if str(item).strip())
    return str(rubric or "").strip()


def canonical_metric(row: dict[str, Any]) -> str:
    return str(row.get("metric_canonical") or row.get("metric_raw") or row.get("metric") or "").strip()


def rubric_unit_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["question_key"]), canonical_metric(row), normalize_text(raw_rubric_text(row)),
        str(row.get("language") or "unknown"),
    )


def rubric_unit_id(row: dict[str, Any]) -> str:
    return "exp41a_ru_" + stable_hash(rubric_unit_key(row))[:24]


def non_label_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata_raw = row.get("metadata_raw") if isinstance(row.get("metadata_raw"), dict) else {}
    return {
        "education_level": row.get("education_level_canonical") or row.get("education_level_raw"),
        "subject": row.get("subject_canonical") or row.get("subject_raw"),
        "task": metadata_raw.get("task"),
    }


def sample_id(row: dict[str, Any]) -> str:
    value = row.get("record_id") or row.get("sample_id") or row.get("id")
    if not value:
        raise ValueError("Missing sample ID")
    return str(value)


def human_distribution(row: dict[str, Any]) -> list[float]:
    scores = [int(float(row.get(f"human_{index}_5", row[f"human_{index}"]))) for index in (1, 2, 3)]
    return [scores.count(label) / 3.0 for label in range(1, 6)]


def human_stats(row: dict[str, Any]) -> dict[str, Any]:
    scores = [int(float(row.get(f"human_{index}_5", row[f"human_{index}"]))) for index in (1, 2, 3)]
    distribution = human_distribution(row)
    entropy = -sum(value * math.log(value) for value in distribution if value > 0)
    return {
        "human_distribution_5": distribution,
        "rounded_human_label_5": int(row["label_5"]),
        "expected_human_score": sum((index + 1) * value for index, value in enumerate(distribution)),
        "human_entropy": entropy,
        "human_score_range": max(scores) - min(scores),
    }


def lexical_tokens(text: str) -> set[str]:
    normalized = normalize_text(text).lower()
    return set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", normalized))


def serialize_compiled_rubric(compiled: dict[str, Any]) -> str:
    lines = ["[RUBRIC CHECKLIST]"]
    for criterion in compiled["criteria"]:
        lines.append(f"{criterion['criterion_id']}. {criterion['check_question']}")
        lines.append(f"Original clause: \"{criterion['rubric_quote']}\"")
    if compiled.get("score_level_rules"):
        lines.extend(["", "[SCORE LEVEL CONDITIONS]"])
        for rule in compiled["score_level_rules"]:
            ids = ", ".join(rule["required_criteria"])
            lines.append(f"Level {rule['score_level']}: {ids}; Original clause: \"{rule['rubric_quote']}\"")
    return "\n".join(lines)


def fit_compiled_rubric(compiled: dict[str, Any], tokenizer: Any, max_tokens: int = 256) -> tuple[dict[str, Any], str, int]:
    """Fit at complete criterion boundaries without rewriting teacher content."""
    criteria = list(compiled["criteria"])
    if not criteria:
        raise ValueError("Compiled rubric has no criteria")
    for keep in range(len(criteria), 0, -1):
        kept = criteria[:keep]
        kept_ids = {str(item["criterion_id"]) for item in kept}
        candidate = {
            "rubric_unit_id": compiled["rubric_unit_id"],
            "target_scope": compiled["target_scope"],
            "criteria": kept,
            "score_level_rules": [
                rule for rule in compiled.get("score_level_rules", [])
                if set(map(str, rule.get("required_criteria", []))) <= kept_ids
            ],
        }
        text = serialize_compiled_rubric(candidate)
        if len(tokenizer.encode(text, add_special_tokens=False)) <= max_tokens:
            return candidate, text, len(criteria) - keep
    raise ValueError("A single complete compiled criterion exceeds the rubric token budget")


def truncate_tokens(tokenizer: Any, text: str, max_tokens: int) -> tuple[str, int, bool]:
    token_ids = tokenizer.encode(str(text or ""), add_special_tokens=False)
    truncated = len(token_ids) > max_tokens
    kept = token_ids[:max_tokens]
    return tokenizer.decode(kept, skip_special_tokens=True), len(kept), truncated


def canonical_model_text(
    tokenizer: Any,
    row: dict[str, Any],
    rubric_representation: str | None,
    *,
    max_length: int = 2048,
    question_budget: int = 256,
    metric_budget: int = 64,
    rubric_budget: int = 256,
) -> tuple[str, dict[str, Any]]:
    """Format every variant with locked field budgets and answer-last truncation."""
    question, question_tokens, question_truncated = truncate_tokens(tokenizer, row.get("question", ""), question_budget)
    metric, metric_tokens, metric_truncated = truncate_tokens(tokenizer, canonical_metric(row), metric_budget)
    rubric = ""
    rubric_tokens = 0
    rubric_truncated = False
    if rubric_representation is not None:
        rubric, rubric_tokens, rubric_truncated = truncate_tokens(tokenizer, rubric_representation, rubric_budget)
    prefix_parts = [f"Question:\n{question}", f"Evaluation dimension:\n{metric}"]
    if rubric_representation is not None:
        prefix_parts.append(f"Rubric:\n{rubric}")
    prefix = "\n\n".join(prefix_parts) + "\n\nAnswer:\n"
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    answer_ids = tokenizer.encode(str(row.get("answer") or ""), add_special_tokens=False)
    if rubric_representation is not None:
        fixed_prefix = "\n\n".join(prefix_parts[:2] + ["Rubric:\n"]) + "\n\nAnswer:\n"
        # Reserve the complete rubric budget for every rubric-bearing variant.
        # A small boundary cushion prevents decode/re-encode merges from exceeding max_length.
        answer_budget = max(0, max_length - len(tokenizer.encode(fixed_prefix, add_special_tokens=False)) - rubric_budget - 4)
    else:
        answer_budget = max(0, max_length - len(prefix_ids))
    answer_truncated = len(answer_ids) > answer_budget
    answer = tokenizer.decode(answer_ids[:answer_budget], skip_special_tokens=True)
    text = prefix + answer
    total_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    if total_tokens > max_length:
        raise RuntimeError(f"Canonical formatter exceeded max length: {total_tokens}>{max_length}")
    return text, {
        "input_tokens": total_tokens,
        "question_tokens": question_tokens,
        "metric_tokens": metric_tokens,
        "rubric_tokens": rubric_tokens,
        "answer_tokens": min(len(answer_ids), answer_budget),
        "answer_token_budget": answer_budget,
        "question_truncated": question_truncated,
        "metric_truncated": metric_truncated,
        "rubric_truncated": rubric_truncated,
        "answer_truncated": answer_truncated,
    }


def deterministic_checklist(raw_rubric: str) -> str:
    clauses = []
    for block in raw_rubric.splitlines():
        for clause in re.split(r"(?:\s*[;；]\s*|(?<=[.!?。！？])\s+|^\s*[-*•]\s*)", block):
            cleaned = clause.strip(" \t-*•")
            if cleaned and cleaned not in clauses:
                clauses.append(cleaned)
    return "[RUBRIC CHECKLIST]\n" + "\n".join(f"Check: {clause}" for clause in clauses)


def qwk(gold: Iterable[int], pred: Iterable[int]) -> float:
    left, right = list(gold), list(pred)
    observed = np.zeros((5, 5), dtype=float)
    for a, b in zip(left, right):
        observed[int(a) - 1, int(b) - 1] += 1
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / max(observed.sum(), 1)
    weights = np.fromfunction(lambda i, j: ((i - j) ** 2) / 16.0, (5, 5), dtype=float)
    denominator = float((weights * expected).sum())
    return float("nan") if denominator == 0 else 1.0 - float((weights * observed).sum()) / denominator


def kendall_tau_b(gold: list[int], pred: list[int]) -> float:
    table = np.zeros((5, 5), dtype=int)
    for a, b in zip(gold, pred):
        table[int(a) - 1, int(b) - 1] += 1
    concordant = discordant = 0
    for i in range(5):
        for j in range(5):
            n = int(table[i, j])
            concordant += n * int(table[i + 1 :, j + 1 :].sum())
            discordant += n * int(table[i + 1 :, :j].sum())
    tied_gold = sum(int(n) * (int(n) - 1) // 2 for n in table.sum(axis=1))
    tied_pred = sum(int(n) * (int(n) - 1) // 2 for n in table.sum(axis=0))
    tied_both = sum(int(n) * (int(n) - 1) // 2 for n in table.reshape(-1))
    denominator = math.sqrt((concordant + discordant + tied_gold - tied_both) * (concordant + discordant + tied_pred - tied_both))
    return float((concordant - discordant) / denominator) if denominator else float("nan")


def prediction_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    gold = np.asarray([int(row["gold_label_5"]) for row in rows])
    pred = np.asarray([int(row["pred_label_5"]) for row in rows])
    expected = np.asarray([float(row["pred_score_expected"]) for row in rows])
    low, high = gold <= 2, gold >= 4
    result: dict[str, Any] = {
        "n": len(rows), "MAE": float(np.mean(np.abs(pred - gold))),
        "expected_score_MAE": float(np.mean(np.abs(expected - gold))),
        "Signed_Bias": float(np.mean(pred - gold)), "abs_Signed_Bias": float(abs(np.mean(pred - gold))),
        "Exact_Match": float(np.mean(pred == gold)), "Kendall_tau": kendall_tau_b(gold.tolist(), pred.tolist()),
        "QWK": qwk(gold.tolist(), pred.tolist()),
        "Bin_Agreement": float(np.mean(np.where(gold <= 2, 0, np.where(gold == 3, 1, 2)) == np.where(pred <= 2, 0, np.where(pred == 3, 1, 2)))),
        "low_n": int(low.sum()), "low_to_high_count": int(np.sum(low & (pred >= 4))),
        "low_to_high_rate": float(np.mean(pred[low] >= 4)) if low.any() else float("nan"),
        "high_n": int(high.sum()), "high_to_low_count": int(np.sum(high & (pred <= 2))),
        "high_to_low_rate": float(np.mean(pred[high] <= 2)) if high.any() else float("nan"),
    }
    counts = Counter(pred.tolist())
    for label in range(1, 6):
        mask = gold == label
        result[f"label{label}_recall"] = float(np.mean(pred[mask] == label)) if mask.any() else float("nan")
        result[f"pred_count_{label}"] = counts[label]
    return result


def human_distribution_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    targets = np.asarray([row["human_distribution_5"] for row in rows], dtype=float)
    probs = np.asarray([[row[f"prob_{k}"] for k in range(1, 6)] for row in rows], dtype=float)
    clipped = np.clip(probs, 1e-12, 1.0)
    return {
        "human_CE": float(np.mean(-np.sum(targets * np.log(clipped), axis=1))),
        "human_Brier": float(np.mean(np.sum((probs - targets) ** 2, axis=1))),
        "human_RPS": float(np.mean(np.sum((np.cumsum(probs, axis=1)[:, :-1] - np.cumsum(targets, axis=1)[:, :-1]) ** 2, axis=1) / 4.0)),
    }


def ensure_output_dirs(root: Path) -> None:
    for name in (
        "configs", "tables", "reports", "decision", "hashes", "prompts", "schemas", "raw_api",
        "private/rubric_units", "private/compiled_rubrics", "private/data", "private/groupcv_predictions",
        "private/checkpoints", "private/smoke", "logs_private",
    ):
        (root / name).mkdir(parents=True, exist_ok=True)
