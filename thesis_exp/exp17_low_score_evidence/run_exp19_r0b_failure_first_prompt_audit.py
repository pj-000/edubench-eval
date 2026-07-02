"""Exp19-R0B Qwen3-4B failure-first prompt audit.

This script evaluates prompt-only ways to make a frozen instruction model look
for rubric-linked failures before assigning a 1-5 score. It does not train, does
not read test by default, and does not include human rationales or D1 labels in
the model prompt.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence.run_exp19_r0a_qwen4b_direct_baseline import (
    LABELS,
    clean_text,
    fmt,
    gold_label,
    json_metric,
    kendall_tau_b,
    mean,
    messages_to_prompt,
    quadratic_weighted_kappa,
    safe_float,
    safe_int,
    safe_rate,
    sample_id,
)
from thesis_exp.src.edujudge.utils.io import read_jsonl, relpath, write_csv, write_json, write_text


DEFAULT_DATA_PATH = Path("thesis_exp/data/splits/question_seed42/dev.jsonl")
DEFAULT_R0A_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_r0a_qwen4b_direct_baseline_seed42")
DEFAULT_D1_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/d1_hidden_failure_audit_seed42_dev/summary_human_rationale_recovered"
)
DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_r0b_failure_first_prompt_audit_seed42")
EVALUATOR = "Qwen3-4B-Instruct-failure-first"
SPLIT_NAME = "dev"
PROMPT_CONFIGS = [
    "R0B_0_direct_reproduction",
    "R0B_1_failure_first",
    "R0B_2_rubric_clause_first",
    "R0B_3_conservative_low_score_cap",
]
ALLOWED_FAILURES = [
    "missing_key_point",
    "factual_or_rubric_mismatch",
    "answer_key_or_reference_mismatch",
    "surface_fluent_but_hidden_defect",
    "insufficient_evidence",
    "task_constraint_violation",
    "format_violation",
    "possible_label_conflict",
    "no_major_failure",
    "unclear",
]
FAILURE_EVAL_TYPES = [name for name in ALLOWED_FAILURES if name not in {"no_major_failure", "unclear"}]

SYSTEM_MESSAGE = (
    "You are a strict educational assessment evaluator. Return only valid JSON. "
    "Do not mention hidden labels, training data, or model internals."
)
SCHEMA_TEXT = """Return only valid JSON with this schema:
{
  "rubric_clause": "<short rubric clause most relevant to the score>",
  "major_failures": ["missing_key_point"],
  "score_cap": 2,
  "score": 2,
  "brief_reason": "<one concise reason linked to the rubric>",
  "confidence": 0.74
}

Allowed major_failures values:
missing_key_point, factual_or_rubric_mismatch, answer_key_or_reference_mismatch,
surface_fluent_but_hidden_defect, insufficient_evidence, task_constraint_violation,
format_violation, possible_label_conflict, no_major_failure, unclear.

Use score_cap as null only when no cap is needed. score must be an integer from 1 to 5."""

COMMON_INPUT_TEMPLATE = """Question:
{question}

Answer:
{answer}

Evaluation metric:
{metric}

Rubric:
{rubric}

Metadata:
{metadata}"""

PROMPT_PREAMBLES = {
    "R0B_0_direct_reproduction": (
        "Assign a human-aligned 1-5 score directly from the question, answer, metric, rubric, and metadata. "
        "Fill the JSON fields, but do not force a failure if the answer satisfies the rubric."
    ),
    "R0B_1_failure_first": (
        "Before scoring, first check whether the answer has major rubric-linked failures. "
        "Look for missing required content, factual mismatches, unsupported reasoning, task constraint violations, "
        "or fluent wording that hides a substantive defect. Then assign the score."
    ),
    "R0B_2_rubric_clause_first": (
        "First identify the exact rubric clause that should control the score. "
        "Then compare the answer against that clause and assign a score. "
        "Do not reward fluent language if the answer misses the clause."
    ),
    "R0B_3_conservative_low_score_cap": (
        "Use conservative low-score caps. If there is a major required-content failure, a factual/rubric mismatch, "
        "or a task constraint violation, set score_cap to 2 or 3 and make the final score no higher than score_cap. "
        "Only give 4 or 5 when there is no major failure."
    ),
}

PROMPT_METRIC_FIELDS = [
    "prompt_config",
    "split",
    "n",
    "parse_success_rate",
    "invalid_score_rate",
    "MAE",
    "QWK",
    "Signed_Bias",
    "Exact_Match",
    "Kendall_tau",
    "low_to_high_count",
    "low_to_high_rate",
    "label1_recall",
    "label2_recall",
    "label2_pred_ge4_rate",
    "mean_pred_low",
    "median_pred_low",
]
D1_FIELDS = [
    "prompt_config",
    "n_d1_cases",
    "mean_pred_d1_hidden",
    "pred_ge4_rate_d1_hidden",
    "pred_5_rate_d1_hidden",
    "label2_recall_d1",
    "mean_score_cap_d1_hidden",
    "major_failure_nonempty_rate",
    "matched_control_mean_pred",
    "hidden_control_score_gap",
]
FAILURE_TYPE_FIELDS = [
    "prompt_config",
    "failure_type_micro_f1",
    "failure_type_macro_f1",
    "missing_key_point_recall",
    "factual_or_rubric_mismatch_recall",
    "insufficient_evidence_recall",
    "task_constraint_violation_recall",
    "no_major_failure_rate_on_controls",
]
BY_METRIC_FIELDS = [
    "prompt_config",
    "split",
    "metric",
    "n",
    "MAE",
    "QWK",
    "low_to_high_count",
    "low_to_high_rate",
    "label2_recall",
    "mean_pred_low",
]


def json_candidates(text: str) -> list[str]:
    out: list[str] = []
    stripped = clean_text(text)
    if stripped:
        out.append(stripped)
    fence = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        out.insert(0, fence.group(1).strip())
    stack = 0
    start = -1
    for idx, char in enumerate(stripped):
        if char == "{":
            if stack == 0:
                start = idx
            stack += 1
        elif char == "}":
            stack -= 1
            if stack == 0 and start >= 0:
                out.insert(0, stripped[start : idx + 1])
                start = -1
    seen = set()
    deduped = []
    for candidate in out:
        if candidate not in seen:
            seen.add(candidate)
            deduped.append(candidate)
    return deduped


def normalize_failure_name(value: Any) -> str | None:
    text = clean_text(value).lower().strip()
    if not text:
        return None
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    aliases = {
        "none": "no_major_failure",
        "no_failure": "no_major_failure",
        "no_major_failures": "no_major_failure",
        "major_failure": "unclear",
        "format": "format_violation",
        "task_constraint": "task_constraint_violation",
        "factual_mismatch": "factual_or_rubric_mismatch",
        "rubric_mismatch": "factual_or_rubric_mismatch",
    }
    text = aliases.get(text, text)
    return text if text in ALLOWED_FAILURES else None


def normalize_failure_list(value: Any) -> list[str]:
    items: list[Any]
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        pieces = re.split(r"[,;/\n]+", value)
        items = pieces if len(pieces) > 1 else [value]
    elif value is None:
        items = []
    else:
        items = [value]
    out = []
    for item in items:
        name = normalize_failure_name(item)
        if name and name not in out:
            out.append(name)
    return out


def safe_confidence(value: Any) -> float | None:
    out = safe_float(value)
    if math.isnan(out):
        return None
    if out < 0 or out > 1:
        return None
    return float(out)


def parse_r0b_output(raw_output: str, allow_regex_score_fallback: bool = False) -> dict[str, Any]:
    text = clean_text(raw_output)
    for candidate in json_candidates(text):
        try:
            data = json.loads(candidate)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        score = safe_int(data.get("score"))
        score_cap = safe_int(data.get("score_cap"))
        confidence = safe_confidence(data.get("confidence"))
        failures = normalize_failure_list(data.get("major_failures"))
        rubric_clause = clean_text(data.get("rubric_clause"))
        brief_reason = clean_text(data.get("brief_reason"))
        if score is not None:
            return {
                "parse_success": True,
                "pred_label": score,
                "parse_method": "json",
                "score_cap": score_cap,
                "confidence": confidence,
                "major_failures": failures,
                "rubric_clause": rubric_clause,
                "brief_reason": brief_reason,
            }
        return {
            "parse_success": False,
            "pred_label": None,
            "parse_method": "json_invalid_score",
            "score_cap": score_cap,
            "confidence": confidence,
            "major_failures": failures,
            "rubric_clause": rubric_clause,
            "brief_reason": brief_reason,
        }
    if allow_regex_score_fallback:
        regex = re.search(r"(?<![\d.])([1-5])(?![\d.])", text)
        if regex:
            return {
                "parse_success": True,
                "pred_label": int(regex.group(1)),
                "parse_method": "regex_fallback",
                "score_cap": None,
                "confidence": None,
                "major_failures": [],
                "rubric_clause": "",
                "brief_reason": "",
            }
    return {
        "parse_success": False,
        "pred_label": None,
        "parse_method": "failed",
        "score_cap": None,
        "confidence": None,
        "major_failures": [],
        "rubric_clause": "",
        "brief_reason": "",
    }


def metadata_for_prompt(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("metadata")
    if raw is None:
        raw = row.get("metadata_raw")
    metadata = {
        "language": row.get("language", ""),
        "subject": row.get("subject_canonical") or row.get("subject_raw") or "",
        "education_level": row.get("education_level_canonical") or row.get("education_level_raw") or "",
        "scenario": row.get("scenario_canonical") or row.get("scenario_raw") or "",
    }
    if raw:
        metadata["source_metadata"] = raw
    return metadata


def row_metric(row: dict[str, Any]) -> str:
    return clean_text(row.get("metric_canonical") or row.get("metric_raw") or row.get("metric_abbr"))


def prompt_messages(row: dict[str, Any], prompt_config: str) -> list[dict[str, str]]:
    user = "\n\n".join(
        [
            PROMPT_PREAMBLES[prompt_config],
            COMMON_INPUT_TEMPLATE.format(
                question=clean_text(row.get("question")),
                answer=clean_text(row.get("answer")),
                metric=row_metric(row),
                rubric=clean_text(row.get("rubric")),
                metadata=json.dumps(metadata_for_prompt(row), ensure_ascii=False, sort_keys=True),
            ),
            SCHEMA_TEXT,
        ]
    )
    return [{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": user}]


def fake_output(task_index: int, gold: int, prompt_config: str) -> str:
    if task_index % 11 == 10:
        return "The answer is mostly correct and deserves a high score."
    score = max(1, min(5, gold + ((task_index % 3) - 1)))
    failures = ["no_major_failure"] if score >= 4 else ["missing_key_point"]
    score_cap = None if score >= 4 else min(3, score)
    if prompt_config == "R0B_3_conservative_low_score_cap" and gold <= 2:
        score = min(score, 2)
        score_cap = 2
    return json.dumps(
        {
            "rubric_clause": "Relevant rubric clause",
            "major_failures": failures,
            "score_cap": score_cap,
            "score": score,
            "brief_reason": "Dry-run synthetic reason.",
            "confidence": 0.71,
        },
        ensure_ascii=False,
    )


def prediction_key(prompt_config: str, sid: str) -> str:
    return f"{prompt_config}::{sid}"


def load_dataset(data_path: Path, max_examples: int = 0) -> list[dict[str, Any]]:
    rows = read_jsonl(data_path)
    if max_examples > 0:
        rows = rows[:max_examples]
    out = []
    for idx, row in enumerate(rows):
        item = dict(row)
        sid = sample_id(item) or f"row_{idx}"
        item["sample_id"] = sid
        item["gold_label"] = gold_label(item)
        item["split"] = SPLIT_NAME
        out.append(item)
    return out


def append_prediction(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_existing_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = prediction_key(clean_text(row.get("prompt_config")), clean_text(row.get("sample_id")))
            if key.strip(":"):
                out[key] = row
    return out


def prediction_row(
    row: dict[str, Any],
    prompt_config: str,
    raw_output: str,
    model_name: str,
    backend: str,
    allow_regex_score_fallback: bool,
) -> dict[str, Any]:
    parsed = parse_r0b_output(raw_output, allow_regex_score_fallback)
    failures = parsed["major_failures"]
    return {
        "sample_id": row["sample_id"],
        "prompt_config": prompt_config,
        "split": row.get("split", SPLIT_NAME),
        "gold_label": row["gold_label"],
        "pred_label": parsed["pred_label"],
        "parse_success": bool(parsed["parse_success"]),
        "parse_method": parsed["parse_method"],
        "score_cap": parsed["score_cap"],
        "confidence": parsed["confidence"],
        "major_failures": failures,
        "major_failures_joined": "|".join(failures),
        "rubric_clause_truncated": clean_text(parsed["rubric_clause"])[:300],
        "brief_reason_truncated": clean_text(parsed["brief_reason"])[:500],
        "raw_output_truncated": clean_text(raw_output)[:800],
        "model_name_or_path": model_name,
        "backend": backend,
        "metric": row_metric(row),
        "language": clean_text(row.get("language")),
        "subject": clean_text(row.get("subject_canonical") or row.get("subject_raw")),
    }


def all_tasks(rows: list[dict[str, Any]], prompt_configs: list[str]) -> list[tuple[str, dict[str, Any]]]:
    return [(prompt_config, row) for prompt_config in prompt_configs for row in rows]


def expected_prediction_keys(rows: list[dict[str, Any]], prompt_configs: list[str]) -> set[str]:
    return {prediction_key(prompt_config, row["sample_id"]) for prompt_config, row in all_tasks(rows, prompt_configs)}


def run_dry_predictions(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    internal_path: Path,
    prompt_configs: list[str],
) -> list[dict[str, Any]]:
    if internal_path.exists():
        internal_path.unlink()
    predictions = []
    for idx, (prompt_config, row) in enumerate(all_tasks(rows, prompt_configs)):
        raw = fake_output(idx, int(row["gold_label"]), prompt_config)
        pred_row = prediction_row(
            row,
            prompt_config,
            raw,
            args.model_name_or_path,
            "dry_run",
            bool(args.allow_regex_score_fallback),
        )
        append_prediction(internal_path, pred_row)
        predictions.append(pred_row)
    return predictions


def run_vllm_predictions(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    internal_path: Path,
    done: dict[str, dict[str, Any]],
    prompt_configs: list[str],
) -> list[dict[str, Any]]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    llm = LLM(
        model=args.model_name_or_path,
        trust_remote_code=True,
        tensor_parallel_size=int(args.tensor_parallel_size),
        dtype=args.dtype,
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_model_len=int(args.max_model_len) if int(args.max_model_len) > 0 else None,
    )
    sampling = SamplingParams(
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_new_tokens),
        seed=int(args.seed),
    )
    predictions = list(done.values())
    tasks = all_tasks(rows, prompt_configs)
    pending = [(cfg, row) for cfg, row in tasks if prediction_key(cfg, row["sample_id"]) not in done]
    for start in range(0, len(pending), int(args.batch_size)):
        batch = pending[start : start + int(args.batch_size)]
        prompts = [
            messages_to_prompt(tokenizer, prompt_messages(row, cfg), bool(args.enable_thinking))
            for cfg, row in batch
        ]
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        for (prompt_config, row), output in zip(batch, outputs):
            text = output.outputs[0].text if output.outputs else ""
            pred_row = prediction_row(
                row,
                prompt_config,
                text,
                args.model_name_or_path,
                "vllm",
                bool(args.allow_regex_score_fallback),
            )
            append_prediction(internal_path, pred_row)
            predictions.append(pred_row)
        print(
            f"[exp19-r0b] vllm generated {min(start + len(batch), len(pending))}/{len(pending)} pending",
            flush=True,
        )
    return predictions


def run_transformers_predictions(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    internal_path: Path,
    done: dict[str, dict[str, Any]],
    prompt_configs: list[str],
) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = (
        torch.bfloat16
        if args.dtype in {"auto", "bfloat16", "bf16"}
        else torch.float16
        if args.dtype in {"float16", "fp16"}
        else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    predictions = list(done.values())
    tasks = all_tasks(rows, prompt_configs)
    pending = [(cfg, row) for cfg, row in tasks if prediction_key(cfg, row["sample_id"]) not in done]
    for start in range(0, len(pending), int(args.batch_size)):
        batch = pending[start : start + int(args.batch_size)]
        prompts = [
            messages_to_prompt(tokenizer, prompt_messages(row, cfg), bool(args.enable_thinking))
            for cfg, row in batch
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=int(args.max_new_tokens),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
        texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for (prompt_config, row), text in zip(batch, texts):
            pred_row = prediction_row(
                row,
                prompt_config,
                text,
                args.model_name_or_path,
                "transformers",
                bool(args.allow_regex_score_fallback),
            )
            append_prediction(internal_path, pred_row)
            predictions.append(pred_row)
        print(
            f"[exp19-r0b] transformers generated {min(start + len(batch), len(pending))}/{len(pending)} pending",
            flush=True,
        )
    return predictions


def prediction_groups(predictions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        groups[clean_text(row.get("prompt_config"))].append(row)
    return groups


def prompt_metric_summary(rows: list[dict[str, Any]], prompt_config: str, split: str = SPLIT_NAME) -> dict[str, Any]:
    n = len(rows)
    valid = [row for row in rows if row.get("pred_label") is not None and bool(row.get("parse_success"))]
    gold = [int(row["gold_label"]) for row in valid]
    pred = [int(row["pred_label"]) for row in valid]
    low = [row for row in rows if int(row["gold_label"]) <= 2]
    low_valid = [int(row["pred_label"]) for row in low if safe_int(row.get("pred_label")) is not None]
    label1 = [row for row in rows if int(row["gold_label"]) == 1]
    label2 = [row for row in rows if int(row["gold_label"]) == 2]
    low_to_high = sum(1 for row in low if safe_int(row.get("pred_label")) is not None and int(row["pred_label"]) >= 4)
    label1_hits = sum(1 for row in label1 if safe_int(row.get("pred_label")) == 1)
    label2_hits = sum(1 for row in label2 if safe_int(row.get("pred_label")) == 2)
    label2_ge4 = sum(1 for row in label2 if safe_int(row.get("pred_label")) is not None and int(row["pred_label"]) >= 4)
    exact = sum(1 for row in rows if safe_int(row.get("pred_label")) == int(row["gold_label"]))
    return {
        "prompt_config": prompt_config,
        "split": split,
        "n": n,
        "parse_success_rate": safe_rate(len(valid), n),
        "invalid_score_rate": safe_rate(n - len(valid), n),
        "MAE": mean([abs(g - p) for g, p in zip(gold, pred)]),
        "QWK": quadratic_weighted_kappa(gold, pred),
        "Signed_Bias": mean([p - g for g, p in zip(gold, pred)]),
        "Exact_Match": safe_rate(exact, n),
        "Kendall_tau": kendall_tau_b(gold, pred),
        "low_to_high_count": low_to_high,
        "low_to_high_rate": safe_rate(low_to_high, len(low)),
        "label1_recall": safe_rate(label1_hits, len(label1)),
        "label2_recall": safe_rate(label2_hits, len(label2)),
        "label2_pred_ge4_rate": safe_rate(label2_ge4, len(label2)),
        "mean_pred_low": mean(low_valid),
        "median_pred_low": float(median(low_valid)) if low_valid else float("nan"),
    }


def grouped_by_metric(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        groups[(clean_text(row.get("prompt_config")), clean_text(row.get("metric")) or "unknown")].append(row)
    out = []
    for (prompt_config, metric), rows in sorted(groups.items()):
        summary = prompt_metric_summary(rows, prompt_config)
        out.append(
            {
                "prompt_config": prompt_config,
                "split": SPLIT_NAME,
                "metric": metric,
                "n": len(rows),
                "MAE": summary["MAE"],
                "QWK": summary["QWK"],
                "low_to_high_count": summary["low_to_high_count"],
                "low_to_high_rate": summary["low_to_high_rate"],
                "label2_recall": summary["label2_recall"],
                "mean_pred_low": summary["mean_pred_low"],
            }
        )
    return out


def resolve_d1_base(path: Path) -> Path:
    path = Path(path)
    if (path / "d1_hidden_failure_annotation_template_filled_human_rationale_recovered.csv").exists():
        return path
    if path.name.startswith("summary") and (
        path.parent / "d1_hidden_failure_annotation_template_filled_human_rationale_recovered.csv"
    ).exists():
        return path.parent
    return path


def truthy(value: Any, blank_is_true: bool = False) -> bool:
    text = clean_text(value).lower()
    if not text:
        return bool(blank_is_true)
    return text in {"1", "true", "yes", "y"}


def load_d1_annotations(d1_dir: Path) -> tuple[dict[str, dict[str, str]], list[tuple[str, str]], set[str]]:
    base = resolve_d1_base(d1_dir)
    annotation_path = base / "d1_hidden_failure_annotation_template_filled_human_rationale_recovered.csv"
    annotations: dict[str, dict[str, str]] = {}
    if annotation_path.exists():
        with annotation_path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                sid = clean_text(row.get("sample_id"))
                if not sid:
                    continue
                hidden = truthy(row.get("is_hidden_failure_manual"), blank_is_true=True)
                if hidden:
                    annotations[sid] = row
    pair_path = base / "d1_matched_case_control_review.csv"
    pairs: list[tuple[str, str]] = []
    controls: set[str] = set()
    if pair_path.exists():
        with pair_path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                case_id = clean_text(row.get("case_sample_id"))
                control_id = clean_text(row.get("control_sample_id"))
                if case_id and control_id:
                    pairs.append((case_id, control_id))
                    controls.add(control_id)
    return annotations, pairs, controls


def has_substantive_failure(row: dict[str, Any]) -> bool:
    failures = normalize_failure_list(row.get("major_failures"))
    return any(name not in {"no_major_failure", "unclear"} for name in failures)


def has_no_major_failure(row: dict[str, Any]) -> bool:
    failures = normalize_failure_list(row.get("major_failures"))
    return bool(not failures or "no_major_failure" in failures)


def d1_eval_rows(predictions: list[dict[str, Any]], d1_dir: Path) -> list[dict[str, Any]]:
    annotations, pairs, control_ids = load_d1_annotations(d1_dir)
    out = []
    groups = prediction_groups(predictions)
    for prompt_config in PROMPT_CONFIGS:
        rows = groups.get(prompt_config, [])
        pred_by_id = {row["sample_id"]: row for row in rows}
        cases = [pred_by_id[sid] for sid in sorted(annotations) if sid in pred_by_id]
        controls = [pred_by_id[sid] for sid in sorted(control_ids) if sid in pred_by_id]
        case_preds = [int(row["pred_label"]) for row in cases if safe_int(row.get("pred_label")) is not None]
        control_preds = [int(row["pred_label"]) for row in controls if safe_int(row.get("pred_label")) is not None]
        label2 = [row for row in cases if int(row["gold_label"]) == 2]
        paired_gaps = []
        for case_id, control_id in pairs:
            if case_id in pred_by_id and control_id in pred_by_id:
                case_pred = safe_int(pred_by_id[case_id].get("pred_label"))
                control_pred = safe_int(pred_by_id[control_id].get("pred_label"))
                if case_pred is not None and control_pred is not None:
                    paired_gaps.append(control_pred - case_pred)
        caps = [safe_int(row.get("score_cap")) for row in cases]
        caps_valid = [int(value) for value in caps if value is not None]
        out.append(
            {
                "prompt_config": prompt_config,
                "n_d1_cases": len(cases),
                "mean_pred_d1_hidden": mean(case_preds),
                "pred_ge4_rate_d1_hidden": safe_rate(sum(1 for value in case_preds if value >= 4), len(cases)),
                "pred_5_rate_d1_hidden": safe_rate(sum(1 for value in case_preds if value == 5), len(cases)),
                "label2_recall_d1": safe_rate(
                    sum(1 for row in label2 if safe_int(row.get("pred_label")) == 2), len(label2)
                ),
                "mean_score_cap_d1_hidden": mean(caps_valid),
                "major_failure_nonempty_rate": safe_rate(sum(1 for row in cases if has_substantive_failure(row)), len(cases)),
                "matched_control_mean_pred": mean(control_preds),
                "hidden_control_score_gap": mean(paired_gaps),
            }
        )
    return out


def f1_score(tp: int, fp: int, fn: int) -> float:
    den = (2 * tp) + fp + fn
    return safe_rate(2 * tp, den)


def failure_type_eval_rows(predictions: list[dict[str, Any]], d1_dir: Path) -> list[dict[str, Any]]:
    annotations, _pairs, control_ids = load_d1_annotations(d1_dir)
    out = []
    groups = prediction_groups(predictions)
    for prompt_config in PROMPT_CONFIGS:
        rows = groups.get(prompt_config, [])
        pred_by_id = {row["sample_id"]: row for row in rows}
        counts: dict[str, Counter[str]] = {name: Counter() for name in FAILURE_EVAL_TYPES}
        micro = Counter()
        for sid, ann in annotations.items():
            if sid not in pred_by_id:
                continue
            gold = normalize_failure_name(ann.get("primary_failure_mode_manual")) or "unclear"
            pred_set = set(normalize_failure_list(pred_by_id[sid].get("major_failures")))
            pred_eval = {name for name in pred_set if name in FAILURE_EVAL_TYPES}
            if gold in FAILURE_EVAL_TYPES and gold in pred_eval:
                counts[gold]["tp"] += 1
                micro["tp"] += 1
            elif gold in FAILURE_EVAL_TYPES:
                counts[gold]["fn"] += 1
                micro["fn"] += 1
            for pred_name in pred_eval:
                if pred_name != gold:
                    counts[pred_name]["fp"] += 1
                    micro["fp"] += 1
        macro_values = []
        for name in FAILURE_EVAL_TYPES:
            if sum(counts[name].values()) > 0:
                macro_values.append(f1_score(counts[name]["tp"], counts[name]["fp"], counts[name]["fn"]))
        controls = [pred_by_id[sid] for sid in sorted(control_ids) if sid in pred_by_id]
        out.append(
            {
                "prompt_config": prompt_config,
                "failure_type_micro_f1": f1_score(micro["tp"], micro["fp"], micro["fn"]),
                "failure_type_macro_f1": mean(macro_values),
                "missing_key_point_recall": safe_rate(
                    counts["missing_key_point"]["tp"],
                    counts["missing_key_point"]["tp"] + counts["missing_key_point"]["fn"],
                ),
                "factual_or_rubric_mismatch_recall": safe_rate(
                    counts["factual_or_rubric_mismatch"]["tp"],
                    counts["factual_or_rubric_mismatch"]["tp"] + counts["factual_or_rubric_mismatch"]["fn"],
                ),
                "insufficient_evidence_recall": safe_rate(
                    counts["insufficient_evidence"]["tp"],
                    counts["insufficient_evidence"]["tp"] + counts["insufficient_evidence"]["fn"],
                ),
                "task_constraint_violation_recall": safe_rate(
                    counts["task_constraint_violation"]["tp"],
                    counts["task_constraint_violation"]["tp"] + counts["task_constraint_violation"]["fn"],
                ),
                "no_major_failure_rate_on_controls": safe_rate(sum(1 for row in controls if has_no_major_failure(row)), len(controls)),
            }
        )
    return out


def load_r0a_dev_baseline(r0a_dir: Path) -> dict[str, float]:
    path = Path(r0a_dir) / "tables" / "qwen3_4b_direct_overall_metrics.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if clean_text(row.get("split")) == SPLIT_NAME:
                return {
                    "low_to_high_rate": safe_float(row.get("low_to_high_rate")),
                    "label2_recall": safe_float(row.get("label2_recall")),
                    "MAE": safe_float(row.get("MAE")),
                    "QWK": safe_float(row.get("QWK")),
                }
    return {}


def prompt_success(metric: dict[str, Any], d1_row: dict[str, Any], r0a_dev: dict[str, float]) -> bool:
    parse_success = safe_float(metric.get("parse_success_rate"))
    low_to_high = safe_float(metric.get("low_to_high_rate"))
    label2_recall = safe_float(metric.get("label2_recall"))
    d1_ge4 = safe_float(d1_row.get("pred_ge4_rate_d1_hidden"))
    r0a_l2h = r0a_dev.get("low_to_high_rate", float("nan"))
    improved_enough = (not math.isnan(r0a_l2h)) and (r0a_l2h - low_to_high >= 0.20)
    return bool(
        parse_success >= 0.95
        and (low_to_high <= 0.60 or improved_enough)
        and label2_recall >= 0.20
        and d1_ge4 <= 0.50
    )


def select_best_prompt(
    metric_rows: list[dict[str, Any]],
    d1_rows_by_prompt: dict[str, dict[str, Any]],
    failure_rows_by_prompt: dict[str, dict[str, Any]],
    r0a_dev: dict[str, float],
) -> dict[str, Any]:
    scored = []
    for row in metric_rows:
        prompt_config = clean_text(row.get("prompt_config"))
        d1_row = d1_rows_by_prompt.get(prompt_config, {})
        failure_row = failure_rows_by_prompt.get(prompt_config, {})
        success = prompt_success(row, d1_row, r0a_dev)
        scored.append(
            (
                1 if success else 0,
                safe_float(row.get("parse_success_rate")),
                -safe_float(row.get("low_to_high_rate")),
                safe_float(row.get("label2_recall")),
                -safe_float(d1_row.get("pred_ge4_rate_d1_hidden")),
                safe_float(failure_row.get("failure_type_micro_f1")),
                -safe_float(row.get("MAE")),
                prompt_config,
                row,
            )
        )
    scored.sort(reverse=True)
    return scored[0][-1] if scored else {}


def decision_json(
    metric_rows: list[dict[str, Any]],
    d1_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    r0a_dev: dict[str, float],
) -> dict[str, Any]:
    d1_by_prompt = {row["prompt_config"]: row for row in d1_rows}
    failure_by_prompt = {row["prompt_config"]: row for row in failure_rows}
    best = select_best_prompt(metric_rows, d1_by_prompt, failure_by_prompt, r0a_dev)
    best_name = clean_text(best.get("prompt_config"))
    best_d1 = d1_by_prompt.get(best_name, {})
    best_failure = failure_by_prompt.get(best_name, {})
    success = prompt_success(best, best_d1, r0a_dev) if best else False
    best_l2h = safe_float(best.get("low_to_high_rate"))
    r0a_l2h = r0a_dev.get("low_to_high_rate", float("nan"))
    improves_l2h = (not math.isnan(r0a_l2h)) and (r0a_l2h - best_l2h > 0)
    failure_promising = safe_float(best_failure.get("failure_type_micro_f1")) >= 0.35 or safe_float(
        best_d1.get("major_failure_nonempty_rate")
    ) >= 0.60
    proceed_to_lora = bool((improves_l2h and not success) or (failure_promising and not success))
    reason = "Prompt audit passes all gates; lock this prompt for held-out evaluation." if success else (
        "Prompting shows partial signal but does not pass all low-score gates; use results to decide whether LoRA/SFT is needed."
        if proceed_to_lora
        else "Prompting does not show enough evidence to lock a prompt or proceed directly to LoRA."
    )
    return {
        "best_prompt_config": best_name,
        "parse_success_best": json_metric(best.get("parse_success_rate")),
        "dev_low_to_high_best": json_metric(best.get("low_to_high_rate")),
        "dev_label2_recall_best": json_metric(best.get("label2_recall")),
        "d1_hidden_pred_ge4_best": json_metric(best_d1.get("pred_ge4_rate_d1_hidden")),
        "prompt_audit_success": success,
        "lock_prompt_for_test_all": success,
        "proceed_to_lora": proceed_to_lora,
        "recommended_lora_variant": "failure-first SFT/LoRA with rationale-free inference JSON schema" if proceed_to_lora else "",
        "reason": reason,
    }


def write_report(
    args: argparse.Namespace,
    metric_rows: list[dict[str, Any]],
    d1_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    decision: dict[str, Any],
    r0a_dev: dict[str, float],
    backend_used: str,
) -> None:
    metric_by_prompt = {row["prompt_config"]: row for row in metric_rows}
    d1_by_prompt = {row["prompt_config"]: row for row in d1_rows}
    failure_by_prompt = {row["prompt_config"]: row for row in failure_rows}
    best = decision.get("best_prompt_config", "")
    r0a_l2h = r0a_dev.get("low_to_high_rate", float("nan"))
    best_l2h = safe_float(metric_by_prompt.get(best, {}).get("low_to_high_rate"))
    improvement = r0a_l2h - best_l2h if not math.isnan(r0a_l2h) and not math.isnan(best_l2h) else float("nan")
    lines = [
        "# Exp19-R0B Failure-First Prompt Audit",
        "",
        "## Setup",
        "",
        f"- Dataset: `{relpath(args.data_path)}`.",
        f"- R0A baseline dir: `{relpath(args.r0a_dir)}`.",
        f"- D1 evaluation dir: `{relpath(args.d1_dir)}`.",
        f"- Model: `{args.model_name_or_path}`.",
        f"- Backend used: `{backend_used}`.",
        "- Training: none.",
        "- Test split: not read by default.",
        "- Prompt inputs: question, answer, metric, rubric, and metadata only.",
        "- Human score, gold label, human rationale, D1 labels, failure-type gold, existing judge scores, and Exp16A predictions are not included in prompts.",
        "",
        "## Prompt Metrics",
        "",
        "| prompt | n | parse | MAE | QWK | low-to-high | label2 recall | D1 hidden pred>=4 | failure micro-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for prompt_config in PROMPT_CONFIGS:
        row = metric_by_prompt.get(prompt_config, {})
        d1 = d1_by_prompt.get(prompt_config, {})
        failure = failure_by_prompt.get(prompt_config, {})
        lines.append(
            f"| `{prompt_config}` | {row.get('n', 0)} | {fmt(row.get('parse_success_rate'))} | "
            f"{fmt(row.get('MAE'))} | {fmt(row.get('QWK'))} | {fmt(row.get('low_to_high_rate'))} | "
            f"{fmt(row.get('label2_recall'))} | {fmt(d1.get('pred_ge4_rate_d1_hidden'))} | "
            f"{fmt(failure.get('failure_type_micro_f1'))} |"
        )
    lines.extend(
        [
            "",
            "## Answers",
            "",
            f"- Does failure-first prompting reduce low-to-high vs R0A direct dev baseline? Best prompt `{best}` has low-to-high `{fmt(best_l2h)}`; R0A dev is `{fmt(r0a_l2h)}`; absolute improvement is `{fmt(improvement)}`.",
            f"- Which prompt config is best? `{best}`.",
            f"- Does rubric-clause-first reduce D1 hidden pred>=4? `R0B_2_rubric_clause_first` D1 pred>=4 is `{fmt(d1_by_prompt.get('R0B_2_rubric_clause_first', {}).get('pred_ge4_rate_d1_hidden'))}`.",
            f"- Does conservative score-cap improve label2 recall? `R0B_3_conservative_low_score_cap` label2 recall is `{fmt(metric_by_prompt.get('R0B_3_conservative_low_score_cap', {}).get('label2_recall'))}`.",
            f"- Does parse success remain acceptable? Best prompt parse success is `{fmt(decision.get('parse_success_best'))}`.",
            f"- Does model still over-score low samples? Best prompt low-to-high is `{fmt(decision.get('dev_low_to_high_best'))}`.",
            f"- Should lock prompt and run test/all_5536? `{decision.get('lock_prompt_for_test_all')}`.",
            f"- Should proceed to LoRA/SFT? `{decision.get('proceed_to_lora')}`.",
            "",
            "## Decision",
            "",
            "```json",
            json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
            "## Guardrails",
            "",
            "- Full internal predictions are written under `predictions/full_predictions_internal.jsonl` and must not be committed.",
            "- Raw prompts and raw answers are not written to lightweight report artifacts.",
        ]
    )
    write_text(Path(args.out_dir) / "reports" / "exp19_r0b_failure_first_prompt_audit_report.md", "\n".join(lines))


def write_all_outputs(
    args: argparse.Namespace,
    predictions: list[dict[str, Any]],
    backend_used: str,
) -> None:
    out_dir = Path(args.out_dir)
    tables_dir = out_dir / "tables"
    decision_dir = out_dir / "decision"
    groups = prediction_groups(predictions)
    metric_rows = [prompt_metric_summary(groups.get(prompt_config, []), prompt_config) for prompt_config in PROMPT_CONFIGS]
    d1_rows = d1_eval_rows(predictions, Path(args.d1_dir)) if resolve_d1_base(Path(args.d1_dir)).exists() else []
    failure_rows = failure_type_eval_rows(predictions, Path(args.d1_dir)) if resolve_d1_base(Path(args.d1_dir)).exists() else []
    r0a_dev = load_r0a_dev_baseline(Path(args.r0a_dir))
    decision = decision_json(metric_rows, d1_rows, failure_rows, r0a_dev)

    write_csv(tables_dir / "exp19_r0b_prompt_metrics.csv", metric_rows, PROMPT_METRIC_FIELDS)
    write_csv(tables_dir / "exp19_r0b_d1_hidden_eval.csv", d1_rows, D1_FIELDS)
    write_csv(tables_dir / "exp19_r0b_failure_type_eval.csv", failure_rows, FAILURE_TYPE_FIELDS)
    write_csv(tables_dir / "exp19_r0b_by_metric.csv", grouped_by_metric(predictions), BY_METRIC_FIELDS)
    write_json(decision_dir / "exp19_r0b_decision.json", decision)
    write_report(args, metric_rows, d1_rows, failure_rows, decision, r0a_dev, backend_used)


def parse_prompt_configs(value: str) -> list[str]:
    items = [item.strip() for item in re.split(r"[, ]+", value) if item.strip()]
    invalid = [item for item in items if item not in PROMPT_CONFIGS]
    if invalid:
        raise ValueError(f"Unknown prompt configs: {invalid}. Allowed: {PROMPT_CONFIGS}")
    return items or list(PROMPT_CONFIGS)


def infer_backend_from_predictions(predictions: Iterable[dict[str, Any]], fallback: str) -> str:
    seen = {clean_text(row.get("backend")) for row in predictions if clean_text(row.get("backend"))}
    if not seen:
        return fallback
    if len(seen) == 1:
        return next(iter(seen))
    return "+".join(sorted(seen))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Exp19-R0B Qwen3-4B failure-first prompt audit.")
    parser.add_argument("--model_name_or_path", default="/home/jpang/models/modelscope/Qwen/Qwen3-4B")
    parser.add_argument("--backend", choices=["transformers", "vllm"], default="vllm")
    parser.add_argument("--data_path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--r0a_dir", type=Path, default=DEFAULT_R0A_DIR)
    parser.add_argument("--d1_dir", type=Path, default=DEFAULT_D1_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prompt_configs", default=" ".join(PROMPT_CONFIGS))
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=0)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--allow_regex_score_fallback", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    prompt_configs = parse_prompt_configs(args.prompt_configs)
    out_dir = Path(args.out_dir)
    internal_path = out_dir / "predictions" / "full_predictions_internal.jsonl"
    if bool(args.overwrite) and internal_path.exists():
        internal_path.unlink()
    data_rows = load_dataset(Path(args.data_path), int(args.max_examples))
    expected_keys = expected_prediction_keys(data_rows, prompt_configs)
    if bool(args.dry_run):
        predictions = run_dry_predictions(args, data_rows, internal_path, prompt_configs)
        backend_used = "dry_run"
    else:
        done = load_existing_predictions(internal_path)
        if expected_keys.issubset(set(done)):
            predictions = list(done.values())
            backend_used = infer_backend_from_predictions(predictions, str(args.backend))
            print(f"[exp19-r0b] using existing internal predictions: {len(predictions)}", flush=True)
        elif args.backend == "vllm":
            try:
                predictions = run_vllm_predictions(args, data_rows, internal_path, done, prompt_configs)
                backend_used = "vllm"
            except Exception as exc:
                print(f"WARNING: vLLM backend failed or is unavailable ({exc}); falling back to transformers.", flush=True)
                predictions = run_transformers_predictions(args, data_rows, internal_path, done, prompt_configs)
                backend_used = "transformers_fallback"
        else:
            predictions = run_transformers_predictions(args, data_rows, internal_path, done, prompt_configs)
            backend_used = "transformers"
        predictions = list(load_existing_predictions(internal_path).values())
    write_all_outputs(args, predictions, backend_used)
    print(
        json.dumps(
            {
                "status": "COMPLETED",
                "out_dir": relpath(out_dir),
                "n_predictions": len(predictions),
                "dry_run": bool(args.dry_run),
                "backend": backend_used,
                "prompt_configs": prompt_configs,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
