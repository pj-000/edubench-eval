"""Exp19-R0C calibrated failure-first prompt audit.

This script evaluates prompt-only calibration variants for Qwen3-4B. It does
not train, does not read test by default, and never places human rationale,
gold labels, D1 labels, existing judge scores, or Exp16A predictions in the
prompt.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import time
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
    prompt_messages as r0a_prompt_messages,
    quadratic_weighted_kappa,
    safe_float,
    safe_int,
    safe_rate,
    sample_id,
)
from thesis_exp.exp17_low_score_evidence.run_exp19_r0b_failure_first_prompt_audit import (
    FAILURE_EVAL_TYPES,
    all_tasks,
    append_prediction,
    expected_prediction_keys,
    f1_score,
    has_no_major_failure,
    has_substantive_failure,
    json_candidates,
    load_d1_annotations,
    load_existing_predictions,
    metadata_for_prompt,
    normalize_failure_list,
    normalize_failure_name,
    prediction_key,
    resolve_d1_base,
    row_metric,
    safe_confidence,
)
from thesis_exp.src.edujudge.utils.io import read_jsonl, relpath, write_csv, write_json, write_text


DEFAULT_DATA_PATH = Path("thesis_exp/data/splits/question_seed42/dev.jsonl")
DEFAULT_R0A_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_r0a_qwen4b_direct_baseline_seed42")
DEFAULT_R0B_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_r0b_failure_first_prompt_audit_seed42")
DEFAULT_D1_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/d1_hidden_failure_audit_seed42_dev/summary_human_rationale_recovered"
)
DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_r0c_calibrated_prompt_audit_seed42")
SPLIT_NAME = "dev"
PROMPT_CONFIGS = [
    "R0C_0_exact_r0a_reproduction",
    "R0C_1_balanced_failure_first",
    "R0C_2_evidence_required_cap",
    "R0C_3_two_pass_balanced",
    "R0C_4_high_score_protection",
]
CALIBRATED_PROMPTS = [name for name in PROMPT_CONFIGS if name != "R0C_0_exact_r0a_reproduction"]
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

SYSTEM_MESSAGE = (
    "You are a calibrated educational assessment evaluator. Return only valid JSON. "
    "Do not mention hidden labels, training data, or model internals."
)
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
SCHEMA_TEXT = """Return only valid JSON with this schema:
{
  "rubric_clause": "<short rubric clause most relevant to the score>",
  "major_failures": ["missing_key_point"],
  "failure_evidence": "<specific evidence tied to the rubric, or empty if no major failure>",
  "rubric_satisfied": false,
  "score_cap": 2,
  "score": 2,
  "brief_reason": "<one concise reason linked to the rubric>",
  "confidence": 0.74
}

Allowed major_failures values:
missing_key_point, factual_or_rubric_mismatch, answer_key_or_reference_mismatch,
surface_fluent_but_hidden_defect, insufficient_evidence, task_constraint_violation,
format_violation, possible_label_conflict, no_major_failure, unclear.

Use score_cap as null when no cap is justified. score must be an integer from 1 to 5.
Do not apply a low score cap unless the failure evidence is concrete and tied to the rubric."""

PROMPT_PREAMBLES = {
    "R0C_1_balanced_failure_first": (
        "Identify major failures first, but only apply a score cap if the failure is concrete and rubric-linked. "
        "If no major failure is found and the answer substantially satisfies the rubric, allow 4 or 5. "
        "Do not penalize merely for being concise unless the rubric requires detail."
    ),
    "R0C_2_evidence_required_cap": (
        "A score cap is valid only if you provide specific failure_evidence tied to the rubric. "
        "If the evidence is generic, weak, or not rubric-linked, score_cap must be null or at least 4. "
        "Use low scores only when the concrete defect justifies them."
    ),
    "R0C_3_two_pass_balanced": (
        "Use a two-pass process. Step 1: assign a preliminary score from the answer and rubric. "
        "Step 2: audit for concrete major failures. Step 3: apply a score cap only for severe rubric-linked failures. "
        "Step 4: output the final score. If final score differs from the preliminary score, explain why in brief_reason."
    ),
    "R0C_4_high_score_protection": (
        "Protect substantially correct answers. Use 1-2 only for severe failures. "
        "Use 3 for partial but meaningful answers. Use 4-5 for substantially correct answers. "
        "Do not assign below 4 merely because an answer is concise or imperfect if rubric requirements are substantially satisfied."
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
    "high_to_low_count",
    "high_to_low_rate",
    "label1_recall",
    "label2_recall",
    "label4_recall",
    "label5_recall",
    "label2_pred_ge4_rate",
    "mean_pred_low",
    "median_pred_low",
    "mean_pred_high",
    "median_pred_high",
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
PER_LABEL_FIELDS = [
    "prompt_config",
    "gold_label",
    "n",
    "exact_accuracy",
    "mean_pred",
    "pred_1_rate",
    "pred_2_rate",
    "pred_3_rate",
    "pred_4_rate",
    "pred_5_rate",
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
    "high_to_low_count",
    "high_to_low_rate",
    "label2_recall",
    "label5_recall",
    "mean_pred_low",
]


def safe_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = clean_text(value).lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def parse_output(raw_output: str, prompt_config: str, allow_regex_score_fallback: bool = False) -> dict[str, Any]:
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
        if score is not None:
            return {
                "parse_success": True,
                "pred_label": score,
                "parse_method": "json",
                "score_cap": score_cap,
                "confidence": confidence,
                "major_failures": failures,
                "rubric_clause": clean_text(data.get("rubric_clause")),
                "failure_evidence": clean_text(data.get("failure_evidence")),
                "rubric_satisfied": safe_bool(data.get("rubric_satisfied")),
                "brief_reason": clean_text(data.get("brief_reason")),
            }
        return {
            "parse_success": False,
            "pred_label": None,
            "parse_method": "json_invalid_score",
            "score_cap": score_cap,
            "confidence": confidence,
            "major_failures": failures,
            "rubric_clause": clean_text(data.get("rubric_clause")),
            "failure_evidence": clean_text(data.get("failure_evidence")),
            "rubric_satisfied": safe_bool(data.get("rubric_satisfied")),
            "brief_reason": clean_text(data.get("brief_reason")),
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
                "failure_evidence": "",
                "rubric_satisfied": None,
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
        "failure_evidence": "",
        "rubric_satisfied": None,
        "brief_reason": "",
    }


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


def prompt_messages(row: dict[str, Any], prompt_config: str) -> list[dict[str, str]]:
    if prompt_config == "R0C_0_exact_r0a_reproduction":
        return r0a_prompt_messages(row)
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
    if task_index % 13 == 12:
        return "The score is probably high."
    score = max(1, min(5, gold + ((task_index % 3) - 1)))
    if prompt_config == "R0C_0_exact_r0a_reproduction":
        return json.dumps({"score": score}, ensure_ascii=False)
    failures = ["no_major_failure"] if score >= 4 else ["missing_key_point"]
    score_cap = None if score >= 4 else min(3, score)
    if prompt_config == "R0C_4_high_score_protection" and gold >= 4:
        score = max(score, 4)
        score_cap = None
        failures = ["no_major_failure"]
    return json.dumps(
        {
            "rubric_clause": "Relevant rubric clause",
            "major_failures": failures,
            "failure_evidence": "" if failures == ["no_major_failure"] else "Dry-run concrete failure evidence.",
            "rubric_satisfied": failures == ["no_major_failure"],
            "score_cap": score_cap,
            "score": score,
            "brief_reason": "Dry-run synthetic reason.",
            "confidence": 0.72,
        },
        ensure_ascii=False,
    )


def prediction_row(
    row: dict[str, Any],
    prompt_config: str,
    raw_output: str,
    model_name: str,
    backend: str,
    allow_regex_score_fallback: bool,
) -> dict[str, Any]:
    parsed = parse_output(raw_output, prompt_config, allow_regex_score_fallback)
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
        "rubric_satisfied": parsed["rubric_satisfied"],
        "rubric_clause_truncated": clean_text(parsed["rubric_clause"])[:300],
        "failure_evidence_truncated": clean_text(parsed["failure_evidence"])[:500],
        "brief_reason_truncated": clean_text(parsed["brief_reason"])[:500],
        "raw_output_truncated": clean_text(raw_output)[:800],
        "model_name_or_path": model_name,
        "backend": backend,
        "metric": row_metric(row),
        "language": clean_text(row.get("language")),
        "subject": clean_text(row.get("subject_canonical") or row.get("subject_raw")),
    }


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


def format_duration(seconds: float) -> str:
    if math.isnan(seconds) or seconds < 0:
        return "NA"
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def progress_message(prefix: str, current: int, total: int, started_at: float) -> str:
    elapsed = max(time.monotonic() - started_at, 1e-9)
    rate = current / elapsed if current > 0 else 0.0
    remaining = max(total - current, 0)
    eta = remaining / rate if rate > 0 else float("nan")
    percent = safe_rate(current * 100.0, total)
    return (
        f"[exp19-r0c] {prefix} generated {current}/{total} "
        f"({fmt(percent)}%) speed={rate:.2f}/s elapsed={format_duration(elapsed)} eta={format_duration(eta)}"
    )


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
    started_at = time.monotonic()
    total_pending = len(pending)
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
        print(progress_message("vllm", min(start + len(batch), total_pending), total_pending, started_at), flush=True)
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
    started_at = time.monotonic()
    total_pending = len(pending)
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
            progress_message("transformers", min(start + len(batch), total_pending), total_pending, started_at),
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
    high = [row for row in rows if int(row["gold_label"]) >= 4]
    low_valid = [int(row["pred_label"]) for row in low if safe_int(row.get("pred_label")) is not None]
    high_valid = [int(row["pred_label"]) for row in high if safe_int(row.get("pred_label")) is not None]
    label1 = [row for row in rows if int(row["gold_label"]) == 1]
    label2 = [row for row in rows if int(row["gold_label"]) == 2]
    label4 = [row for row in rows if int(row["gold_label"]) == 4]
    label5 = [row for row in rows if int(row["gold_label"]) == 5]
    low_to_high = sum(1 for row in low if safe_int(row.get("pred_label")) is not None and int(row["pred_label"]) >= 4)
    high_to_low = sum(1 for row in high if safe_int(row.get("pred_label")) is not None and int(row["pred_label"]) <= 2)
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
        "high_to_low_count": high_to_low,
        "high_to_low_rate": safe_rate(high_to_low, len(high)),
        "label1_recall": safe_rate(sum(1 for row in label1 if safe_int(row.get("pred_label")) == 1), len(label1)),
        "label2_recall": safe_rate(sum(1 for row in label2 if safe_int(row.get("pred_label")) == 2), len(label2)),
        "label4_recall": safe_rate(sum(1 for row in label4 if safe_int(row.get("pred_label")) == 4), len(label4)),
        "label5_recall": safe_rate(sum(1 for row in label5 if safe_int(row.get("pred_label")) == 5), len(label5)),
        "label2_pred_ge4_rate": safe_rate(label2_ge4, len(label2)),
        "mean_pred_low": mean(low_valid),
        "median_pred_low": float(median(low_valid)) if low_valid else float("nan"),
        "mean_pred_high": mean(high_valid),
        "median_pred_high": float(median(high_valid)) if high_valid else float("nan"),
    }


def per_label_rows(rows: list[dict[str, Any]], prompt_config: str) -> list[dict[str, Any]]:
    out = []
    for label in LABELS:
        items = [row for row in rows if int(row["gold_label"]) == label]
        preds = [safe_int(row.get("pred_label")) for row in items]
        valid = [int(value) for value in preds if value is not None]
        item = {
            "prompt_config": prompt_config,
            "gold_label": label,
            "n": len(items),
            "exact_accuracy": safe_rate(sum(1 for value in preds if value == label), len(items)),
            "mean_pred": mean(valid),
        }
        for pred_label in LABELS:
            item[f"pred_{pred_label}_rate"] = safe_rate(sum(1 for value in preds if value == pred_label), len(items))
        out.append(item)
    return out


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
                "high_to_low_count": summary["high_to_low_count"],
                "high_to_low_rate": summary["high_to_low_rate"],
                "label2_recall": summary["label2_recall"],
                "label5_recall": summary["label5_recall"],
                "mean_pred_low": summary["mean_pred_low"],
            }
        )
    return out


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


def load_dev_row(path: Path, table_name: str, prompt_column: str | None = None, prompt_value: str | None = None) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("split", SPLIT_NAME) != SPLIT_NAME:
                continue
            if prompt_column and prompt_value and clean_text(row.get(prompt_column)) != prompt_value:
                continue
            return {key: safe_float(value) for key, value in row.items() if key not in {"evaluator", "split", "prompt_config"}}
    return {}


def load_r0a_dev_baseline(r0a_dir: Path) -> dict[str, float]:
    return load_dev_row(Path(r0a_dir) / "tables" / "qwen3_4b_direct_overall_metrics.csv", "r0a")


def load_r0b_best(r0b_dir: Path) -> dict[str, float]:
    decision_path = Path(r0b_dir) / "decision" / "exp19_r0b_decision.json"
    prompt = "R0B_3_conservative_low_score_cap"
    if decision_path.exists():
        try:
            prompt = json.loads(decision_path.read_text(encoding="utf-8")).get("best_prompt_config") or prompt
        except Exception:
            pass
    return load_dev_row(
        Path(r0b_dir) / "tables" / "exp19_r0b_prompt_metrics.csv",
        "r0b",
        prompt_column="prompt_config",
        prompt_value=prompt,
    )


def calibrated_success(metric: dict[str, Any], d1_row: dict[str, Any]) -> bool:
    bias = safe_float(metric.get("Signed_Bias"))
    return bool(
        safe_float(metric.get("parse_success_rate")) >= 0.95
        and safe_float(metric.get("low_to_high_rate")) <= 0.40
        and safe_float(metric.get("label2_recall")) >= 0.40
        and safe_float(d1_row.get("pred_ge4_rate_d1_hidden")) <= 0.40
        and safe_float(metric.get("MAE")) <= 1.0
        and safe_float(metric.get("QWK")) >= 0.10
        and -0.75 <= bias <= 0.25
        and safe_float(metric.get("high_to_low_rate")) <= 0.30
        and safe_float(metric.get("label5_recall")) >= 0.50
    )


def finite(value: Any, default: float = -1e9) -> float:
    out = safe_float(value)
    return default if math.isnan(out) else out


def select_best_prompt(metric_rows: list[dict[str, Any]], d1_by_prompt: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in metric_rows if row.get("prompt_config") in CALIBRATED_PROMPTS]
    scored = []
    for row in candidates:
        d1_row = d1_by_prompt.get(row["prompt_config"], {})
        success = calibrated_success(row, d1_row)
        scored.append(
            (
                1 if success else 0,
                finite(row.get("parse_success_rate")),
                -finite(row.get("low_to_high_rate"), 1e9),
                finite(row.get("label2_recall")),
                -finite(row.get("high_to_low_rate"), 1e9),
                finite(row.get("label5_recall")),
                -finite(row.get("MAE"), 1e9),
                finite(row.get("QWK")),
                -abs(finite(row.get("Signed_Bias"), 1e9)),
                row["prompt_config"],
                row,
            )
        )
    scored.sort(reverse=True)
    return scored[0][-1] if scored else {}


def decision_json(
    metric_rows: list[dict[str, Any]],
    d1_rows: list[dict[str, Any]],
    r0a_dev: dict[str, float],
) -> dict[str, Any]:
    d1_by_prompt = {row["prompt_config"]: row for row in d1_rows}
    best = select_best_prompt(metric_rows, d1_by_prompt)
    best_name = clean_text(best.get("prompt_config"))
    best_d1 = d1_by_prompt.get(best_name, {})
    success = calibrated_success(best, best_d1) if best else False
    r0a_l2h = r0a_dev.get("low_to_high_rate", float("nan"))
    best_l2h = safe_float(best.get("low_to_high_rate"))
    improves_l2h = (not math.isnan(r0a_l2h)) and (r0a_l2h - best_l2h >= 0.20)
    proceed_to_lora = bool(improves_l2h and not success)
    reason = "Calibrated prompt passes all risk, calibration, and high-score protection gates." if success else (
        "Prompt improves low-to-high meaningfully, but calibration or high-score protection remains weak."
        if proceed_to_lora
        else "No calibrated prompt passes the risk and calibration gates."
    )
    return {
        "best_prompt_config": best_name,
        "parse_success_best": json_metric(best.get("parse_success_rate")),
        "dev_low_to_high_best": json_metric(best.get("low_to_high_rate")),
        "dev_high_to_low_best": json_metric(best.get("high_to_low_rate")),
        "dev_label2_recall_best": json_metric(best.get("label2_recall")),
        "dev_label5_recall_best": json_metric(best.get("label5_recall")),
        "dev_MAE_best": json_metric(best.get("MAE")),
        "dev_QWK_best": json_metric(best.get("QWK")),
        "dev_signed_bias_best": json_metric(best.get("Signed_Bias")),
        "d1_hidden_pred_ge4_best": json_metric(best_d1.get("pred_ge4_rate_d1_hidden")),
        "calibrated_prompt_success": success,
        "lock_prompt_for_test_all": success,
        "proceed_to_lora": proceed_to_lora,
        "recommended_lora_variant": "score + failure evidence + score_cap LoRA/SFT" if proceed_to_lora else "",
        "reason": reason,
    }


def exact_reproduction_summary(metric_rows: list[dict[str, Any]], r0a_dev: dict[str, float]) -> dict[str, float]:
    exact = next((row for row in metric_rows if row.get("prompt_config") == "R0C_0_exact_r0a_reproduction"), {})
    return {
        "mae_delta": safe_float(exact.get("MAE")) - r0a_dev.get("MAE", float("nan")),
        "low_to_high_delta": safe_float(exact.get("low_to_high_rate")) - r0a_dev.get("low_to_high_rate", float("nan")),
        "label2_recall_delta": safe_float(exact.get("label2_recall")) - r0a_dev.get("label2_recall", float("nan")),
    }


def write_report(
    args: argparse.Namespace,
    metric_rows: list[dict[str, Any]],
    d1_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    decision: dict[str, Any],
    r0a_dev: dict[str, float],
    r0b_best: dict[str, float],
    backend_used: str,
) -> None:
    metric_by_prompt = {row["prompt_config"]: row for row in metric_rows}
    d1_by_prompt = {row["prompt_config"]: row for row in d1_rows}
    failure_by_prompt = {row["prompt_config"]: row for row in failure_rows}
    exact_delta = exact_reproduction_summary(metric_rows, r0a_dev)
    best = clean_text(decision.get("best_prompt_config"))
    lines = [
        "# Exp19-R0C Calibrated Failure-First Prompt Audit",
        "",
        "## Setup",
        "",
        f"- Dataset: `{relpath(args.data_path)}`.",
        f"- R0A baseline dir: `{relpath(args.r0a_dir)}`.",
        f"- R0B baseline dir: `{relpath(args.r0b_dir)}`.",
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
        "| prompt | n | parse | MAE | QWK | bias | low-to-high | high-to-low | label2 recall | label5 recall | D1 hidden pred>=4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for prompt_config in PROMPT_CONFIGS:
        row = metric_by_prompt.get(prompt_config, {})
        d1 = d1_by_prompt.get(prompt_config, {})
        lines.append(
            f"| `{prompt_config}` | {row.get('n', 0)} | {fmt(row.get('parse_success_rate'))} | "
            f"{fmt(row.get('MAE'))} | {fmt(row.get('QWK'))} | {fmt(row.get('Signed_Bias'))} | "
            f"{fmt(row.get('low_to_high_rate'))} | {fmt(row.get('high_to_low_rate'))} | "
            f"{fmt(row.get('label2_recall'))} | {fmt(row.get('label5_recall'))} | "
            f"{fmt(d1.get('pred_ge4_rate_d1_hidden'))} |"
        )
    best_row = metric_by_prompt.get(best, {})
    best_d1 = d1_by_prompt.get(best, {})
    best_failure = failure_by_prompt.get(best, {})
    lines.extend(
        [
            "",
            "## Baseline References",
            "",
            f"- R0A dev: MAE `{fmt(r0a_dev.get('MAE'))}`, QWK `{fmt(r0a_dev.get('QWK'))}`, low-to-high `{fmt(r0a_dev.get('low_to_high_rate'))}`, label2 recall `{fmt(r0a_dev.get('label2_recall'))}`.",
            f"- R0B best dev: MAE `{fmt(r0b_best.get('MAE'))}`, QWK `{fmt(r0b_best.get('QWK'))}`, low-to-high `{fmt(r0b_best.get('low_to_high_rate'))}`, label2 recall `{fmt(r0b_best.get('label2_recall'))}`.",
            "",
            "## Answers",
            "",
            f"- Does exact R0A reproduction match R0A dev metrics? MAE delta `{fmt(exact_delta['mae_delta'])}`, low-to-high delta `{fmt(exact_delta['low_to_high_delta'])}`, label2 recall delta `{fmt(exact_delta['label2_recall_delta'])}`.",
            f"- Which calibrated prompt best balances low-to-high and MAE/QWK? `{best}`.",
            f"- Does the prompt avoid R0B over-conservatism? Best bias `{fmt(best_row.get('Signed_Bias'))}`, high-to-low `{fmt(best_row.get('high_to_low_rate'))}`, MAE `{fmt(best_row.get('MAE'))}`.",
            f"- Does high_to_low remain acceptable? `{fmt(best_row.get('high_to_low_rate'))}`.",
            f"- Does label5 recall remain acceptable? `{fmt(best_row.get('label5_recall'))}`.",
            f"- Does D1 hidden pred>=4 stay low? `{fmt(best_d1.get('pred_ge4_rate_d1_hidden'))}`.",
            f"- Does failure type accuracy improve? Best failure micro-F1 `{fmt(best_failure.get('failure_type_micro_f1'))}`.",
            f"- Should lock prompt and run held-out test/all_5536? `{decision.get('lock_prompt_for_test_all')}`.",
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
    write_text(Path(args.out_dir) / "reports" / "exp19_r0c_calibrated_prompt_audit_report.md", "\n".join(lines))


def write_all_outputs(args: argparse.Namespace, predictions: list[dict[str, Any]], backend_used: str) -> None:
    out_dir = Path(args.out_dir)
    tables_dir = out_dir / "tables"
    decision_dir = out_dir / "decision"
    groups = prediction_groups(predictions)
    metric_rows = [prompt_metric_summary(groups.get(prompt_config, []), prompt_config) for prompt_config in PROMPT_CONFIGS]
    d1_rows = d1_eval_rows(predictions, Path(args.d1_dir)) if resolve_d1_base(Path(args.d1_dir)).exists() else []
    failure_rows = failure_type_eval_rows(predictions, Path(args.d1_dir)) if resolve_d1_base(Path(args.d1_dir)).exists() else []
    per_label = [row for prompt_config in PROMPT_CONFIGS for row in per_label_rows(groups.get(prompt_config, []), prompt_config)]
    r0a_dev = load_r0a_dev_baseline(Path(args.r0a_dir))
    r0b_best = load_r0b_best(Path(args.r0b_dir))
    decision = decision_json(metric_rows, d1_rows, r0a_dev)

    write_csv(tables_dir / "exp19_r0c_prompt_metrics.csv", metric_rows, PROMPT_METRIC_FIELDS)
    write_csv(tables_dir / "exp19_r0c_d1_hidden_eval.csv", d1_rows, D1_FIELDS)
    write_csv(tables_dir / "exp19_r0c_failure_type_eval.csv", failure_rows, FAILURE_TYPE_FIELDS)
    write_csv(tables_dir / "exp19_r0c_per_label_accuracy.csv", per_label, PER_LABEL_FIELDS)
    write_csv(tables_dir / "exp19_r0c_by_metric.csv", grouped_by_metric(predictions), BY_METRIC_FIELDS)
    write_json(decision_dir / "exp19_r0c_decision.json", decision)
    write_report(args, metric_rows, d1_rows, failure_rows, decision, r0a_dev, r0b_best, backend_used)


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
    parser = argparse.ArgumentParser(description="Run Exp19-R0C calibrated failure-first prompt audit.")
    parser.add_argument("--model_name_or_path", default="/home/jpang/models/modelscope/Qwen/Qwen3-4B")
    parser.add_argument("--backend", choices=["transformers", "vllm"], default="vllm")
    parser.add_argument("--data_path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--r0a_dir", type=Path, default=DEFAULT_R0A_DIR)
    parser.add_argument("--r0b_dir", type=Path, default=DEFAULT_R0B_DIR)
    parser.add_argument("--d1_dir", type=Path, default=DEFAULT_D1_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prompt_configs", default=" ".join(PROMPT_CONFIGS))
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=224)
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
            print(f"[exp19-r0c] using existing internal predictions: {len(predictions)}", flush=True)
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
