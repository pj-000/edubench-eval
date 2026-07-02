"""Exp19-R0A Qwen3-4B direct scoring baseline.

This script runs a frozen instruction model as a direct 1-5 educational
assessment judge. It does not train, does not use human rationales in prompts,
and writes full per-sample predictions only to an internal ignored jsonl file.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.src.edujudge.utils.io import read_jsonl, relpath, write_csv, write_json, write_jsonl, write_text


DEFAULT_DATA_PATH = Path("thesis_exp/data/processed/edubench_scoring_all.jsonl")
DEFAULT_SPLIT_DIR = Path("thesis_exp/data/splits/question_seed42")
DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_r0a_qwen4b_direct_baseline_seed42")
DEFAULT_D1_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/d1_hidden_failure_audit_seed42_dev")
EVALUATOR = "Qwen3-4B-Instruct-direct"
SPLITS = ["all_5536", "train", "dev", "test"]
LABELS = [1, 2, 3, 4, 5]
SYSTEM_MESSAGE = (
    "You are an educational assessment evaluator. You assign human-aligned scores from 1 to 5 "
    "based on the question, answer, evaluation metric, and rubric."
)
USER_TEMPLATE = """Evaluate the following answer.

Question:
{question}

Answer:
{answer}

Evaluation metric:
{metric}

Rubric:
{rubric}

Metadata:
{metadata}

Return only valid JSON:
{{"score": <integer from 1 to 5>}}"""

OVERALL_FIELDS = [
    "evaluator",
    "split",
    "n",
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
    "parse_success_rate",
    "invalid_score_rate",
]
LOW_BIAS_FIELDS = [
    "evaluator",
    "split",
    "n_low",
    "mean_pred_when_human_1_2",
    "median_pred_when_human_1_2",
    "pred_ge4_count_when_human_1_2",
    "pred_ge4_rate_when_human_1_2",
    "pred_5_rate_when_human_1_2",
    "label1_recall",
    "label2_recall",
    "severity_label",
]
PER_LABEL_FIELDS = [
    "evaluator",
    "split",
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
    "metric",
    "split",
    "n",
    "MAE",
    "QWK",
    "Signed_Bias",
    "Exact_Match",
    "low_to_high_rate",
    "label2_recall",
    "mean_pred_low",
]
BY_LANGUAGE_FIELDS = [
    "language",
    "split",
    "n",
    "MAE",
    "QWK",
    "Signed_Bias",
    "Exact_Match",
    "low_to_high_rate",
    "label2_recall",
    "parse_success_rate",
]
D1_FIELDS = [
    "evaluator",
    "n_d1_cases",
    "mean_pred_d1_hidden",
    "pred_ge4_rate_d1_hidden",
    "pred_5_rate_d1_hidden",
    "label2_recall_d1",
    "mean_pred_matched_controls",
    "hidden_control_score_gap",
]


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        out = int(float(value))
        return out if out in LABELS else None
    except Exception:
        return None


def safe_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def safe_rate(num: int | float, den: int | float) -> float:
    return float(num) / float(den) if den else float("nan")


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if not math.isnan(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def quadratic_weighted_kappa(gold: list[int], pred: list[int]) -> float:
    if not gold:
        return float("nan")
    observed = np.zeros((5, 5), dtype=float)
    index = {label: idx for idx, label in enumerate(LABELS)}
    for g, p in zip(gold, pred):
        if g in index and p in index:
            observed[index[g], index[p]] += 1.0
    hist_g = observed.sum(axis=1)
    hist_p = observed.sum(axis=0)
    expected = np.outer(hist_g, hist_p) / max(float(observed.sum()), 1.0)
    weights = np.zeros((5, 5), dtype=float)
    for i in range(5):
        for j in range(5):
            weights[i, j] = ((i - j) ** 2) / 16.0
    numerator = float((weights * observed).sum())
    denominator = float((weights * expected).sum())
    if denominator == 0:
        return 1.0 if numerator == 0 else float("nan")
    return 1.0 - numerator / denominator


def kendall_tau_b(gold: list[int], pred: list[int]) -> float:
    if len(gold) < 2:
        return float("nan")
    table = np.zeros((5, 5), dtype=int)
    idx = {label: i for i, label in enumerate(LABELS)}
    for g, p in zip(gold, pred):
        if g in idx and p in idx:
            table[idx[g], idx[p]] += 1
    concordant = 0
    discordant = 0
    for i in range(5):
        for j in range(5):
            n = int(table[i, j])
            if n == 0:
                continue
            concordant += n * int(table[i + 1 :, j + 1 :].sum())
            discordant += n * int(table[i + 1 :, :j].sum())
    tied_gold = sum(int(n) * (int(n) - 1) // 2 for n in table.sum(axis=1))
    tied_pred = sum(int(n) * (int(n) - 1) // 2 for n in table.sum(axis=0))
    tied_both = sum(int(n) * (int(n) - 1) // 2 for n in table.reshape(-1))
    only_gold = tied_gold - tied_both
    only_pred = tied_pred - tied_both
    denom = math.sqrt((concordant + discordant + only_gold) * (concordant + discordant + only_pred))
    return float((concordant - discordant) / denom) if denom else float("nan")


def parse_score(raw_output: str) -> dict[str, Any]:
    text = clean_text(raw_output)
    candidates = [text]
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if match:
        candidates.insert(0, match.group(0))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except Exception:
            continue
        if isinstance(data, dict):
            score = safe_int(data.get("score"))
            if score is not None:
                return {"parse_success": True, "pred_label": score, "parse_method": "json"}
    regex = re.search(r"(?<![\d.])([1-5])(?![\d.])", text)
    if regex:
        return {"parse_success": True, "pred_label": int(regex.group(1)), "parse_method": "regex_fallback"}
    return {"parse_success": False, "pred_label": None, "parse_method": "failed"}


def sample_id(row: dict[str, Any]) -> str:
    return clean_text(row.get("record_id") or row.get("sample_id") or row.get("id"))


def gold_label(row: dict[str, Any]) -> int:
    value = safe_int(row.get("label_5") or row.get("label") or row.get("gold_label"))
    if value is None:
        raise ValueError(f"Missing label_5 for sample {sample_id(row)}")
    return value


def split_mapping(split_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for split in ["train", "dev", "test"]:
        path = split_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        for row in read_jsonl(path):
            sid = sample_id(row)
            if sid:
                out[sid] = split
    return out


def load_dataset(data_path: Path, split_dir: Path, max_examples: int = 0) -> list[dict[str, Any]]:
    split_by_id = split_mapping(split_dir)
    rows = read_jsonl(data_path)
    if max_examples > 0:
        rows = rows[:max_examples]
    out = []
    for idx, row in enumerate(rows):
        sid = sample_id(row) or f"row_{idx}"
        row = dict(row)
        row["sample_id"] = sid
        row["gold_label"] = gold_label(row)
        row["split"] = split_by_id.get(sid, "unknown")
        out.append(row)
    return out


def prompt_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    metadata = {
        "language": row.get("language", ""),
        "subject": row.get("subject_canonical") or row.get("subject_raw") or "",
        "education_level": row.get("education_level_canonical") or row.get("education_level_raw") or "",
        "scenario": row.get("scenario_canonical") or row.get("scenario_raw") or "",
    }
    user = USER_TEMPLATE.format(
        question=clean_text(row.get("question")),
        answer=clean_text(row.get("answer")),
        metric=clean_text(row.get("metric_canonical") or row.get("metric_raw") or row.get("metric_abbr")),
        rubric=clean_text(row.get("rubric")),
        metadata=json.dumps(metadata, ensure_ascii=False, sort_keys=True),
    )
    return [{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": user}]


def messages_to_prompt(tokenizer: Any, messages: list[dict[str, str]], enable_thinking: bool = False) -> str:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=bool(enable_thinking),
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return (
        f"System:\n{messages[0]['content']}\n\n"
        f"User:\n{messages[1]['content']}\n\n"
        "Assistant:\n"
    )


def fake_output(index: int, gold: int) -> str:
    if index % 5 == 4:
        return "The score should be high."
    if index % 5 == 3:
        return f"score: {max(1, min(5, gold + 1))}"
    return json.dumps({"score": max(1, min(5, gold + (index % 3) - 1))})


def load_existing_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = clean_text(row.get("sample_id"))
            if sid:
                out[sid] = row
    return out


def infer_backend_from_predictions(predictions: Iterable[dict[str, Any]], fallback: str) -> str:
    seen = {clean_text(row.get("backend")) for row in predictions if clean_text(row.get("backend"))}
    if not seen:
        return fallback
    if len(seen) == 1:
        return next(iter(seen))
    return "+".join(sorted(seen))


def append_prediction(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prediction_row(row: dict[str, Any], raw_output: str, model_name: str, backend: str) -> dict[str, Any]:
    parsed = parse_score(raw_output)
    pred = parsed["pred_label"]
    return {
        "sample_id": row["sample_id"],
        "split": row.get("split", "unknown"),
        "gold_label": row["gold_label"],
        "pred_label": pred,
        "parse_success": bool(parsed["parse_success"]),
        "parse_method": parsed["parse_method"],
        "raw_output_truncated": clean_text(raw_output)[:512],
        "model_name_or_path": model_name,
        "backend": backend,
        "metric": clean_text(row.get("metric_canonical") or row.get("metric_raw") or row.get("metric_abbr")),
        "language": clean_text(row.get("language")),
        "subject": clean_text(row.get("subject_canonical") or row.get("subject_raw")),
    }


def run_dry_predictions(rows: list[dict[str, Any]], internal_path: Path, model_name: str) -> list[dict[str, Any]]:
    if internal_path.exists():
        internal_path.unlink()
    predictions = []
    for idx, row in enumerate(rows):
        pred_row = prediction_row(row, fake_output(idx, int(row["gold_label"])), model_name, "dry_run")
        append_prediction(internal_path, pred_row)
        predictions.append(pred_row)
    return predictions


def run_vllm_predictions(args: argparse.Namespace, rows: list[dict[str, Any]], internal_path: Path, done: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
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
    pending = [row for row in rows if row["sample_id"] not in done]
    for start in range(0, len(pending), int(args.batch_size)):
        batch = pending[start : start + int(args.batch_size)]
        prompts = [messages_to_prompt(tokenizer, prompt_messages(row), bool(args.enable_thinking)) for row in batch]
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        for row, output in zip(batch, outputs):
            text = output.outputs[0].text if output.outputs else ""
            pred_row = prediction_row(row, text, args.model_name_or_path, "vllm")
            append_prediction(internal_path, pred_row)
            predictions.append(pred_row)
        print(f"[exp19-r0a] vllm generated {min(start + len(batch), len(pending))}/{len(pending)} pending", flush=True)
    return predictions


def run_transformers_predictions(args: argparse.Namespace, rows: list[dict[str, Any]], internal_path: Path, done: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if args.dtype in {"auto", "bfloat16", "bf16"} else torch.float16 if args.dtype in {"float16", "fp16"} else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    predictions = list(done.values())
    pending = [row for row in rows if row["sample_id"] not in done]
    for start in range(0, len(pending), int(args.batch_size)):
        batch = pending[start : start + int(args.batch_size)]
        prompts = [messages_to_prompt(tokenizer, prompt_messages(row), bool(args.enable_thinking)) for row in batch]
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
        for row, text in zip(batch, texts):
            pred_row = prediction_row(row, text, args.model_name_or_path, "transformers")
            append_prediction(internal_path, pred_row)
            predictions.append(pred_row)
        print(f"[exp19-r0a] transformers generated {min(start + len(batch), len(pending))}/{len(pending)} pending", flush=True)
    return predictions


def split_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    if split == "all_5536":
        return rows
    return [row for row in rows if row.get("split") == split]


def metric_summary(rows: list[dict[str, Any]], evaluator: str, split: str) -> dict[str, Any]:
    n = len(rows)
    valid = [row for row in rows if row.get("pred_label") is not None and bool(row.get("parse_success"))]
    gold = [int(row["gold_label"]) for row in valid]
    pred = [int(row["pred_label"]) for row in valid]
    low = [row for row in rows if int(row["gold_label"]) <= 2]
    label1 = [row for row in rows if int(row["gold_label"]) == 1]
    label2 = [row for row in rows if int(row["gold_label"]) == 2]
    low_to_high = sum(1 for row in low if safe_int(row.get("pred_label")) is not None and int(row["pred_label"]) >= 4)
    label1_hits = sum(1 for row in label1 if safe_int(row.get("pred_label")) == 1)
    label2_hits = sum(1 for row in label2 if safe_int(row.get("pred_label")) == 2)
    label2_ge4 = sum(1 for row in label2 if safe_int(row.get("pred_label")) is not None and int(row["pred_label"]) >= 4)
    exact = sum(1 for row in rows if safe_int(row.get("pred_label")) == int(row["gold_label"]))
    return {
        "evaluator": evaluator,
        "split": split,
        "n": n,
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
        "parse_success_rate": safe_rate(len(valid), n),
        "invalid_score_rate": safe_rate(n - len(valid), n),
    }


def low_bias_row(rows: list[dict[str, Any]], evaluator: str, split: str) -> dict[str, Any]:
    low = [row for row in rows if int(row["gold_label"]) <= 2]
    preds = [int(row["pred_label"]) for row in low if safe_int(row.get("pred_label")) is not None]
    ge4 = sum(1 for value in preds if value >= 4)
    p5 = sum(1 for value in preds if value == 5)
    label1 = [row for row in rows if int(row["gold_label"]) == 1]
    label2 = [row for row in rows if int(row["gold_label"]) == 2]
    rate = safe_rate(ge4, len(low))
    severity = "low"
    if not math.isnan(rate) and rate >= 0.5:
        severity = "severe"
    elif not math.isnan(rate) and rate >= 0.25:
        severity = "moderate"
    return {
        "evaluator": evaluator,
        "split": split,
        "n_low": len(low),
        "mean_pred_when_human_1_2": mean(preds),
        "median_pred_when_human_1_2": float(median(preds)) if preds else float("nan"),
        "pred_ge4_count_when_human_1_2": ge4,
        "pred_ge4_rate_when_human_1_2": rate,
        "pred_5_rate_when_human_1_2": safe_rate(p5, len(low)),
        "label1_recall": safe_rate(sum(1 for row in label1 if safe_int(row.get("pred_label")) == 1), len(label1)),
        "label2_recall": safe_rate(sum(1 for row in label2 if safe_int(row.get("pred_label")) == 2), len(label2)),
        "severity_label": severity,
    }


def per_label_rows(rows: list[dict[str, Any]], evaluator: str, split: str) -> list[dict[str, Any]]:
    out = []
    for label in LABELS:
        items = [row for row in rows if int(row["gold_label"]) == label]
        preds = [safe_int(row.get("pred_label")) for row in items]
        valid = [int(value) for value in preds if value is not None]
        row = {
            "evaluator": evaluator,
            "split": split,
            "gold_label": label,
            "n": len(items),
            "exact_accuracy": safe_rate(sum(1 for value in preds if value == label), len(items)),
            "mean_pred": mean(valid),
        }
        for pred_label in LABELS:
            row[f"pred_{pred_label}_rate"] = safe_rate(sum(1 for value in preds if value == pred_label), len(items))
        out.append(row)
    return out


def confusion_rows(rows: list[dict[str, Any]], evaluator: str, split: str, normalized: bool) -> list[dict[str, Any]]:
    out = []
    for label in LABELS:
        items = [row for row in rows if int(row["gold_label"]) == label]
        row = {"evaluator": evaluator, "split": split, "gold_label": label, "n": len(items)}
        for pred_label in LABELS:
            count = sum(1 for item in items if safe_int(item.get("pred_label")) == pred_label)
            row[f"pred_{pred_label}"] = safe_rate(count, len(items)) if normalized else count
        out.append(row)
    return out


def grouped_rows(rows: list[dict[str, Any]], group_key: str, split: str, fieldnames: list[str]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in split_rows(rows, split):
        groups[clean_text(row.get(group_key)) or "unknown"].append(row)
    out = []
    for key, items in sorted(groups.items()):
        summary = metric_summary(items, EVALUATOR, split)
        low = [row for row in items if int(row["gold_label"]) <= 2 and safe_int(row.get("pred_label")) is not None]
        base = {
            group_key: key,
            "split": split,
            "n": len(items),
            "MAE": summary["MAE"],
            "QWK": summary["QWK"],
            "Signed_Bias": summary["Signed_Bias"],
            "Exact_Match": summary["Exact_Match"],
            "low_to_high_rate": summary["low_to_high_rate"],
            "label2_recall": summary["label2_recall"],
            "parse_success_rate": summary["parse_success_rate"],
            "mean_pred_low": mean([int(row["pred_label"]) for row in low]),
        }
        out.append({key_name: base.get(key_name, "") for key_name in fieldnames})
    return out


def prediction_groups(predictions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {split: split_rows(predictions, split) for split in SPLITS}


def existing_judge_predictions(data_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    expected = ["EduBenchEvaluator", "deepseek-v3", "deepseek-r1", "qwq-plus", "gpt-4o"]
    rows = []
    seen = set()
    for judge in expected:
        coverage = 0
        for row in data_rows:
            scores = row.get("judge_scores") or {}
            if isinstance(scores, str):
                try:
                    scores = json.loads(scores)
                except Exception:
                    scores = {}
            if not isinstance(scores, dict) or judge not in scores:
                continue
            pred = safe_int(scores.get(judge))
            if pred is None:
                continue
            coverage += 1
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "split": row.get("split", "unknown"),
                    "gold_label": row["gold_label"],
                    "pred_label": pred,
                    "parse_success": True,
                    "metric": clean_text(row.get("metric_canonical") or row.get("metric_raw") or row.get("metric_abbr")),
                    "language": clean_text(row.get("language")),
                    "evaluator": judge,
                }
            )
        if coverage:
            seen.add(judge)
    missing = [judge for judge in expected if judge not in seen]
    return rows, missing


def compare_existing_judges(data_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    judge_rows, missing = existing_judge_predictions(data_rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in judge_rows:
        grouped[str(row["evaluator"])].append(row)
    out = []
    for evaluator, rows in sorted(grouped.items()):
        for split in SPLITS:
            out.append(metric_summary(split_rows(rows, split), evaluator, split))
    return out, missing


def load_d1_case_ids(d1_dir: Path) -> tuple[set[str], set[str], list[tuple[str, str]]]:
    annotation = d1_dir / "d1_hidden_failure_annotation_template_filled_human_rationale_recovered.csv"
    case_ids: set[str] = set()
    if annotation.exists():
        with annotation.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                hidden = clean_text(row.get("is_hidden_failure_manual")).lower()
                if hidden in {"1", "true", "yes", "y"} or hidden == "":
                    sid = clean_text(row.get("sample_id"))
                    if sid:
                        case_ids.add(sid)
    controls: set[str] = set()
    pairs: list[tuple[str, str]] = []
    pair_path = d1_dir / "d1_matched_case_control_review.csv"
    if pair_path.exists():
        with pair_path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                case_id = clean_text(row.get("case_sample_id"))
                control_id = clean_text(row.get("control_sample_id"))
                if case_id and control_id:
                    pairs.append((case_id, control_id))
                    controls.add(control_id)
    return case_ids, controls, pairs


def d1_eval_rows(predictions: list[dict[str, Any]], d1_dir: Path) -> list[dict[str, Any]]:
    case_ids, control_ids, pairs = load_d1_case_ids(d1_dir)
    pred_by_id = {row["sample_id"]: row for row in predictions}
    cases = [pred_by_id[sid] for sid in sorted(case_ids) if sid in pred_by_id]
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
    return [
        {
            "evaluator": EVALUATOR,
            "n_d1_cases": len(cases),
            "mean_pred_d1_hidden": mean(case_preds),
            "pred_ge4_rate_d1_hidden": safe_rate(sum(1 for value in case_preds if value >= 4), len(cases)),
            "pred_5_rate_d1_hidden": safe_rate(sum(1 for value in case_preds if value == 5), len(cases)),
            "label2_recall_d1": safe_rate(sum(1 for row in label2 if safe_int(row.get("pred_label")) == 2), len(label2)),
            "mean_pred_matched_controls": mean(control_preds),
            "hidden_control_score_gap": mean(paired_gaps),
        }
    ]


def write_all_outputs(args: argparse.Namespace, data_rows: list[dict[str, Any]], predictions: list[dict[str, Any]], backend_used: str, missing_judges: list[str] | None = None) -> None:
    out_dir = Path(args.out_dir)
    tables_dir = out_dir / "tables"
    reports_dir = out_dir / "reports"
    decision_dir = out_dir / "decision"
    groups = prediction_groups(predictions)

    overall_rows = [metric_summary(groups[split], EVALUATOR, split) for split in SPLITS]
    write_csv(tables_dir / "qwen3_4b_direct_overall_metrics.csv", overall_rows, OVERALL_FIELDS)
    write_csv(tables_dir / "qwen3_4b_low_score_bias.csv", [low_bias_row(groups[split], EVALUATOR, split) for split in SPLITS], LOW_BIAS_FIELDS)
    write_csv(tables_dir / "qwen3_4b_per_label_accuracy.csv", [row for split in SPLITS for row in per_label_rows(groups[split], EVALUATOR, split)], PER_LABEL_FIELDS)
    confusion_fields = ["evaluator", "split", "gold_label", "n", *[f"pred_{label}" for label in LABELS]]
    write_csv(tables_dir / "qwen3_4b_confusion_matrix_counts.csv", [row for split in SPLITS for row in confusion_rows(groups[split], EVALUATOR, split, False)], confusion_fields)
    write_csv(tables_dir / "qwen3_4b_confusion_matrix_row_normalized.csv", [row for split in SPLITS for row in confusion_rows(groups[split], EVALUATOR, split, True)], confusion_fields)
    write_csv(tables_dir / "qwen3_4b_by_metric.csv", [row for split in SPLITS for row in grouped_rows(predictions, "metric", split, BY_METRIC_FIELDS)], BY_METRIC_FIELDS)
    write_csv(tables_dir / "qwen3_4b_by_language.csv", [row for split in SPLITS for row in grouped_rows(predictions, "language", split, BY_LANGUAGE_FIELDS)], BY_LANGUAGE_FIELDS)
    existing_rows, missing = compare_existing_judges(data_rows)
    missing_judges = missing_judges if missing_judges is not None else missing
    write_csv(tables_dir / "qwen3_4b_compare_existing_judges.csv", existing_rows, OVERALL_FIELDS)
    d1_rows = d1_eval_rows(predictions, Path(args.d1_dir)) if Path(args.d1_dir).exists() else []
    write_csv(tables_dir / "qwen3_4b_d1_hidden_eval.csv", d1_rows, D1_FIELDS)

    all_row = overall_rows[0]
    low_to_high_all = safe_float(all_row.get("low_to_high_rate"))
    label2_recall_all = safe_float(all_row.get("label2_recall"))
    parse_success_all = safe_float(all_row.get("parse_success_rate"))
    existing_l2h = [
        safe_float(row.get("low_to_high_rate"))
        for row in existing_rows
        if row.get("split") == "all_5536" and not math.isnan(safe_float(row.get("low_to_high_rate")))
    ]
    clearly_lower = bool(existing_l2h and low_to_high_all < min(existing_l2h) - 0.05)
    direct_success = bool(parse_success_all >= 0.95 and clearly_lower and label2_recall_all > 0)
    decision = {
        "prompt_fixed": True,
        "model_name_or_path": args.model_name_or_path,
        "backend": backend_used,
        "n_total": all_row.get("n"),
        "parse_success_rate_all": json_metric(parse_success_all),
        "low_to_high_all": json_metric(low_to_high_all),
        "label2_recall_all": json_metric(label2_recall_all),
        "direct_baseline_success": direct_success,
        "proceed_to_failure_first_prompt": bool(parse_success_all >= 0.95 and not direct_success),
        "proceed_to_lora": False,
        "reason": (
            "Direct baseline passes the parse and low-score gates."
            if direct_success
            else "Direct baseline is a reference baseline unless it lowers low-to-high and recovers label2 recall."
        ),
    }
    write_json(decision_dir / "exp19_r0a_decision.json", decision)

    report_lines = [
        "# Exp19-R0A Qwen3-4B Direct Scoring Baseline",
        "",
        "## Setup",
        "",
        f"- Dataset: `{relpath(args.data_path)}`.",
        f"- Split mapping: `{relpath(args.split_dir)}`.",
        f"- Model: `{args.model_name_or_path}`.",
        f"- Backend used: `{backend_used}`.",
        "- Training: none.",
        "- Prompt fixed before evaluation: yes.",
        "- Human rationale, human score, D1 label, failure type gold, and existing judge scores are not included in the prompt.",
        "",
        "## Overall Metrics",
        "",
        "| split | n | MAE | QWK | Exact | low-to-high | label2 recall | parse success |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall_rows:
        report_lines.append(
            f"| `{row['split']}` | {row['n']} | {fmt(row['MAE'])} | {fmt(row['QWK'])} | {fmt(row['Exact_Match'])} | "
            f"{fmt(row['low_to_high_rate'])} | {fmt(row['label2_recall'])} | {fmt(row['parse_success_rate'])} |"
        )
    report_lines.extend(
        [
            "",
            "## Answers",
            "",
            f"- Did the model overestimate low-score samples? all_5536 low-to-high rate is `{fmt(low_to_high_all)}`.",
            f"- Does it recover label2 recall? all_5536 label2 recall is `{fmt(label2_recall_all)}`.",
            f"- Existing judge comparison table: `tables/qwen3_4b_compare_existing_judges.csv`.",
            f"- Missing existing judge columns: `{', '.join(missing_judges) if missing_judges else 'none'}`.",
            "- Exp16A/C0b references should be compared from their committed reports; this script does not read model checkpoints.",
            f"- Suitable to proceed to failure-type prompting or LoRA? failure-type prompting: `{decision['proceed_to_failure_first_prompt']}`; LoRA: `{decision['proceed_to_lora']}`.",
            "",
            "## Guardrails",
            "",
            "- Full internal predictions are written under `predictions/full_predictions_internal.jsonl` and must not be committed.",
            "- Full prompts and raw answers are not written to output artifacts.",
        ]
    )
    write_text(reports_dir / "exp19_r0a_qwen4b_direct_baseline_report.md", "\n".join(report_lines))


def fmt(value: Any) -> str:
    try:
        value = float(value)
    except Exception:
        return "NA"
    if math.isnan(value):
        return "NA"
    return f"{value:.4f}"


def json_metric(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return None if math.isnan(out) else out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Exp19-R0A Qwen3-4B direct scoring baseline.")
    parser.add_argument("--model_name_or_path", default="/home/jpang/models/modelscope/Qwen/Qwen3-4B")
    parser.add_argument("--backend", choices=["transformers", "vllm"], default="vllm")
    parser.add_argument("--data_path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--split_dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--d1_dir", type=Path, default=DEFAULT_D1_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=0)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    out_dir = Path(args.out_dir)
    internal_path = out_dir / "predictions" / "full_predictions_internal.jsonl"
    if bool(args.overwrite) and internal_path.exists():
        internal_path.unlink()
    data_rows = load_dataset(Path(args.data_path), Path(args.split_dir), int(args.max_examples))
    if bool(args.dry_run):
        predictions = run_dry_predictions(data_rows, internal_path, args.model_name_or_path)
        backend_used = "dry_run"
    else:
        done = load_existing_predictions(internal_path)
        if len(done) >= len(data_rows):
            predictions = list(done.values())
            backend_used = infer_backend_from_predictions(predictions, str(args.backend))
            print(f"[exp19-r0a] using existing internal predictions: {len(predictions)}", flush=True)
        elif args.backend == "vllm":
            try:
                predictions = run_vllm_predictions(args, data_rows, internal_path, done)
                backend_used = "vllm"
            except Exception as exc:
                print(f"WARNING: vLLM backend failed or is unavailable ({exc}); falling back to transformers.", flush=True)
                predictions = run_transformers_predictions(args, data_rows, internal_path, done)
                backend_used = "transformers_fallback"
        else:
            predictions = run_transformers_predictions(args, data_rows, internal_path, done)
            backend_used = "transformers"
        predictions = list(load_existing_predictions(internal_path).values())
    write_all_outputs(args, data_rows, predictions, backend_used)
    print(
        json.dumps(
            {
                "status": "COMPLETED",
                "out_dir": relpath(out_dir),
                "n_predictions": len(predictions),
                "dry_run": bool(args.dry_run),
                "backend": backend_used,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
