"""Build locked Exp43 train/dev variant records with field-level truncation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.exp43_rubimor.common import (
    DEV_PATH, ROOT, RUBRIC_VARIANTS, TRAIN_PATH, VARIANTS, canonical_metric,
    ensure_dirs, human_stats, raw_rubric, read_jsonl, sample_id, sha256_file,
    stable_hash, write_csv, write_json, write_jsonl,
)

FIELD_BUDGETS = {"question": 256, "metric": 64, "rubric": 384, "metadata": 128, "answer": 1024}


def truncate(tokenizer: Any, value: Any, budget: int) -> tuple[str, int, bool]:
    ids = tokenizer.encode(str(value or ""), add_special_tokens=False)
    return tokenizer.decode(ids[:budget], skip_special_tokens=True), min(len(ids), budget), len(ids) > budget


def context(row: dict[str, Any]) -> str:
    return "\n".join((
        f"Subject: {row.get('subject_canonical') or row.get('subject_raw') or 'unknown'}",
        f"Education level: {row.get('education_level_canonical') or row.get('education_level_raw') or 'unknown'}",
        f"Scenario: {row.get('scenario_canonical') or row.get('scenario_raw') or 'unknown'}",
        f"Language: {row.get('language') or 'unknown'}",
    ))


def format_row(tokenizer: Any, row: dict[str, Any], with_rubric: bool, max_length: int = 2048) -> tuple[str, dict[str, Any]]:
    values = {
        "question": row.get("question"), "metric": canonical_metric(row),
        "rubric": raw_rubric(row), "metadata": context(row), "answer": row.get("answer"),
    }
    fitted: dict[str, str] = {}
    audit: dict[str, Any] = {}
    active = ("question", "metric", "rubric", "metadata", "answer") if with_rubric else ("question", "metric", "answer")
    for field in active:
        fitted[field], audit[f"{field}_tokens"], audit[f"{field}_truncated"] = truncate(tokenizer, values[field], FIELD_BUDGETS[field])
    parts = [f"[QUESTION]\n{fitted['question']}", f"[EVALUATION DIMENSION]\n{fitted['metric']}"]
    if with_rubric:
        parts += [f"[RUBRIC]\n{fitted['rubric']}", f"[CONTEXT]\n{fitted['metadata']}"]
    parts.append(f"[ANSWER]\n{fitted['answer']}")
    text = "\n\n".join(parts)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > max_length:
        # Fixed field budgets are primary; this final answer-only trim handles tokenizer boundary overhead.
        overflow = len(ids) - max_length
        answer_ids = tokenizer.encode(fitted["answer"], add_special_tokens=False)
        keep = max(0, len(answer_ids) - overflow - 2)
        fitted["answer"] = tokenizer.decode(answer_ids[:keep], skip_special_tokens=True)
        parts[-1] = f"[ANSWER]\n{fitted['answer']}"
        text = "\n\n".join(parts)
        audit["answer_truncated"] = True
        audit["answer_tokens"] = keep
    audit["input_tokens"] = len(tokenizer.encode(text, add_special_tokens=False))
    if audit["input_tokens"] > max_length:
        raise RuntimeError(f"Formatter exceeded max_length for {sample_id(row)}")
    for field in FIELD_BUDGETS:
        audit.setdefault(f"{field}_tokens", 0)
        audit.setdefault(f"{field}_truncated", False)
    return text, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--dev", type=Path, default=DEV_PATH)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs(args.out_dir)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=True)
    mapping = json.loads((args.out_dir / "configs/exp43_metric_mapping.json").read_text(encoding="utf-8"))["metrics"]
    summaries, lengths, hashes = [], [], {}
    for split, path in (("train", args.train), ("dev", args.dev)):
        source = read_jsonl(path)
        for variant in VARIANTS:
            built = []
            with_rubric = variant in RUBRIC_VARIANTS
            for position, row in enumerate(source):
                stats = human_stats(row)
                hard = [1.0 if label == stats["gold_label_5"] else 0.0 for label in range(1, 6)]
                text, audit = format_row(tokenizer, row, with_rubric)
                built.append({
                    "position": position, "split": split, "variant": variant,
                    "sample_id": sample_id(row), "question_key": str(row["question_key"]),
                    "text": text, "metric": canonical_metric(row), "metric_id": int(mapping[canonical_metric(row)]),
                    "soft_target_5": stats["human_distribution_5"] if variant not in {"E0", "E1"} else hard,
                    **stats, "language": row.get("language") or "unknown",
                    "subject": row.get("subject_canonical") or row.get("subject_raw") or "unknown",
                    "scenario": row.get("scenario_canonical") or row.get("scenario_raw") or "unknown",
                    "education_level": row.get("education_level_canonical") or row.get("education_level_raw") or "unknown",
                    "core_input_hash": stable_hash((sample_id(row), row.get("question"), row.get("answer"), canonical_metric(row))),
                    **audit,
                })
            output = args.out_dir / f"private/data/exp43_{split}_{variant}.jsonl"
            write_jsonl(output, built)
            hashes[f"{split}_{variant}"] = {"path": str(output), "sha256": sha256_file(output), "rows": len(built)}
            summaries.append({"split": split, "variant": variant, "rows": len(built), "question_keys": len({r['question_key'] for r in built}), "mean_input_tokens": float(np.mean([r['input_tokens'] for r in built])), "p95_input_tokens": float(np.percentile([r['input_tokens'] for r in built], 95)), "max_input_tokens": max(r['input_tokens'] for r in built), **{f"{field}_truncation_count": sum(bool(r[f'{field}_truncated']) for r in built) for field in FIELD_BUDGETS}})
            for row in built:
                lengths.append({"split": split, "variant": variant, "sample_id_hash": stable_hash(row["sample_id"]), **{key: row[key] for key in row if key.endswith("_tokens") or key.endswith("_truncated")}})
    write_csv(args.out_dir / "tables/exp43_dataset_summary.csv", summaries)
    write_csv(args.out_dir / "tables/exp43_input_length_audit.csv", lengths)
    write_json(args.out_dir / "hashes/exp43_data_hashes.json", hashes)
    model_config = Path(args.model_name_or_path) / "config.json"
    tokenizer_config = Path(args.model_name_or_path) / "tokenizer_config.json"
    write_json(args.out_dir / "hashes/exp43_model_hashes.json", {"model_path": args.model_name_or_path, "config_sha256": sha256_file(model_config), "tokenizer_config_sha256": sha256_file(tokenizer_config)})
    protocol = {"variants": list(VARIANTS), "field_budgets": FIELD_BUDGETS, "max_length": 2048, "forbidden_input_fields": ["generator_model", "answer_model", "judge_scores", "human_1", "human_2", "human_3", "label_5", "teacher_output", "sample_weight"]}
    write_json(args.out_dir / "configs/exp43_model_protocol_lock.json", protocol)
    write_json(args.out_dir / "configs/exp43_training_protocol_lock.json", {"epochs": 10, "learning_rate": 2e-5, "optimizer": "AdamW", "weight_decay": .01, "scheduler": "cosine", "warmup_ratio": .05, "point_batch_size": 4, "eval_batch_size": 4, "gradient_accumulation_steps": 32, "max_length": 2048, "gradient_clip": 1.0, "precision": "bf16_all_A6000", "gradient_checkpointing": False, "seeds": [42,43,44], "full_finetuning": True, "groupcv_checkpoint_selection": "fixed_epoch_10", "headline_checkpoint_selection": "highest_dev_exact_then_lower_mae_then_earlier_epoch"})
    write_json(args.out_dir / "configs/exp43_success_gates.json", {"source": "Exp43 preregistration locked at commit 4843230", "stage2": "E3 guards vs E0", "stage3": "E4 guards plus one ordinal improvement", "stage4": "E5 guards plus one metric improvement", "stage5": "E6 guards plus mechanism improvement and E6N control", "stage6": "crossed-CI primary and incremental gains with protection/stability", "headline": "dev primary gain, 2/3 seeds, protection, incremental, E6N and GroupCV direction", "final_test": "one campaign only after GroupCV and headline GO"})
    print(json.dumps({"status": "DATASETS_PREPARED", "artifacts": len(hashes), "test_access_count": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
