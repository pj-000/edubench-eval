"""Prepare Exp19-R5A risk-balanced synthetic-template DPO scout data and QC.

This script constructs DPO preference pairs from the train split only. It does
not train a model, read test, or use dev/D1 annotations for training. R5A is not
strictly on-policy because most rejected responses are template fallback. Full
DPO JSON is written under the gitignored output/data directory; committed
artifacts are limited to QC tables, configs, reports, and redacted samples.
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
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence.collect_exp19_sft_first_round_dev_results import (  # noqa: E402
    find_prediction_file,
    has_no_major_failure,
    has_substantive_failure,
    load_prediction_records,
    normalize_failure_list,
    parsed_payload,
    safe_int,
)
from thesis_exp.exp17_low_score_evidence.prepare_exp19_sft_dpo_datasets import (  # noqa: E402
    MODEL_PATH,
    OPENAI_TAGS,
    assistant_json,
    clean,
    clamp_score,
    dpo_dataset_entry,
    evidence_target,
    language,
    messages_for,
    metric_name,
    question_group,
    read_csv_rows,
    read_jsonl,
    recovered_reason,
    redacted_input,
    redacted_target,
    sample_id,
    sha1,
    subject,
    user_content,
)
from thesis_exp.src.edujudge.utils.io import write_csv, write_json, write_text  # noqa: E402


DEFAULT_TRAIN_JSONL = Path("thesis_exp/data/splits/question_seed42/train.jsonl")
DEFAULT_A0_CANDIDATES = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp17_a0_train_hidden_failure_signals_seed42/"
    "train_hidden_failure_candidates.csv"
)
DEFAULT_A0_HIGH_CONTROLS = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp17_a0_train_hidden_failure_signals_seed42/"
    "train_clean_high_controls.csv"
)
DEFAULT_D2_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_sft_train_dev_diagnostics_seed42")
DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_r5a_onpolicy_dpo_seed42")

DATASET_NAME = "edubench_r5a_onpolicy_risk_balanced_dpo_train"
DATA_FILE = "data/edubench_r5a_onpolicy_risk_balanced_dpo_train.json"
RISK_TYPES = ["low_to_high", "high_to_low_protection", "mid_score_calibration"]
TARGET_FRACTIONS = {
    "low_to_high": 0.40,
    "high_to_low_protection": 0.40,
    "mid_score_calibration": 0.20,
}
ADAPTER_SOURCES = {
    "r1b": "r1b_score_only_balanced",
    "r2n": "r2n_reason_score_natural",
    "r2c": "r2c_clean_reason_score_balanced",
    "r4b": "r4b_shuffled_reason_balanced",
}
MODEL_SOURCES = set(ADAPTER_SOURCES)
TEMPLATE_SOURCES = {
    "template_high_no_failure",
    "template_low_false_failure",
    "template_extreme",
}
USABLE_LOW_USES = {"evidence_positive", "weak_evidence_positive", "pairwise_low", "format_auxiliary"}
USABLE_LOW_TYPES = {"strong_evidence_positive", "weak_evidence_positive", "pairwise_low", "format_auxiliary"}
EXCLUDED_LOW_FAILURES = {"", "unclear", "possible_label_conflict", "no_major_failure"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_rate(num: int | float, den: int | float) -> float:
    return float(num) / float(den) if den else float("nan")


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def fmt(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except Exception:
        return str(value)
    if math.isnan(val):
        return "nan"
    return f"{val:.{digits}f}"


def json_payload(content: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def score_of(target: dict[str, Any]) -> int | None:
    return safe_int(target.get("score"))


def target_from_raw_output(raw_output: str) -> dict[str, Any] | None:
    structured = parsed_payload(raw_output)
    payload = structured.get("payload") or {}
    score = safe_int(payload.get("score"))
    if score is None:
        parsed_score = re.search(r"\b([1-5])\b", clean(raw_output))
        if parsed_score:
            score = int(parsed_score.group(1))
    if score is None:
        return None
    failures = normalize_failure_list(payload.get("major_failures"))
    if not failures:
        failures = ["no_major_failure"] if score >= 4 else ["missing_key_point"] if score <= 2 else ["partial_or_incomplete"]
    score_cap = payload.get("score_cap")
    if score_cap is not None:
        score_cap = safe_int(score_cap)
    elif score <= 2:
        score_cap = min(2, score)
    elif score == 3 and "no_major_failure" not in failures:
        score_cap = 3
    else:
        score_cap = None
    rubric_value = payload.get("rubric_satisfied")
    if isinstance(rubric_value, bool):
        rubric_satisfied = rubric_value
    elif isinstance(rubric_value, str) and rubric_value.lower().strip() in {"true", "false"}:
        rubric_satisfied = rubric_value.lower().strip() == "true"
    else:
        rubric_satisfied = score >= 4 and "no_major_failure" in failures
    return {
        "score": score,
        "major_failures": failures,
        "score_cap": score_cap,
        "rubric_satisfied": rubric_satisfied,
    }


def template_high_no_failure(score: int = 5) -> dict[str, Any]:
    return {
        "score": score,
        "major_failures": ["no_major_failure"],
        "score_cap": None,
        "rubric_satisfied": True,
    }


def template_low_false_failure(score: int = 2) -> dict[str, Any]:
    return {
        "score": score,
        "major_failures": ["missing_key_point"],
        "score_cap": min(2, score),
        "rubric_satisfied": False,
    }


def template_mid_extreme(row: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    if rng.random() < 0.5:
        return template_high_no_failure(5)
    return template_low_false_failure(1)


def is_low_to_high_rejected(target: dict[str, Any]) -> bool:
    score = score_of(target)
    if score is not None and score >= 4:
        return True
    return has_no_major_failure(target) and (score is None or score >= 3)


def is_high_to_low_rejected(target: dict[str, Any]) -> bool:
    score = score_of(target)
    return (score is not None and score <= 2) or has_substantive_failure(target)


def is_usable_low_candidate(signal: dict[str, str] | None) -> bool:
    if signal is None:
        return False
    use = clean(signal.get("recommended_training_use"))
    candidate_type = clean(signal.get("hidden_failure_candidate_type"))
    failure = clean(signal.get("failure_mode_auto"))
    return (
        (use in USABLE_LOW_USES or candidate_type in USABLE_LOW_TYPES)
        and candidate_type not in {"answer_key_dependent", "conflict_or_exclude", "unclear"}
        and failure not in EXCLUDED_LOW_FAILURES
    )


def leakage_check(rows: list[dict[str, Any]], candidates: dict[str, dict[str, str]]) -> tuple[int, int]:
    checked = 0
    leaks = 0
    for row in rows:
        reason = recovered_reason(candidates.get(sample_id(row)))
        if len(reason) < 24:
            continue
        checked += 1
        if reason in user_content(row):
            leaks += 1
    return checked, leaks


def load_d2_subset_predictions(d2_dir: Path, subset: str) -> dict[str, list[tuple[str, dict[str, Any]]]]:
    refs_path = d2_dir / "references" / f"{subset}_reference.csv"
    if not refs_path.exists():
        return {}
    refs = read_csv(refs_path)
    out: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for source, adapter_name in ADAPTER_SOURCES.items():
        run_dir = d2_dir / "predictions" / adapter_name / subset
        try:
            pred_file = find_prediction_file(run_dir)
        except FileNotFoundError:
            continue
        records = load_prediction_records(pred_file)
        for ref, record in zip(refs, records):
            target = target_from_raw_output(str(record.get("raw_output", "")))
            sid = clean(ref.get("sample_id"))
            if sid and target is not None:
                out[sid].append((source, target))
    return out


def dpo_example(
    row: dict[str, Any],
    chosen: dict[str, Any],
    rejected: dict[str, Any],
    risk_type: str,
    rejected_source: str,
    copy_index: int,
) -> dict[str, Any]:
    return {
        "messages": messages_for(row),
        "chosen": {"role": "assistant", "content": assistant_json(chosen)},
        "rejected": {"role": "assistant", "content": assistant_json(rejected)},
        "risk_type": risk_type,
        "rejected_source": rejected_source,
        "source_sample_id": sample_id(row),
        "oversample_copy_index": copy_index,
    }


def target_gap(chosen: dict[str, Any], rejected: dict[str, Any]) -> int | None:
    chosen_score = score_of(chosen)
    rejected_score = score_of(rejected)
    if chosen_score is None or rejected_score is None:
        return None
    return abs(chosen_score - rejected_score)


def add_pair_candidate(
    pool: list[dict[str, Any]],
    row: dict[str, Any],
    chosen: dict[str, Any],
    rejected: dict[str, Any],
    risk_type: str,
    rejected_source: str,
) -> None:
    pool.append(
        {
            "row": row,
            "chosen": chosen,
            "rejected": rejected,
            "risk_type": risk_type,
            "rejected_source": rejected_source,
        }
    )


def build_pair_pools(
    train_rows: list[dict[str, Any]],
    candidates: dict[str, dict[str, str]],
    high_controls: dict[str, dict[str, str]],
    d2_dir: Path,
    rng: random.Random,
) -> dict[str, list[dict[str, Any]]]:
    low_predictions = load_d2_subset_predictions(d2_dir, "train_low_reasoned_subset")
    high_predictions = load_d2_subset_predictions(d2_dir, "train_clean_high_subset")
    pools: dict[str, list[dict[str, Any]]] = {risk_type: [] for risk_type in RISK_TYPES}
    for row in train_rows:
        sid = sample_id(row)
        label = clamp_score(row.get("label_5"))
        if label <= 2 and is_usable_low_candidate(candidates.get(sid)):
            chosen = evidence_target(row, candidates.get(sid), include_reason=False)
            actuals = [
                (source, target)
                for source, target in low_predictions.get(sid, [])
                if is_low_to_high_rejected(target)
            ]
            if actuals:
                for source, rejected in actuals:
                    add_pair_candidate(pools["low_to_high"], row, chosen, rejected, "low_to_high", source)
            else:
                add_pair_candidate(
                    pools["low_to_high"],
                    row,
                    chosen,
                    template_high_no_failure(5),
                    "low_to_high",
                    "template_high_no_failure",
                )
        elif label >= 4 and sid in high_controls:
            chosen = template_high_no_failure(label)
            actuals = [
                (source, target)
                for source, target in high_predictions.get(sid, [])
                if is_high_to_low_rejected(target)
            ]
            if actuals:
                for source, rejected in actuals:
                    add_pair_candidate(
                        pools["high_to_low_protection"],
                        row,
                        chosen,
                        rejected,
                        "high_to_low_protection",
                        source,
                    )
            else:
                add_pair_candidate(
                    pools["high_to_low_protection"],
                    row,
                    chosen,
                    template_low_false_failure(2),
                    "high_to_low_protection",
                    "template_low_false_failure",
                )
        elif label == 3:
            chosen = {
                "score": 3,
                "major_failures": ["partial_or_incomplete"],
                "score_cap": 3,
                "rubric_satisfied": False,
            }
            add_pair_candidate(
                pools["mid_score_calibration"],
                row,
                chosen,
                template_mid_extreme(row, rng),
                "mid_score_calibration",
                "template_extreme",
            )
    return pools


def sample_balanced_pairs(
    pools: dict[str, list[dict[str, Any]]],
    total_pairs: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    targets = {
        "low_to_high": int(round(total_pairs * TARGET_FRACTIONS["low_to_high"])),
        "high_to_low_protection": int(round(total_pairs * TARGET_FRACTIONS["high_to_low_protection"])),
    }
    targets["mid_score_calibration"] = total_pairs - targets["low_to_high"] - targets["high_to_low_protection"]
    out: list[dict[str, Any]] = []
    for risk_type in RISK_TYPES:
        pool = pools.get(risk_type, [])
        if not pool:
            continue
        count = targets[risk_type]
        selected = rng.sample(pool, count) if count <= len(pool) else [rng.choice(pool) for _ in range(count)]
        for copy_index, item in enumerate(selected):
            out.append(
                dpo_example(
                    item["row"],
                    item["chosen"],
                    item["rejected"],
                    risk_type,
                    item["rejected_source"],
                    copy_index,
                )
            )
    rng.shuffle(out)
    return out


def target_from_message(message: dict[str, str]) -> dict[str, Any]:
    return json_payload(clean(message.get("content")))


def pair_quality_rows(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        groups[pair["risk_type"]].append(pair)
    rows: list[dict[str, Any]] = []
    for risk_type in RISK_TYPES:
        items = groups.get(risk_type, [])
        chosen_targets = [target_from_message(pair["chosen"]) for pair in items]
        rejected_targets = [target_from_message(pair["rejected"]) for pair in items]
        chosen_scores = [score_of(target) for target in chosen_targets]
        rejected_scores = [score_of(target) for target in rejected_targets]
        gaps = [
            abs(int(chosen) - int(rejected))
            for chosen, rejected in zip(chosen_scores, rejected_scores)
            if chosen is not None and rejected is not None
        ]
        n = len(items)
        rows.append(
            {
                "risk_type": risk_type,
                "n": n,
                "chosen_score_mean": mean([float(x) for x in chosen_scores if x is not None]),
                "rejected_score_mean": mean([float(x) for x in rejected_scores if x is not None]),
                "score_gap_mean": mean([float(x) for x in gaps]),
                "chosen_has_failure_rate": safe_rate(sum(1 for target in chosen_targets if has_substantive_failure(target)), n),
                "rejected_has_failure_rate": safe_rate(
                    sum(1 for target in rejected_targets if has_substantive_failure(target)), n
                ),
                "chosen_no_major_failure_rate": safe_rate(sum(1 for target in chosen_targets if has_no_major_failure(target)), n),
                "rejected_no_major_failure_rate": safe_rate(
                    sum(1 for target in rejected_targets if has_no_major_failure(target)), n
                ),
                "chosen_score_cap_nonnull_rate": safe_rate(
                    sum(1 for target in chosen_targets if target.get("score_cap") is not None), n
                ),
                "rejected_score_cap_nonnull_rate": safe_rate(
                    sum(1 for target in rejected_targets if target.get("score_cap") is not None), n
                ),
            }
        )
    return rows


def source_count_rows(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter((pair["rejected_source"], pair["risk_type"]) for pair in pairs)
    return [
        {"rejected_source": source, "risk_type": risk_type, "n": n}
        for (source, risk_type), n in sorted(counts.items(), key=lambda item: (item[0][1], item[0][0]))
    ]


def pair_count_rows(pools: dict[str, list[dict[str, Any]]], pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded = Counter(pair["risk_type"] for pair in pairs)
    total = len(pairs)
    return [
        {
            "risk_type": risk_type,
            "original_count": len(pools.get(risk_type, [])),
            "expanded_count": expanded.get(risk_type, 0),
            "target_fraction": TARGET_FRACTIONS[risk_type],
            "actual_fraction": safe_rate(expanded.get(risk_type, 0), total),
        }
        for risk_type in RISK_TYPES
    ]


def redacted_pair(pair: dict[str, Any]) -> dict[str, Any]:
    row = {"sample_id": pair.get("source_sample_id", "")}
    # Find the user message preview without serializing raw answers in full.
    messages = pair.get("messages") or []
    user_text = ""
    for msg in messages:
        if msg.get("role") == "user":
            user_text = clean(msg.get("content"))
            break
    return {
        "sample_id": row["sample_id"],
        "prompt_hash": sha1(user_text),
        "prompt_preview": " ".join(user_text.split())[:180],
        "chosen_redacted": redacted_target(target_from_message(pair["chosen"])),
        "rejected_redacted": redacted_target(target_from_message(pair["rejected"])),
        "risk_type": pair.get("risk_type", ""),
        "rejected_source": pair.get("rejected_source", ""),
        "oversample_copy_index": pair.get("oversample_copy_index", ""),
    }


def write_redacted_samples(out_dir: Path, pairs: list[dict[str, Any]], limit: int) -> None:
    redacted_dir = out_dir / "redacted_samples"
    for risk_type, filename in [
        ("low_to_high", "low_to_high_pairs_redacted.json"),
        ("high_to_low_protection", "high_to_low_pairs_redacted.json"),
        ("mid_score_calibration", "mid_pairs_redacted.json"),
    ]:
        selected = [pair for pair in pairs if pair["risk_type"] == risk_type][:limit]
        write_json(redacted_dir / filename, [redacted_pair(pair) for pair in selected])


def write_json_array(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def write_dpo_config(path: Path, adapter_path: str, output_dir: str, dataset_dir: Path) -> None:
    mapping = {
        "model_name_or_path": MODEL_PATH,
        "trust_remote_code": True,
        "stage": "dpo",
        "do_train": True,
        "finetuning_type": "lora",
        "adapter_name_or_path": adapter_path,
        "ref_model": MODEL_PATH,
        "ref_model_adapters": adapter_path,
        "dataset_dir": str(dataset_dir),
        "dataset": DATASET_NAME,
        "template": "qwen3_nothink",
        "cutoff_len": 4096,
        "overwrite_cache": True,
        "preprocessing_num_workers": 16,
        "pref_loss": "sigmoid",
        "pref_beta": 0.1,
        "pref_ftx": 0.1,
        "output_dir": output_dir,
        "logging_steps": 10,
        "save_steps": 100,
        "save_only_model": True,
        "plot_loss": True,
        "report_to": "none",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": "5.0e-6",
        "num_train_epochs": 1,
        "bf16": True,
        "gradient_checkpointing": True,
    }
    text = "\n".join(f"{key}: {yaml_scalar(value)}" for key, value in mapping.items()) + "\n"
    write_text(path, text)


def report_text(
    pairs: list[dict[str, Any]],
    count_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    quality_rows: list[dict[str, Any]],
    leakage_checked: int,
    leakage_count: int,
) -> tuple[str, bool]:
    total = len(pairs)
    source_counts = Counter(pair["rejected_source"] for pair in pairs)
    actual_model_count = sum(source_counts.get(source, 0) for source in MODEL_SOURCES)
    template_count = sum(source_counts.get(source, 0) for source in TEMPLATE_SOURCES)
    low_pairs = [pair for pair in pairs if pair["risk_type"] == "low_to_high"]
    high_pairs = [pair for pair in pairs if pair["risk_type"] == "high_to_low_protection"]
    low_valid = sum(1 for pair in low_pairs if is_low_to_high_rejected(target_from_message(pair["rejected"])))
    high_valid = sum(1 for pair in high_pairs if is_high_to_low_rejected(target_from_message(pair["rejected"])))
    scout_ready = (
        total > 0
        and leakage_count == 0
        and all(int(row["expanded_count"]) > 0 for row in count_rows)
        and safe_rate(low_valid, len(low_pairs)) >= 0.95
        and safe_rate(high_valid, len(high_pairs)) >= 0.95
    )
    actual_rate = safe_rate(actual_model_count, total)
    main_ready = scout_ready and actual_rate >= 0.30
    lines = [
        "# Exp19-R5A Risk-Balanced Synthetic-Template DPO Scout QC Report",
        "",
        "Exp19-R5A constructs DPO preference pairs from the train split only. It does not train, read test, or use dev/D1 annotations for training.",
        "",
        "R5A is **not strictly on-policy**: actual model-output rejected responses are limited, while template rejected responses dominate. Treat R5A as a DPO pipeline scout/control, not as the main DPO experiment.",
        "",
        "## Summary",
        "",
        f"- total DPO pairs constructed: {total}",
        f"- actual model-output rejected responses: {actual_model_count} ({fmt(safe_rate(actual_model_count, total))})",
        f"- template rejected responses: {template_count} ({fmt(safe_rate(template_count, total))})",
        f"- recovered human rationales checked for prompt leakage: {leakage_checked}",
        f"- exact human-rationale leakage count in prompts: {leakage_count}",
        f"- low_to_high rejected validity rate: {fmt(safe_rate(low_valid, len(low_pairs)))}",
        f"- high_to_low rejected validity rate: {fmt(safe_rate(high_valid, len(high_pairs)))}",
        f"- dataset ready for DPO pipeline scout: `{scout_ready}`",
        f"- dataset ready for main DPO: `{main_ready}`",
        "- better name: `risk-balanced synthetic-template DPO scout`.",
        "- R5B rejection-mined hybrid DPO should be used before claiming a main DPO result.",
        "",
        "## Pair Counts",
        "",
        "| risk_type | original | expanded | target | actual |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in count_rows:
        lines.append(
            f"| {row['risk_type']} | {row['original_count']} | {row['expanded_count']} | "
            f"{fmt(row['target_fraction'])} | {fmt(row['actual_fraction'])} |"
        )
    lines.extend(["", "## Rejected Source Counts", "", "| source | risk_type | n |", "|---|---|---:|"])
    for row in source_rows:
        lines.append(f"| {row['rejected_source']} | {row['risk_type']} | {row['n']} |")
    lines.extend(["", "## Pair Quality", "", "| risk_type | n | chosen_mean | rejected_mean | gap | chosen_failure | rejected_failure | rejected_no_failure |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in quality_rows:
        lines.append(
            f"| {row['risk_type']} | {row['n']} | {fmt(row['chosen_score_mean'])} | "
            f"{fmt(row['rejected_score_mean'])} | {fmt(row['score_gap_mean'])} | "
            f"{fmt(row['chosen_has_failure_rate'])} | {fmt(row['rejected_has_failure_rate'])} | "
            f"{fmt(row['rejected_no_major_failure_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- DPO prompt contains only question, answer, metric, rubric, and metadata.",
            "- Human-rationale-derived fields appear only in chosen assistant targets.",
            "- Full DPO JSON is written under gitignored `data/` and should not be committed.",
            "- Dev/D1 annotations are not read by this script and must remain evaluation-only.",
        ]
    )
    return "\n".join(lines), main_ready


def build(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    train_rows = read_jsonl(args.train_jsonl)
    candidates = {clean(row.get("sample_id")): row for row in read_csv_rows(args.a0_candidates)}
    high_controls = {clean(row.get("sample_id")): row for row in read_csv_rows(args.a0_high_controls)}
    pools = build_pair_pools(train_rows, candidates, high_controls, args.d2_dir, rng)
    pairs = sample_balanced_pairs(pools, args.total_pairs, rng)

    write_json_array(args.out_dir / DATA_FILE, pairs)
    dataset_info = {DATASET_NAME: dpo_dataset_entry(DATA_FILE)}
    if dataset_info[DATASET_NAME].get("tags") != OPENAI_TAGS:
        raise ValueError("Unexpected ShareGPT tag mapping.")
    write_json(args.out_dir / "dataset_info_r5a_snippet.json", dataset_info)

    count_rows = pair_count_rows(pools, pairs)
    source_rows = source_count_rows(pairs)
    quality_rows = pair_quality_rows(pairs)
    write_csv(args.out_dir / "tables" / "exp19_r5a_dpo_pair_counts.csv", count_rows)
    write_csv(args.out_dir / "tables" / "exp19_r5a_rejected_source_counts.csv", source_rows)
    write_csv(args.out_dir / "tables" / "exp19_r5a_pair_quality_stats.csv", quality_rows)
    write_redacted_samples(args.out_dir, pairs, args.redacted_limit)
    leakage_checked, leakage_count = leakage_check(train_rows, candidates)
    report, ready = report_text(pairs, count_rows, source_rows, quality_rows, leakage_checked, leakage_count)
    write_text(args.out_dir / "reports" / "exp19_r5a_onpolicy_dpo_qc_report.md", report)
    write_dpo_config(
        args.out_dir / "configs" / "llamafactory_qwen3_4b_r5a_dpo_from_r2c.yaml",
        "saves/edubench/qwen3-4b/r2_clean_reason_score_balanced_lora",
        "saves/edubench/qwen3-4b/r5a_dpo_from_r2c_lora",
        args.out_dir,
    )
    write_dpo_config(
        args.out_dir / "configs" / "llamafactory_qwen3_4b_r5a_dpo_from_r1b.yaml",
        "saves/edubench/qwen3-4b/r1_score_only_balanced_lora",
        "saves/edubench/qwen3-4b/r5a_dpo_from_r1b_lora",
        args.out_dir,
    )
    return {
        "pairs": len(pairs),
        "ready_for_dpo": ready,
        "leakage_count": leakage_count,
        "counts": {row["risk_type"]: row["expanded_count"] for row in count_rows},
        "actual_model_rejected": sum(1 for pair in pairs if pair["rejected_source"] in MODEL_SOURCES),
        "template_rejected": sum(1 for pair in pairs if pair["rejected_source"] in TEMPLATE_SOURCES),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Exp19-R5A risk-balanced synthetic-template DPO scout data.")
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--a0-candidates", type=Path, default=DEFAULT_A0_CANDIDATES)
    parser.add_argument("--a0-high-controls", type=Path, default=DEFAULT_A0_HIGH_CONTROLS)
    parser.add_argument("--d2-dir", type=Path, default=DEFAULT_D2_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-pairs", type=int, default=3000)
    parser.add_argument("--redacted-limit", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build(args)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
