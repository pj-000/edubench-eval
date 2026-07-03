"""Prepare Exp19-R5B rejection-mined hybrid DPO data and QC.

R5B is the corrected follow-up to R5A. It still builds preference pairs from
the train split only, but it prioritizes actual model mistakes mined from
existing SFT prediction outputs. Hard synthetic rejected targets are used only
when actual wrong outputs are unavailable; extreme templates are the final
fallback. The script does not train models, read test, or use dev/D1
annotations for training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
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
    safe_int,
)
from thesis_exp.exp17_low_score_evidence.prepare_exp19_r5a_onpolicy_dpo import (  # noqa: E402
    ADAPTER_SOURCES,
    EXCLUDED_LOW_FAILURES,
    RISK_TYPES,
    TARGET_FRACTIONS,
    USABLE_LOW_TYPES,
    USABLE_LOW_USES,
    fmt,
    json_payload,
    mean,
    safe_rate,
    score_of,
    target_from_raw_output,
)
from thesis_exp.exp17_low_score_evidence.prepare_exp19_sft_dpo_datasets import (  # noqa: E402
    MODEL_PATH,
    OPENAI_TAGS,
    assistant_json,
    clamp_score,
    clean,
    dpo_dataset_entry,
    evidence_target,
    messages_for,
    read_csv_rows,
    read_jsonl,
    recovered_reason,
    redacted_target,
    sample_id,
    sha1,
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
DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_r5b_rejection_mined_dpo_seed42")

DATASET_NAME = "edubench_r5b_rejection_mined_dpo_train"
DATA_FILE = "data/edubench_r5b_rejection_mined_dpo_train.json"
MODEL_SOURCES = set(ADAPTER_SOURCES)
HARD_SYNTHETIC_SOURCES = {
    "hard_synthetic_high_no_failure",
    "hard_synthetic_high_unclear",
    "hard_synthetic_conservative_insufficient",
    "hard_synthetic_conservative_unclear",
    "hard_synthetic_mid_high",
    "hard_synthetic_mid_low",
}
EXTREME_TEMPLATE_SOURCES = {
    "template_extreme_high5",
    "template_extreme_low1",
    "template_extreme_low2",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def write_json_array(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def target_from_message(message: dict[str, str]) -> dict[str, Any]:
    return json_payload(clean(message.get("content")))


def target_gap(chosen: dict[str, Any], rejected: dict[str, Any]) -> int | None:
    chosen_score = score_of(chosen)
    rejected_score = score_of(rejected)
    if chosen_score is None or rejected_score is None:
        return None
    return abs(chosen_score - rejected_score)


def source_category(source: str) -> str:
    if source in MODEL_SOURCES or source.startswith("actual_"):
        return "actual_model"
    if source in HARD_SYNTHETIC_SOURCES:
        return "hard_synthetic"
    if source in EXTREME_TEMPLATE_SOURCES or source.startswith("template_"):
        return "extreme_template"
    return "unknown"


def hard_high_rejected(rng: random.Random) -> tuple[str, dict[str, Any]]:
    if rng.random() < 0.7:
        return (
            "hard_synthetic_high_no_failure",
            {
                "score": 4,
                "major_failures": ["no_major_failure"],
                "score_cap": None,
                "rubric_satisfied": True,
            },
        )
    return (
        "hard_synthetic_high_unclear",
        {
            "score": 4,
            "major_failures": ["unclear"],
            "score_cap": None,
            "rubric_satisfied": True,
        },
    )


def hard_conservative_rejected(rng: random.Random) -> tuple[str, dict[str, Any]]:
    if rng.random() < 0.7:
        return (
            "hard_synthetic_conservative_insufficient",
            {
                "score": 3,
                "major_failures": ["insufficient_evidence"],
                "score_cap": 3,
                "rubric_satisfied": False,
            },
        )
    return (
        "hard_synthetic_conservative_unclear",
        {
            "score": 2,
            "major_failures": ["unclear"],
            "score_cap": 2,
            "rubric_satisfied": False,
        },
    )


def hard_mid_rejected(rng: random.Random) -> tuple[str, dict[str, Any]]:
    if rng.random() < 0.5:
        return (
            "hard_synthetic_mid_high",
            {
                "score": 5,
                "major_failures": ["no_major_failure"],
                "score_cap": None,
                "rubric_satisfied": True,
            },
        )
    return (
        "hard_synthetic_mid_low",
        {
            "score": 1,
            "major_failures": ["missing_key_point"],
            "score_cap": 1,
            "rubric_satisfied": False,
        },
    )


def extreme_high_rejected() -> tuple[str, dict[str, Any]]:
    return (
        "template_extreme_high5",
        {
            "score": 5,
            "major_failures": ["no_major_failure"],
            "score_cap": None,
            "rubric_satisfied": True,
        },
    )


def extreme_low_rejected(score: int = 1) -> tuple[str, dict[str, Any]]:
    return (
        f"template_extreme_low{score}",
        {
            "score": score,
            "major_failures": ["missing_key_point"],
            "score_cap": min(score, 2),
            "rubric_satisfied": False,
        },
    )


def low_chosen(row: dict[str, Any], signal: dict[str, str] | None) -> dict[str, Any]:
    return evidence_target(row, signal, include_reason=False)


def high_chosen(row: dict[str, Any]) -> dict[str, Any]:
    label = clamp_score(row.get("label_5"))
    return {
        "score": label,
        "major_failures": ["no_major_failure"],
        "score_cap": None,
        "rubric_satisfied": True,
    }


def mid_chosen() -> dict[str, Any]:
    return {
        "score": 3,
        "major_failures": ["partial_or_incomplete"],
        "score_cap": 3,
        "rubric_satisfied": False,
    }


def is_low_to_high_rejected(target: dict[str, Any]) -> bool:
    score = score_of(target)
    if score is not None and score >= 4:
        return True
    if has_no_major_failure(target):
        return True
    if target.get("score_cap") is None:
        return True
    if target.get("rubric_satisfied") is True:
        return True
    return False


def is_high_to_low_rejected(target: dict[str, Any]) -> bool:
    score = score_of(target)
    if score is not None and score <= 2:
        return True
    if has_substantive_failure(target):
        return True
    cap = safe_int(target.get("score_cap"))
    if cap is not None and cap <= 2:
        return True
    if target.get("rubric_satisfied") is False:
        return True
    return False


def is_mid_rejected(target: dict[str, Any]) -> bool:
    score = score_of(target)
    return score is not None and (score <= 1 or score >= 5)


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


def load_subset_predictions(d2_dir: Path, subset: str) -> dict[str, list[tuple[str, dict[str, Any]]]]:
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
            "rejected_category": source_category(rejected_source),
        }
    )


def build_pair_pools(
    train_rows: list[dict[str, Any]],
    candidates: dict[str, dict[str, str]],
    high_controls: dict[str, dict[str, str]],
    d2_dir: Path,
    rng: random.Random,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    low_predictions = load_subset_predictions(d2_dir, "train_low_reasoned_subset")
    high_predictions = load_subset_predictions(d2_dir, "train_clean_high_subset")
    mid_predictions = load_subset_predictions(d2_dir, "train_mid_subset")
    pools: dict[str, list[dict[str, Any]]] = {risk_type: [] for risk_type in RISK_TYPES}
    diagnostics = {
        "prediction_subsets_found": {
            "train_low_reasoned_subset": bool(low_predictions),
            "train_clean_high_subset": bool(high_predictions),
            "train_mid_subset": bool(mid_predictions),
        },
        "generation_needed_for": [],
    }

    for row in train_rows:
        sid = sample_id(row)
        label = clamp_score(row.get("label_5"))
        if label <= 2 and is_usable_low_candidate(candidates.get(sid)):
            chosen = low_chosen(row, candidates.get(sid))
            actuals = [
                (source, target)
                for source, target in low_predictions.get(sid, [])
                if is_low_to_high_rejected(target)
            ]
            if actuals:
                for source, rejected in actuals:
                    add_pair_candidate(pools["low_to_high"], row, chosen, rejected, "low_to_high", source)
            else:
                source, rejected = hard_high_rejected(rng)
                add_pair_candidate(pools["low_to_high"], row, chosen, rejected, "low_to_high", source)
        elif label >= 4 and sid in high_controls:
            chosen = high_chosen(row)
            actuals = [
                (source, target)
                for source, target in high_predictions.get(sid, [])
                if is_high_to_low_rejected(target)
            ]
            if actuals:
                for source, rejected in actuals:
                    add_pair_candidate(pools["high_to_low_protection"], row, chosen, rejected, "high_to_low_protection", source)
            else:
                source, rejected = hard_conservative_rejected(rng)
                add_pair_candidate(pools["high_to_low_protection"], row, chosen, rejected, "high_to_low_protection", source)
        elif label == 3:
            chosen = mid_chosen()
            actuals = [
                (source, target)
                for source, target in mid_predictions.get(sid, [])
                if is_mid_rejected(target)
            ]
            if actuals:
                for source, rejected in actuals:
                    add_pair_candidate(pools["mid_score_calibration"], row, chosen, rejected, "mid_score_calibration", source)
            else:
                source, rejected = hard_mid_rejected(rng)
                add_pair_candidate(pools["mid_score_calibration"], row, chosen, rejected, "mid_score_calibration", source)

    source_counts = Counter(item["rejected_category"] for pool in pools.values() for item in pool)
    if source_counts.get("actual_model", 0) == 0:
        diagnostics["generation_needed_for"].extend(["low_to_high", "high_to_low_protection", "mid_score_calibration"])
    elif "train_mid_subset" not in diagnostics["prediction_subsets_found"] or not mid_predictions:
        diagnostics["generation_needed_for"].append("mid_score_calibration")
    return pools, diagnostics


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
        "rejected_category": source_category(rejected_source),
        "source_sample_id": sample_id(row),
        "oversample_copy_index": copy_index,
    }


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


def source_count_rows(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, str]] = Counter(
        (pair["rejected_source"], pair["rejected_category"], pair["risk_type"]) for pair in pairs
    )
    return [
        {"rejected_source": source, "rejected_category": category, "risk_type": risk_type, "n": n}
        for (source, category, risk_type), n in sorted(counts.items(), key=lambda item: (item[0][2], item[0][1], item[0][0]))
    ]


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


def validity_for_pair(pair: dict[str, Any]) -> bool:
    target = target_from_message(pair["rejected"])
    risk_type = pair.get("risk_type")
    if risk_type == "low_to_high":
        return is_low_to_high_rejected(target)
    if risk_type == "high_to_low_protection":
        return is_high_to_low_rejected(target)
    if risk_type == "mid_score_calibration":
        return is_mid_rejected(target)
    return False


def hardness_rows(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(pairs)
    for risk_type in RISK_TYPES:
        risk_items = [pair for pair in pairs if pair["risk_type"] == risk_type]
        for category in ["actual_model", "hard_synthetic", "extreme_template"]:
            items = [pair for pair in risk_items if pair.get("rejected_category") == category]
            chosen_targets = [target_from_message(pair["chosen"]) for pair in items]
            rejected_targets = [target_from_message(pair["rejected"]) for pair in items]
            gaps = [
                target_gap(chosen, rejected)
                for chosen, rejected in zip(chosen_targets, rejected_targets)
                if target_gap(chosen, rejected) is not None
            ]
            rows.append(
                {
                    "risk_type": risk_type,
                    "rejected_category": category,
                    "n": len(items),
                    "fraction_within_risk": safe_rate(len(items), len(risk_items)),
                    "fraction_overall": safe_rate(len(items), total),
                    "score_gap_mean": mean([float(x) for x in gaps if x is not None]),
                    "validity_rate": safe_rate(sum(1 for pair in items if validity_for_pair(pair)), len(items)),
                }
            )
    return rows


def redacted_pair(pair: dict[str, Any]) -> dict[str, Any]:
    messages = pair.get("messages") or []
    user_text = ""
    for msg in messages:
        if msg.get("role") == "user":
            user_text = clean(msg.get("content"))
            break
    return {
        "sample_id": pair.get("source_sample_id", ""),
        "prompt_hash": sha1(user_text),
        "prompt_preview": " ".join(user_text.split())[:180],
        "chosen_redacted": redacted_target(target_from_message(pair["chosen"])),
        "rejected_redacted": redacted_target(target_from_message(pair["rejected"])),
        "risk_type": pair.get("risk_type", ""),
        "rejected_source": pair.get("rejected_source", ""),
        "rejected_category": pair.get("rejected_category", ""),
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
    write_text(path, "\n".join(f"{key}: {yaml_scalar(value)}" for key, value in mapping.items()) + "\n")


def generation_plan_text(summary: dict[str, Any]) -> str:
    needed = summary.get("generation_needed_for") or []
    lines = [
        "# Exp19-R5B Rejection Generation Plan",
        "",
        "R5B could not reach the desired actual model rejected rate from existing train-subset predictions.",
        "Generate more candidate outputs for rejection mining before treating DPO as a main experiment.",
        "",
        "## Needed Risk Types",
        "",
    ]
    if needed:
        lines.extend(f"- {item}" for item in needed)
    else:
        lines.append("- No missing risk type, but actual rejected rate is still below the main-DPO threshold.")
    lines.extend(
        [
            "",
            "## Adapters",
            "",
            "- r1b_score_only_balanced",
            "- r2n_reason_score_natural",
            "- r2c_clean_reason_score_balanced",
            "- r4b_shuffled_reason_balanced",
            "",
            "## Recommended Sampling Parameters",
            "",
            "- temperature: 0.7",
            "- top_p: 0.9",
            "- max_new_tokens: 256",
            "- num_samples_per_input: 4",
            "",
            "## Mining Filters",
            "",
            "- low_to_high: keep outputs with score >= 4, no_major_failure, score_cap null, or rubric_satisfied true.",
            "- high_to_low_protection: keep outputs with score <= 2, substantive failure, score_cap <= 2, or rubric_satisfied false.",
            "- mid_score_calibration: keep outputs with score <= 1 or score >= 5.",
            "",
            "Do not use test or dev/D1 annotations for training. Human rationale must remain out of the user prompt.",
        ]
    )
    return "\n".join(lines)


def report_text(
    pairs: list[dict[str, Any]],
    count_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    quality_rows: list[dict[str, Any]],
    hardness: list[dict[str, Any]],
    leakage_checked: int,
    leakage_count: int,
    diagnostics: dict[str, Any],
) -> tuple[str, bool, dict[str, Any]]:
    total = len(pairs)
    category_counts = Counter(pair.get("rejected_category") for pair in pairs)
    actual_count = category_counts.get("actual_model", 0)
    hard_count = category_counts.get("hard_synthetic", 0)
    extreme_count = category_counts.get("extreme_template", 0)
    actual_rate = safe_rate(actual_count, total)
    hard_rate = safe_rate(hard_count, total)
    extreme_rate = safe_rate(extreme_count, total)
    low_pairs = [pair for pair in pairs if pair["risk_type"] == "low_to_high"]
    high_pairs = [pair for pair in pairs if pair["risk_type"] == "high_to_low_protection"]
    low_valid = safe_rate(sum(1 for pair in low_pairs if validity_for_pair(pair)), len(low_pairs))
    high_valid = safe_rate(sum(1 for pair in high_pairs if validity_for_pair(pair)), len(high_pairs))
    all_risk_present = all(int(row["expanded_count"]) > 0 for row in count_rows)
    # Conservative gate: below 20% actual model rejected, R5B is still only a
    # scout even when hard synthetic fallback is clean.
    ready_for_main = (
        total > 0
        and leakage_count == 0
        and all_risk_present
        and low_valid >= 0.95
        and high_valid >= 0.95
        and actual_rate >= 0.20
        and (actual_rate >= 0.30 or (hard_rate >= 0.50 and extreme_rate <= 0.30))
    )
    need_generation = actual_rate < 0.30
    if need_generation and not diagnostics.get("generation_needed_for"):
        diagnostics["generation_needed_for"] = ["more_actual_rejections_all_risk_types"]

    lines = [
        "# Exp19-R5B Rejection-Mined Hybrid DPO QC Report",
        "",
        "Exp19-R5B constructs train-only DPO preference pairs by prioritizing actual SFT model mistakes as rejected responses. Hard synthetic fallback is used when real mistakes are unavailable; extreme templates are last fallback.",
        "",
        "## Summary",
        "",
        f"- total DPO pairs constructed: {total}",
        f"- actual model-output rejected responses: {actual_count} ({fmt(actual_rate)})",
        f"- hard synthetic rejected responses: {hard_count} ({fmt(hard_rate)})",
        f"- extreme template rejected responses: {extreme_count} ({fmt(extreme_rate)})",
        f"- recovered human rationales checked for prompt leakage: {leakage_checked}",
        f"- exact human-rationale leakage count in prompts: {leakage_count}",
        f"- low_to_high rejected validity rate: {fmt(low_valid)}",
        f"- high_to_low rejected validity rate: {fmt(high_valid)}",
        f"- need more rejection generation: `{need_generation}`",
        f"- ready_for_main_dpo: `{ready_for_main}`",
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
    lines.extend(["", "## Rejected Source Counts", "", "| source | category | risk_type | n |", "|---|---|---|---:|"])
    for row in source_rows:
        lines.append(f"| {row['rejected_source']} | {row['rejected_category']} | {row['risk_type']} | {row['n']} |")
    lines.extend(
        [
            "",
            "## Pair Quality",
            "",
            "| risk_type | n | chosen_mean | rejected_mean | gap | chosen_failure | rejected_failure | rejected_no_failure |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
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
            "## Hardness",
            "",
            "| risk_type | category | n | within_risk | overall | gap | validity |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in hardness:
        if int(row["n"]) == 0:
            continue
        lines.append(
            f"| {row['risk_type']} | {row['rejected_category']} | {row['n']} | "
            f"{fmt(row['fraction_within_risk'])} | {fmt(row['fraction_overall'])} | "
            f"{fmt(row['score_gap_mean'])} | {fmt(row['validity_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- This dataset is hybrid, not purely on-policy.",
            "- If actual model rejected rate remains below 0.20, use it only as a DPO scout/control.",
            "- If more generations raise actual rejected rate above 0.30, it can be reconsidered as the main DPO dataset.",
            "",
            "## Guardrails",
            "",
            "- The DPO prompt contains only question, answer, metric, rubric, and metadata.",
            "- Human-rationale-derived fields appear only in chosen assistant targets.",
            "- Full DPO JSON is written under gitignored `data/` and must not be committed.",
            "- Test is not read. Dev/D1 annotations are not used as training labels.",
        ]
    )
    summary = {
        "actual_model_rejected_rate": actual_rate,
        "hard_synthetic_rate": hard_rate,
        "extreme_template_rate": extreme_rate,
        "need_generation": need_generation,
        "ready_for_main_dpo": ready_for_main,
        "generation_needed_for": diagnostics.get("generation_needed_for", []),
    }
    return "\n".join(lines), ready_for_main, summary


def build(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    train_rows = read_jsonl(args.train_jsonl)
    candidates = {clean(row.get("sample_id")): row for row in read_csv_rows(args.a0_candidates)}
    high_controls = {clean(row.get("sample_id")): row for row in read_csv_rows(args.a0_high_controls)}
    pools, diagnostics = build_pair_pools(train_rows, candidates, high_controls, args.d2_dir, rng)
    pairs = sample_balanced_pairs(pools, args.total_pairs, rng)

    write_json_array(args.out_dir / DATA_FILE, pairs)
    dataset_info = {DATASET_NAME: dpo_dataset_entry(DATA_FILE)}
    if dataset_info[DATASET_NAME].get("tags") != OPENAI_TAGS:
        raise ValueError("Unexpected ShareGPT tag mapping.")
    write_json(args.out_dir / "dataset_info_r5b_snippet.json", dataset_info)

    count_rows = pair_count_rows(pools, pairs)
    source_rows = source_count_rows(pairs)
    quality_rows = pair_quality_rows(pairs)
    hardness = hardness_rows(pairs)
    write_csv(args.out_dir / "tables" / "exp19_r5b_pair_counts.csv", count_rows)
    write_csv(args.out_dir / "tables" / "exp19_r5b_rejected_source_counts.csv", source_rows)
    write_csv(args.out_dir / "tables" / "exp19_r5b_pair_quality_stats.csv", quality_rows)
    write_csv(args.out_dir / "tables" / "exp19_r5b_rejection_hardness_stats.csv", hardness)
    write_redacted_samples(args.out_dir, pairs, args.redacted_limit)
    leakage_checked, leakage_count = leakage_check(train_rows, candidates)
    report, ready, summary = report_text(
        pairs,
        count_rows,
        source_rows,
        quality_rows,
        hardness,
        leakage_checked,
        leakage_count,
        diagnostics,
    )
    write_text(args.out_dir / "reports" / "exp19_r5b_rejection_mined_dpo_qc_report.md", report)
    if summary["need_generation"]:
        write_text(args.out_dir / "reports" / "exp19_r5b_rejection_generation_plan.md", generation_plan_text(summary))
    write_dpo_config(
        args.out_dir / "configs" / "llamafactory_qwen3_4b_r5b_dpo_from_r2c.yaml",
        "saves/edubench/qwen3-4b/r2_clean_reason_score_balanced_lora",
        "saves/edubench/qwen3-4b/r5b_dpo_from_r2c_lora",
        args.out_dir,
    )
    write_dpo_config(
        args.out_dir / "configs" / "llamafactory_qwen3_4b_r5b_dpo_from_r1b.yaml",
        "saves/edubench/qwen3-4b/r1_score_only_balanced_lora",
        "saves/edubench/qwen3-4b/r5b_dpo_from_r1b_lora",
        args.out_dir,
    )
    category_counts = Counter(pair["rejected_category"] for pair in pairs)
    return {
        "pairs": len(pairs),
        "ready_for_main_dpo": ready,
        "leakage_count": leakage_count,
        "counts": {row["risk_type"]: row["expanded_count"] for row in count_rows},
        "actual_model_rejected": category_counts.get("actual_model", 0),
        "hard_synthetic_rejected": category_counts.get("hard_synthetic", 0),
        "extreme_template_rejected": category_counts.get("extreme_template", 0),
        "actual_model_rejected_rate": summary["actual_model_rejected_rate"],
        "need_generation": summary["need_generation"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Exp19-R5B rejection-mined hybrid DPO data.")
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
