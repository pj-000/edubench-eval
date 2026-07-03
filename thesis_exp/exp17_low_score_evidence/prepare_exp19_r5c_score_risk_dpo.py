"""Prepare Exp19-R5C score-risk DPO data and R5E hard-synthetic control.

R5C splits score-level risk from the broader R5B evidence-consistency signal.
Low-score rejected responses must be genuine high-score risk (score >= 4, or
score 3 with no failure and no score cap). Low-score outputs with inconsistent
failure fields are intentionally left for R5D.

The script reads train-only inputs, never reads test, and never uses dev/D1
annotations as training labels. Full DPO JSON stays under gitignored data/.
"""

from __future__ import annotations

import argparse
import csv
import json
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
    safe_int,
)
from thesis_exp.exp17_low_score_evidence.prepare_exp19_r5a_onpolicy_dpo import (  # noqa: E402
    ADAPTER_SOURCES,
    EXCLUDED_LOW_FAILURES,
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
DEFAULT_R5C_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_r5c_score_risk_dpo_seed42")
DEFAULT_R5E_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_r5e_hard_synthetic_dpo_control_seed42")

R5C_MAIN_DATASET = "edubench_r5c_score_risk_dpo_train"
R5C_MAIN_FILE = "data/edubench_r5c_score_risk_dpo_train.json"
R5C_NO_MID_DATASET = "edubench_r5c_score_risk_no_mid_dpo_train"
R5C_NO_MID_FILE = "data/edubench_r5c_score_risk_no_mid_dpo_train.json"
R5E_DATASET = "edubench_r5e_hard_synthetic_dpo_control_train"
R5E_FILE = "data/edubench_r5e_hard_synthetic_dpo_control_train.json"

RISK_TYPES = ["low_to_high_score_risk", "high_to_low_score_risk", "mid_score_calibration"]
MAIN_FRACTIONS = {
    "low_to_high_score_risk": 0.45,
    "high_to_low_score_risk": 0.45,
    "mid_score_calibration": 0.10,
}
NO_MID_FRACTIONS = {
    "low_to_high_score_risk": 0.50,
    "high_to_low_score_risk": 0.50,
}
MODEL_SOURCES = set(ADAPTER_SOURCES)


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


def source_category(source: str) -> str:
    if source in MODEL_SOURCES or source.startswith("actual_"):
        return "actual_model"
    if source.startswith("hard_synthetic"):
        return "hard_synthetic"
    if source.startswith("template_"):
        return "extreme_template"
    return "unknown"


def hard_high_rejected(_: random.Random) -> tuple[str, dict[str, Any]]:
    return (
        "hard_synthetic_score4_no_failure",
        {
            "score": 4,
            "major_failures": ["no_major_failure"],
            "score_cap": None,
            "rubric_satisfied": True,
        },
    )


def hard_conservative_rejected(rng: random.Random) -> tuple[str, dict[str, Any]]:
    if rng.random() < 0.65:
        return (
            "hard_synthetic_score3_conservative_failure",
            {
                "score": 3,
                "major_failures": ["insufficient_evidence"],
                "score_cap": 3,
                "rubric_satisfied": False,
            },
        )
    return (
        "hard_synthetic_score2_conservative_failure",
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
            "hard_synthetic_mid_high5",
            {
                "score": 5,
                "major_failures": ["no_major_failure"],
                "score_cap": None,
                "rubric_satisfied": True,
            },
        )
    return (
        "hard_synthetic_mid_low1",
        {
            "score": 1,
            "major_failures": ["missing_key_point"],
            "score_cap": 1,
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


def is_low_score_risk_rejected(target: dict[str, Any]) -> bool:
    score = score_of(target)
    if score is not None and score >= 4:
        return True
    return score == 3 and has_no_major_failure(target) and target.get("score_cap") is None


def is_high_score_risk_rejected(target: dict[str, Any]) -> bool:
    score = score_of(target)
    if score is not None and score <= 2:
        return True
    cap = safe_int(target.get("score_cap"))
    return score == 3 and has_substantive_failure(target) and cap is not None and cap <= 3


def is_mid_score_risk_rejected(target: dict[str, Any]) -> bool:
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
    hard_synthetic_only: bool = False,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    low_predictions = {} if hard_synthetic_only else load_subset_predictions(d2_dir, "train_low_reasoned_subset")
    high_predictions = {} if hard_synthetic_only else load_subset_predictions(d2_dir, "train_clean_high_subset")
    mid_predictions = {} if hard_synthetic_only else load_subset_predictions(d2_dir, "train_mid_subset")
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
                if is_low_score_risk_rejected(target)
            ]
            if actuals:
                for source, rejected in actuals:
                    add_pair_candidate(pools["low_to_high_score_risk"], row, chosen, rejected, "low_to_high_score_risk", source)
            else:
                source, rejected = hard_high_rejected(rng)
                add_pair_candidate(pools["low_to_high_score_risk"], row, chosen, rejected, "low_to_high_score_risk", source)
        elif label >= 4 and sid in high_controls:
            chosen = high_chosen(row)
            actuals = [
                (source, target)
                for source, target in high_predictions.get(sid, [])
                if is_high_score_risk_rejected(target)
            ]
            if actuals:
                for source, rejected in actuals:
                    add_pair_candidate(pools["high_to_low_score_risk"], row, chosen, rejected, "high_to_low_score_risk", source)
            else:
                source, rejected = hard_conservative_rejected(rng)
                add_pair_candidate(pools["high_to_low_score_risk"], row, chosen, rejected, "high_to_low_score_risk", source)
        elif label == 3:
            chosen = mid_chosen()
            actuals = [
                (source, target)
                for source, target in mid_predictions.get(sid, [])
                if is_mid_score_risk_rejected(target)
            ]
            if actuals:
                for source, rejected in actuals:
                    add_pair_candidate(pools["mid_score_calibration"], row, chosen, rejected, "mid_score_calibration", source)
            else:
                source, rejected = hard_mid_rejected(rng)
                add_pair_candidate(pools["mid_score_calibration"], row, chosen, rejected, "mid_score_calibration", source)

    if not hard_synthetic_only:
        actual_counts = Counter(item["risk_type"] for pool in pools.values() for item in pool if item["rejected_category"] == "actual_model")
        if actual_counts.get("low_to_high_score_risk", 0) < 300:
            diagnostics["generation_needed_for"].append("low_to_high_score_risk")
        if actual_counts.get("high_to_low_score_risk", 0) < 300:
            diagnostics["generation_needed_for"].append("high_to_low_score_risk")
        if actual_counts.get("mid_score_calibration", 0) == 0:
            diagnostics["generation_needed_for"].append("mid_score_calibration")
    return pools, diagnostics


def dpo_example(
    row: dict[str, Any],
    chosen: dict[str, Any],
    rejected: dict[str, Any],
    risk_type: str,
    rejected_source: str,
    copy_index: int,
    dataset_variant: str,
) -> dict[str, Any]:
    return {
        "messages": messages_for(row),
        "chosen": {"role": "assistant", "content": assistant_json(chosen)},
        "rejected": {"role": "assistant", "content": assistant_json(rejected)},
        "risk_type": risk_type,
        "rejected_source": rejected_source,
        "rejected_category": source_category(rejected_source),
        "source_sample_id": sample_id(row),
        "dataset_variant": dataset_variant,
        "oversample_copy_index": copy_index,
    }


def sample_balanced_pairs(
    pools: dict[str, list[dict[str, Any]]],
    total_pairs: int,
    fractions: dict[str, float],
    rng: random.Random,
    dataset_variant: str,
) -> list[dict[str, Any]]:
    targets: dict[str, int] = {}
    allocated = 0
    risk_types = list(fractions)
    for risk_type in risk_types[:-1]:
        targets[risk_type] = int(round(total_pairs * fractions[risk_type]))
        allocated += targets[risk_type]
    targets[risk_types[-1]] = total_pairs - allocated
    out: list[dict[str, Any]] = []
    for risk_type in risk_types:
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
                    dataset_variant,
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


def validity_for_pair(pair: dict[str, Any]) -> bool:
    target = target_from_message(pair["rejected"])
    risk_type = pair["risk_type"]
    if risk_type == "low_to_high_score_risk":
        return is_low_score_risk_rejected(target)
    if risk_type == "high_to_low_score_risk":
        return is_high_score_risk_rejected(target)
    if risk_type == "mid_score_calibration":
        return is_mid_score_risk_rejected(target)
    return False


def metric_rows_for_pairs(pools: dict[str, list[dict[str, Any]]], pairs_by_variant: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    count_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    validity_rows: list[dict[str, Any]] = []
    for variant, pairs in pairs_by_variant.items():
        expanded = Counter(pair["risk_type"] for pair in pairs)
        total = len(pairs)
        for risk_type in sorted(expanded):
            count_rows.append(
                {
                    "dataset_variant": variant,
                    "risk_type": risk_type,
                    "original_count": len(pools.get(risk_type, [])),
                    "expanded_count": expanded.get(risk_type, 0),
                    "actual_fraction": safe_rate(expanded.get(risk_type, 0), total),
                }
            )
        source_counts: Counter[tuple[str, str, str]] = Counter(
            (pair["rejected_source"], pair["rejected_category"], pair["risk_type"]) for pair in pairs
        )
        for (source, category, risk_type), n in sorted(source_counts.items(), key=lambda item: (item[0][2], item[0][1], item[0][0])):
            source_rows.append(
                {
                    "dataset_variant": variant,
                    "rejected_source": source,
                    "rejected_category": category,
                    "risk_type": risk_type,
                    "n": n,
                }
            )
        for risk_type in sorted(expanded):
            items = [pair for pair in pairs if pair["risk_type"] == risk_type]
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
            quality_rows.append(
                {
                    "dataset_variant": variant,
                    "risk_type": risk_type,
                    "n": n,
                    "chosen_score_mean": mean([float(x) for x in chosen_scores if x is not None]),
                    "rejected_score_mean": mean([float(x) for x in rejected_scores if x is not None]),
                    "score_gap_mean": mean([float(x) for x in gaps]),
                    "actual_model_rejected_rate": safe_rate(sum(1 for pair in items if pair["rejected_category"] == "actual_model"), n),
                    "hard_synthetic_rate": safe_rate(sum(1 for pair in items if pair["rejected_category"] == "hard_synthetic"), n),
                    "validity_rate": safe_rate(sum(1 for pair in items if validity_for_pair(pair)), n),
                }
            )
        for risk_type in sorted(expanded):
            items = [pair for pair in pairs if pair["risk_type"] == risk_type]
            rejected_targets = [target_from_message(pair["rejected"]) for pair in items]
            n = len(items)
            validity_rows.append(
                {
                    "dataset_variant": variant,
                    "risk_type": risk_type,
                    "n": n,
                    "validity_rate": safe_rate(sum(1 for pair in items if validity_for_pair(pair)), n),
                    "low_rejected_score_ge4_rate": safe_rate(
                        sum(1 for target in rejected_targets if (score_of(target) or 0) >= 4), n
                    )
                    if risk_type == "low_to_high_score_risk"
                    else "",
                    "low_rejected_score3_no_failure_capnull_rate": safe_rate(
                        sum(
                            1
                            for target in rejected_targets
                            if score_of(target) == 3 and has_no_major_failure(target) and target.get("score_cap") is None
                        ),
                        n,
                    )
                    if risk_type == "low_to_high_score_risk"
                    else "",
                    "high_rejected_score_le2_rate": safe_rate(
                        sum(1 for target in rejected_targets if (score_of(target) or 9) <= 2), n
                    )
                    if risk_type == "high_to_low_score_risk"
                    else "",
                    "high_rejected_score3_failure_cap_le3_rate": safe_rate(
                        sum(
                            1
                            for target in rejected_targets
                            if score_of(target) == 3
                            and has_substantive_failure(target)
                            and safe_int(target.get("score_cap")) is not None
                            and int(safe_int(target.get("score_cap")) or 9) <= 3
                        ),
                        n,
                    )
                    if risk_type == "high_to_low_score_risk"
                    else "",
                }
            )
    return count_rows, source_rows, quality_rows, validity_rows


def redacted_pair(pair: dict[str, Any]) -> dict[str, Any]:
    user_text = ""
    for msg in pair.get("messages") or []:
        if msg.get("role") == "user":
            user_text = clean(msg.get("content"))
            break
    return {
        "sample_id": pair.get("source_sample_id", ""),
        "prompt_hash": sha1(user_text),
        "prompt_preview": " ".join(user_text.split())[:180],
        "chosen_redacted": redacted_target(target_from_message(pair["chosen"])),
        "rejected_redacted": redacted_target(target_from_message(pair["rejected"])),
        "dataset_variant": pair.get("dataset_variant", ""),
        "risk_type": pair.get("risk_type", ""),
        "rejected_source": pair.get("rejected_source", ""),
        "rejected_category": pair.get("rejected_category", ""),
    }


def write_redacted_samples(out_dir: Path, pairs_by_variant: dict[str, list[dict[str, Any]]], limit: int) -> None:
    redacted_dir = out_dir / "redacted_samples"
    for variant, pairs in pairs_by_variant.items():
        for risk_type in RISK_TYPES:
            selected = [pair for pair in pairs if pair["risk_type"] == risk_type][:limit]
            if selected:
                write_json(redacted_dir / f"{variant}_{risk_type}_redacted.json", [redacted_pair(pair) for pair in selected])


def write_dpo_config(path: Path, dataset_name: str, adapter_path: str, output_dir: str, dataset_dir: Path, max_steps: int | None = None) -> None:
    mapping: dict[str, Any] = {
        "model_name_or_path": MODEL_PATH,
        "trust_remote_code": True,
        "stage": "dpo",
        "do_train": True,
        "finetuning_type": "lora",
        "adapter_name_or_path": adapter_path,
        "ref_model": MODEL_PATH,
        "ref_model_adapters": adapter_path,
        "dataset_dir": str(dataset_dir),
        "dataset": dataset_name,
        "template": "qwen3_nothink",
        "cutoff_len": 4096,
        "overwrite_cache": True,
        "preprocessing_num_workers": 16,
        "pref_loss": "sigmoid",
        "pref_beta": 0.1,
        "pref_ftx": 0.1,
        "output_dir": output_dir,
        "logging_steps": 10,
        "save_steps": 100 if max_steps is None else 999999,
        "save_only_model": True,
        "plot_loss": True,
        "report_to": "none",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": "5.0e-6",
        "bf16": True,
        "gradient_checkpointing": True,
    }
    if max_steps is None:
        mapping["num_train_epochs"] = 1
    else:
        mapping["max_steps"] = max_steps
        mapping["pref_beta"] = 0.05
    write_text(path, "\n".join(f"{key}: {yaml_scalar(value)}" for key, value in mapping.items()) + "\n")


def generation_plan_text(generation_needed_for: list[str]) -> str:
    lines = [
        "# Exp19-R5C Generation Mining Plan",
        "",
        "R5C score-risk pairs should be strengthened with additional train-only candidate generations.",
        "",
        "## Needed Risk Types",
        "",
    ]
    if generation_needed_for:
        lines.extend(f"- {item}" for item in generation_needed_for)
    else:
        lines.append("- No required risk type is below the current threshold.")
    lines.extend(
        [
            "",
            "## Subsets",
            "",
            "- train low reasoned plus all train y<=2",
            "- train clean high controls plus stratified train y>=4",
            "- train y=3",
            "",
            "## Adapters",
            "",
            "- r1b_score_only_balanced",
            "- r2n_reason_score_natural",
            "- r2c_clean_reason_score_balanced",
            "- r4b_shuffled_reason_balanced",
            "",
            "## Generation",
            "",
            "- temperature: 0.7",
            "- top_p: 0.9",
            "- max_new_tokens: 256",
            "- num_samples_per_input: 4",
            "",
            "## Filters",
            "",
            "- low: score >= 4, or score == 3 with no_major_failure and score_cap null",
            "- high: score <= 2, or score == 3 with false failure and score_cap <= 3",
            "- mid: score <= 1 or score >= 5",
            "",
            "Do not read test or use dev/D1 annotations for training.",
        ]
    )
    return "\n".join(lines)


def report_text(
    title: str,
    pairs_by_variant: dict[str, list[dict[str, Any]]],
    count_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    quality_rows: list[dict[str, Any]],
    validity_rows: list[dict[str, Any]],
    leakage_checked: int,
    leakage_count: int,
    generation_needed_for: list[str],
    hard_synthetic_only: bool,
) -> tuple[str, dict[str, bool]]:
    ready_by_variant: dict[str, bool] = {}
    lines = [
        f"# {title}",
        "",
        "This train-only DPO dataset isolates score-level risk. Human rationale is never included in the user prompt.",
        "",
        "## Summary",
        "",
    ]
    for variant, pairs in pairs_by_variant.items():
        categories = Counter(pair["rejected_category"] for pair in pairs)
        low_pairs = [pair for pair in pairs if pair["risk_type"] == "low_to_high_score_risk"]
        high_pairs = [pair for pair in pairs if pair["risk_type"] == "high_to_low_score_risk"]
        validity = safe_rate(sum(1 for pair in pairs if validity_for_pair(pair)), len(pairs))
        extreme_rate = safe_rate(categories.get("extreme_template", 0), len(pairs))
        ready = (
            len(low_pairs) >= 300
            and len(high_pairs) >= 300
            and validity >= 0.95
            and extreme_rate == 0
            and leakage_count == 0
        )
        ready_by_variant[variant] = ready
        lines.extend(
            [
                f"- {variant} total pairs: {len(pairs)}",
                f"- {variant} actual model rejected: {categories.get('actual_model', 0)} ({fmt(safe_rate(categories.get('actual_model', 0), len(pairs)))})",
                f"- {variant} hard synthetic rejected: {categories.get('hard_synthetic', 0)} ({fmt(safe_rate(categories.get('hard_synthetic', 0), len(pairs)))})",
                f"- {variant} extreme template rejected: {categories.get('extreme_template', 0)} ({fmt(extreme_rate)})",
                f"- {variant} score-risk validity: {fmt(validity)}",
                f"- {variant} ready_for_main_score_risk_dpo: `{ready}`",
            ]
        )
    lines.extend(
        [
            f"- recovered human rationales checked for prompt leakage: {leakage_checked}",
            f"- exact human-rationale leakage count in prompts: {leakage_count}",
            f"- hard synthetic control mode: `{hard_synthetic_only}`",
            f"- generation mining needed: `{bool(generation_needed_for)}`",
            "",
            "## Pair Counts",
            "",
            "| variant | risk_type | original | expanded | actual_fraction |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in count_rows:
        lines.append(
            f"| {row['dataset_variant']} | {row['risk_type']} | {row['original_count']} | "
            f"{row['expanded_count']} | {fmt(row['actual_fraction'])} |"
        )
    lines.extend(["", "## Rejected Sources", "", "| variant | source | category | risk_type | n |", "|---|---|---|---|---:|"])
    for row in source_rows:
        lines.append(
            f"| {row['dataset_variant']} | {row['rejected_source']} | {row['rejected_category']} | "
            f"{row['risk_type']} | {row['n']} |"
        )
    lines.extend(
        [
            "",
            "## Pair Quality",
            "",
            "| variant | risk_type | n | chosen_mean | rejected_mean | gap | actual_rate | hard_rate | validity |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in quality_rows:
        lines.append(
            f"| {row['dataset_variant']} | {row['risk_type']} | {row['n']} | "
            f"{fmt(row['chosen_score_mean'])} | {fmt(row['rejected_score_mean'])} | "
            f"{fmt(row['score_gap_mean'])} | {fmt(row['actual_model_rejected_rate'])} | "
            f"{fmt(row['hard_synthetic_rate'])} | {fmt(row['validity_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- R5C_main uses 45/45/10 low/high/mid score-risk ratios.",
            "- R5C_no_mid uses 50/50 low/high score-risk ratios.",
            "- R5E hard-synthetic control uses the same ratio logic without actual model rejected responses.",
            "- Full DPO JSON is gitignored; only QC artifacts should be committed.",
        ]
    )
    return "\n".join(lines), ready_by_variant


def build(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    hard_synthetic_only = args.mode == "hard_synthetic_control"
    out_dir = args.out_dir or (DEFAULT_R5E_OUT_DIR if hard_synthetic_only else DEFAULT_R5C_OUT_DIR)
    train_rows = read_jsonl(args.train_jsonl)
    candidates = {clean(row.get("sample_id")): row for row in read_csv_rows(args.a0_candidates)}
    high_controls = {clean(row.get("sample_id")): row for row in read_csv_rows(args.a0_high_controls)}
    pools, diagnostics = build_pair_pools(train_rows, candidates, high_controls, args.d2_dir, rng, hard_synthetic_only)

    pairs_by_variant: dict[str, list[dict[str, Any]]] = {}
    dataset_info: dict[str, Any] = {}
    if hard_synthetic_only:
        pairs_by_variant["r5e_main"] = sample_balanced_pairs(pools, args.total_pairs, MAIN_FRACTIONS, rng, "r5e_main")
        write_json_array(out_dir / R5E_FILE, pairs_by_variant["r5e_main"])
        dataset_info[R5E_DATASET] = dpo_dataset_entry(R5E_FILE)
    else:
        pairs_by_variant["r5c_main"] = sample_balanced_pairs(pools, args.total_pairs, MAIN_FRACTIONS, rng, "r5c_main")
        pairs_by_variant["r5c_no_mid"] = sample_balanced_pairs(pools, args.total_pairs, NO_MID_FRACTIONS, rng, "r5c_no_mid")
        write_json_array(out_dir / R5C_MAIN_FILE, pairs_by_variant["r5c_main"])
        write_json_array(out_dir / R5C_NO_MID_FILE, pairs_by_variant["r5c_no_mid"])
        dataset_info[R5C_MAIN_DATASET] = dpo_dataset_entry(R5C_MAIN_FILE)
        dataset_info[R5C_NO_MID_DATASET] = dpo_dataset_entry(R5C_NO_MID_FILE)
    if any(value.get("tags") != OPENAI_TAGS for value in dataset_info.values()):
        raise ValueError("Unexpected ShareGPT tag mapping.")
    snippet_name = "dataset_info_r5e_snippet.json" if hard_synthetic_only else "dataset_info_r5c_snippet.json"
    write_json(out_dir / snippet_name, dataset_info)

    count_rows, source_rows, quality_rows, validity_rows = metric_rows_for_pairs(pools, pairs_by_variant)
    prefix = "r5e" if hard_synthetic_only else "r5c"
    write_csv(out_dir / "tables" / f"{prefix}_pair_counts.csv", count_rows)
    write_csv(out_dir / "tables" / f"{prefix}_rejected_source_counts.csv", source_rows)
    write_csv(out_dir / "tables" / f"{prefix}_pair_quality_stats.csv", quality_rows)
    write_csv(out_dir / "tables" / f"{prefix}_score_risk_validity.csv", validity_rows)
    write_redacted_samples(out_dir, pairs_by_variant, args.redacted_limit)
    leakage_checked, leakage_count = leakage_check(train_rows, candidates)
    generation_needed_for = [] if hard_synthetic_only else diagnostics.get("generation_needed_for", [])
    title = (
        "Exp19-R5E Hard-Synthetic DPO Control QC Report"
        if hard_synthetic_only
        else "Exp19-R5C Score-Risk DPO QC Report"
    )
    report, ready_by_variant = report_text(
        title,
        pairs_by_variant,
        count_rows,
        source_rows,
        quality_rows,
        validity_rows,
        leakage_checked,
        leakage_count,
        generation_needed_for,
        hard_synthetic_only,
    )
    report_name = "r5e_hard_synthetic_dpo_control_qc_report.md" if hard_synthetic_only else "r5c_score_risk_dpo_qc_report.md"
    write_text(out_dir / "reports" / report_name, report)
    if generation_needed_for:
        write_text(out_dir / "reports" / "exp19_r5c_generation_mining_plan.md", generation_plan_text(generation_needed_for))

    main_dataset = R5E_DATASET if hard_synthetic_only else R5C_MAIN_DATASET
    run_prefix = "r5e_dpo_control" if hard_synthetic_only else "r5c_dpo"
    write_dpo_config(
        out_dir / "configs" / f"llamafactory_qwen3_4b_{run_prefix}_from_r2c.yaml",
        main_dataset,
        "saves/edubench/qwen3-4b/r2_clean_reason_score_balanced_lora",
        f"saves/edubench/qwen3-4b/{run_prefix}_from_r2c_lora",
        out_dir,
    )
    write_dpo_config(
        out_dir / "configs" / f"llamafactory_qwen3_4b_{run_prefix}_from_r1b.yaml",
        main_dataset,
        "saves/edubench/qwen3-4b/r1_score_only_balanced_lora",
        f"saves/edubench/qwen3-4b/{run_prefix}_from_r1b_lora",
        out_dir,
    )
    return {
        "mode": args.mode,
        "pairs": {variant: len(pairs) for variant, pairs in pairs_by_variant.items()},
        "ready": ready_by_variant,
        "leakage_count": leakage_count,
        "generation_needed_for": generation_needed_for,
        "actual_model_rejected": {
            variant: sum(1 for pair in pairs if pair["rejected_category"] == "actual_model")
            for variant, pairs in pairs_by_variant.items()
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Exp19-R5C score-risk DPO or R5E control data.")
    parser.add_argument("--mode", choices=["score_risk", "hard_synthetic_control"], default="score_risk")
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--a0-candidates", type=Path, default=DEFAULT_A0_CANDIDATES)
    parser.add_argument("--a0-high-controls", type=Path, default=DEFAULT_A0_HIGH_CONTROLS)
    parser.add_argument("--d2-dir", type=Path, default=DEFAULT_D2_DIR)
    parser.add_argument("--out-dir", type=Path)
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
