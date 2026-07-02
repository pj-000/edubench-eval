"""Prepare Exp19-S0 reason-aware SFT and DPO datasets for Qwen3-4B.

This script only builds training/evaluation artifacts. It does not train a
model, does not read the test split, and keeps full raw datasets under the
gitignored Exp17/Exp19 outputs directory by default.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.src.edujudge.utils.io import write_csv, write_json, write_text


DEFAULT_TRAIN_PATH = Path("thesis_exp/data/splits/question_seed42/train.jsonl")
DEFAULT_DEV_PATH = Path("thesis_exp/data/splits/question_seed42/dev.jsonl")
DEFAULT_A0_CANDIDATES = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp17_a0_train_hidden_failure_signals_seed42/"
    "train_hidden_failure_candidates.csv"
)
DEFAULT_A0_HIGH_CONTROLS = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp17_a0_train_hidden_failure_signals_seed42/"
    "train_clean_high_controls.csv"
)
DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_sft_dpo_datasets_seed42")

MODEL_PATH = "/home/jpang/models/modelscope/Qwen/Qwen3-4B"
SYSTEM_PROMPT = "You are an educational assessment evaluator. Return only valid JSON."
DATASET_NAMES = {
    "r1": "edubench_r1_score_only_train",
    "r2": "edubench_r2_reason_score_train",
    "r3": "edubench_r3_reason_rationale_train",
    "r4": "edubench_r4_shuffled_reason_control_train",
    "r2_balanced": "edubench_r2_reason_score_balanced_train",
    "r3_balanced": "edubench_r3_reason_rationale_balanced_train",
    "r2_clean_balanced": "edubench_r2_clean_reason_score_balanced_train",
    "r5": "edubench_r5_risk_balanced_dpo_train",
    "r5_high_control": "edubench_r5_high_protection_dpo_control_train",
}
OPENAI_TAGS = {
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant",
    "system_tag": "system",
}
FAILURE_VOCAB = {
    "missing_key_point",
    "factual_or_rubric_mismatch",
    "answer_key_or_reference_mismatch",
    "surface_fluent_but_hidden_defect",
    "insufficient_evidence",
    "partial_or_incomplete",
    "task_constraint_violation",
    "format_violation",
    "possible_label_conflict",
    "no_major_failure",
    "unclear",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def clamp_score(value: Any) -> int:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        score = 3
    return min(5, max(1, score))


def sample_id(row: dict[str, Any]) -> str:
    return clean(row.get("sample_id") or row.get("record_id") or row.get("id"))


def question_group(row: dict[str, Any]) -> str:
    return clean(row.get("question_group_id") or row.get("question_key") or row.get("question_id"))


def metric_name(row: dict[str, Any]) -> str:
    return clean(row.get("metric") or row.get("metric_canonical") or row.get("metric_raw") or row.get("metric_abbr"))


def language(row: dict[str, Any]) -> str:
    return clean(row.get("language") or "unknown") or "unknown"


def subject(row: dict[str, Any]) -> str:
    return clean(row.get("subject") or row.get("subject_canonical") or row.get("subject_raw") or "unknown") or "unknown"


def metadata_text(row: dict[str, Any]) -> str:
    fields = {
        "language": language(row),
        "subject": subject(row),
        "education_level": clean(row.get("education_level_canonical") or row.get("education_level_raw")),
        "scenario": clean(row.get("scenario_canonical") or row.get("scenario_raw")),
        "metric_group": clean(row.get("metric_group")),
    }
    raw_metadata = clean(row.get("metadata") or row.get("metadata_raw"))
    if raw_metadata:
        fields["metadata_raw"] = raw_metadata
    return json.dumps({k: v for k, v in fields.items() if v}, ensure_ascii=False, sort_keys=True)


def user_content(row: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            f"Question:\n{clean(row.get('question'))}",
            f"Answer:\n{clean(row.get('answer'))}",
            f"Evaluation metric:\n{metric_name(row)}",
            f"Rubric:\n{clean(row.get('rubric'))}",
            f"Metadata:\n{metadata_text(row)}",
        ]
    )


def messages_for(row: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content(row)},
    ]


def assistant_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sft_example(row: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    return {"messages": messages_for(row) + [{"role": "assistant", "content": assistant_json(target)}]}


def normalized_failure_mode(value: str) -> str:
    mode = clean(value)
    if not mode:
        return "unclear"
    return mode if mode in FAILURE_VOCAB else "unclear"


def a0_signal_type(row: dict[str, str] | None) -> str:
    if row is None:
        return ""
    return clean(row.get("recommended_training_use") or row.get("hidden_failure_candidate_type"))


def recovered_reason(row: dict[str, str] | None) -> str:
    if row is None:
        return ""
    return clean(row.get("recovered_reason_summary"))


def build_signal_maps(
    candidates_path: Path,
    high_controls_path: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    candidates = {clean(row.get("sample_id")): row for row in read_csv_rows(candidates_path)}
    high_controls = {clean(row.get("sample_id")): row for row in read_csv_rows(high_controls_path)}
    return candidates, high_controls


def reason_for_target(label: int, failures: list[str], a0_row: dict[str, str] | None) -> str:
    reason = recovered_reason(a0_row)
    if reason:
        return reason
    if label <= 2:
        if failures and failures != ["unclear"]:
            return f"The answer receives a low score because the main rubric-linked issue is {failures[0]}."
        return "The answer receives a low score, but no reliable recovered human rationale is available."
    if label == 3:
        return "The answer partially satisfies the rubric, but some requirements remain incomplete."
    return "The answer satisfies the rubric well enough that no major failure is evident."


def evidence_target(row: dict[str, Any], a0_row: dict[str, str] | None, include_reason: bool) -> dict[str, Any]:
    label = clamp_score(row.get("label_5"))
    signal_type = a0_signal_type(a0_row)
    failure_mode = normalized_failure_mode(a0_row.get("failure_mode_auto", "") if a0_row else "")
    if label >= 4:
        failures = ["no_major_failure"]
        score_cap: int | None = None
        rubric_satisfied = True
    elif label == 3:
        failures = ["partial_or_incomplete"]
        score_cap = 3
        rubric_satisfied = False
    else:
        if signal_type in {"weak_evidence_positive", "pairwise_low", "format_auxiliary"}:
            failures = [failure_mode]
        elif signal_type == "review_only" and failure_mode != "unclear":
            failures = [failure_mode]
        else:
            failures = ["unclear"]
        score_cap = min(2, label)
        rubric_satisfied = False
    target: dict[str, Any] = {
        "score": label,
        "major_failures": failures,
        "score_cap": score_cap,
        "rubric_satisfied": rubric_satisfied,
    }
    if include_reason:
        target["brief_reason"] = reason_for_target(label, failures, a0_row)
    return target


def score_only_target(row: dict[str, Any]) -> dict[str, int]:
    return {"score": clamp_score(row.get("label_5"))}


def dpo_chosen_rejected(row: dict[str, Any], a0_row: dict[str, str] | None) -> tuple[dict[str, Any], dict[str, Any], str, str, int]:
    label = clamp_score(row.get("label_5"))
    if label <= 2:
        chosen = evidence_target(row, a0_row, include_reason=False)
        chosen["score"] = label
        chosen["score_cap"] = min(2, label)
        rejected = {
            "score": 5,
            "major_failures": ["no_major_failure"],
            "score_cap": None,
            "rubric_satisfied": True,
        }
        return chosen, rejected, "low_to_high", "high", 3
    if label >= 4:
        chosen = {
            "score": label,
            "major_failures": ["no_major_failure"],
            "score_cap": None,
            "rubric_satisfied": True,
        }
        rejected = {
            "score": 2,
            "major_failures": ["missing_key_point"],
            "score_cap": 2,
            "rubric_satisfied": False,
        }
        return chosen, rejected, "high_to_low_protection", "high", 2
    chosen = {
        "score": 3,
        "major_failures": ["partial_or_incomplete"],
        "score_cap": 3,
        "rubric_satisfied": False,
    }
    rejected = {
        "score": 5,
        "major_failures": ["no_major_failure"],
        "score_cap": None,
        "rubric_satisfied": True,
    }
    return chosen, rejected, "mid_score_calibration", "standard", 1


def dpo_example(
    row: dict[str, Any],
    chosen: dict[str, Any],
    rejected: dict[str, Any],
    risk_type: str,
    weight_bucket: str,
    copy_index: int,
) -> dict[str, Any]:
    return {
        "messages": messages_for(row),
        "chosen": {"role": "assistant", "content": assistant_json(chosen)},
        "rejected": {"role": "assistant", "content": assistant_json(rejected)},
        "risk_type": risk_type,
        "weight_bucket": weight_bucket,
        "oversample_copy_index": copy_index,
    }


def is_low(row: dict[str, Any]) -> bool:
    return clamp_score(row.get("label_5")) <= 2


def is_mid(row: dict[str, Any]) -> bool:
    return clamp_score(row.get("label_5")) == 3


def is_high(row: dict[str, Any]) -> bool:
    return clamp_score(row.get("label_5")) >= 4


def is_clean_low_target(row: dict[str, Any], target: dict[str, Any]) -> bool:
    if not is_low(row):
        return False
    failures = target.get("major_failures") or []
    return failures not in ([], ["unclear"], ["possible_label_conflict"])


def sample_indices(rng: random.Random, pool: list[int], count: int) -> list[int]:
    if not pool:
        return []
    if count <= len(pool):
        return rng.sample(pool, count)
    return [rng.choice(pool) for _ in range(count)]


def build_balanced_indices(
    rows: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    rng: random.Random,
    total: int,
    clean_low_only: bool,
) -> list[int]:
    if clean_low_only:
        low_pool = [idx for idx, (row, target) in enumerate(zip(rows, targets)) if is_clean_low_target(row, target)]
    else:
        low_pool = [idx for idx, row in enumerate(rows) if is_low(row)]
    mid_pool = [idx for idx, row in enumerate(rows) if is_mid(row)]
    high_pool = [idx for idx, row in enumerate(rows) if is_high(row)]
    low_n = int(round(total * 0.30))
    mid_n = int(round(total * 0.175))
    high_n = total - low_n - mid_n
    indices = (
        sample_indices(rng, low_pool, low_n)
        + sample_indices(rng, mid_pool, mid_n)
        + sample_indices(rng, high_pool, high_n)
    )
    rng.shuffle(indices)
    return indices


def build_sft_from_indices(
    rows: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    indices: list[int],
) -> list[dict[str, Any]]:
    return [sft_example(rows[idx], targets[idx]) for idx in indices]


def build_dpo_high_protection_control(
    rows: list[dict[str, Any]],
    prefs: list[tuple[dict[str, Any], dict[str, Any], str, str, int]],
) -> tuple[list[dict[str, Any]], Counter[str], Counter[str]]:
    original_counts: Counter[str] = Counter()
    expanded_counts: Counter[str] = Counter()
    out: list[dict[str, Any]] = []
    for row, (chosen, rejected, risk_type, weight_bucket, copies) in zip(rows, prefs):
        original_counts.update([risk_type])
        for copy_index in range(copies):
            out.append(dpo_example(row, chosen, rejected, risk_type, weight_bucket, copy_index))
            expanded_counts.update([risk_type])
    return out, original_counts, expanded_counts


def build_dpo_risk_balanced(
    rows: list[dict[str, Any]],
    prefs: list[tuple[dict[str, Any], dict[str, Any], str, str, int]],
    rng: random.Random,
    total: int,
) -> tuple[list[dict[str, Any]], Counter[str], Counter[str]]:
    original_counts: Counter[str] = Counter()
    groups: dict[str, list[int]] = {"low_to_high": [], "high_to_low_protection": [], "mid_score_calibration": []}
    for idx, (_, _, risk_type, _, _) in enumerate(prefs):
        original_counts.update([risk_type])
        groups.setdefault(risk_type, []).append(idx)

    target_counts = {
        "low_to_high": int(round(total * 0.40)),
        "high_to_low_protection": int(round(total * 0.40)),
    }
    target_counts["mid_score_calibration"] = total - target_counts["low_to_high"] - target_counts["high_to_low_protection"]

    out: list[dict[str, Any]] = []
    expanded_counts: Counter[str] = Counter()
    for risk_type, count in target_counts.items():
        selected = sample_indices(rng, groups.get(risk_type, []), count)
        for copy_index, idx in enumerate(selected):
            chosen, rejected, _, weight_bucket, _ = prefs[idx]
            out.append(dpo_example(rows[idx], chosen, rejected, risk_type, weight_bucket, copy_index))
            expanded_counts.update([risk_type])
    rng.shuffle(out)
    return out, original_counts, expanded_counts


def write_json_array(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def truncate(text: str, limit: int = 160) -> str:
    text = " ".join(clean(text).split())
    return text[:limit] + ("..." if len(text) > limit else "")


def redacted_input(row: dict[str, Any]) -> dict[str, Any]:
    question = clean(row.get("question"))
    answer = clean(row.get("answer"))
    rubric = clean(row.get("rubric"))
    return {
        "sample_id": sample_id(row),
        "question_group": question_group(row),
        "question_hash": sha1(question),
        "answer_hash": sha1(answer),
        "rubric_hash": sha1(rubric),
        "question_preview": truncate(question, 120),
        "answer_preview": truncate(answer, 120),
        "metric": metric_name(row),
        "language": language(row),
        "subject": subject(row),
        "gold_label": clamp_score(row.get("label_5")),
    }


def redacted_target(target: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in target.items() if k != "brief_reason"}
    if "brief_reason" in target:
        reason = clean(target["brief_reason"])
        out["brief_reason_hash"] = sha1(reason)
        out["brief_reason_present"] = bool(reason)
    return out


def redacted_sft_sample(rows: list[dict[str, Any]], targets: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return [
        {"input_redacted": redacted_input(row), "target_redacted": redacted_target(target)}
        for row, target in list(zip(rows, targets))[:limit]
    ]


def redacted_dpo_sample(
    rows: list[dict[str, Any]],
    prefs: list[tuple[dict[str, Any], dict[str, Any], str, str, int]],
    limit: int,
) -> list[dict[str, Any]]:
    out = []
    for row, (chosen, rejected, risk_type, weight_bucket, copies) in list(zip(rows, prefs))[:limit]:
        out.append(
            {
                "input_redacted": redacted_input(row),
                "chosen_redacted": redacted_target(chosen),
                "rejected_redacted": redacted_target(rejected),
                "risk_type": risk_type,
                "weight_bucket": weight_bucket,
                "oversample_copies": copies,
            }
        )
    return out


def redacted_sft_filtered_sample(
    rows: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    limit: int,
    predicate,
) -> list[dict[str, Any]]:
    out = []
    for row, target in zip(rows, targets):
        if predicate(row, target):
            out.append({"input_redacted": redacted_input(row), "target_redacted": redacted_target(target)})
        if len(out) >= limit:
            break
    return out


def redacted_dpo_filtered_sample(
    rows: list[dict[str, Any]],
    prefs: list[tuple[dict[str, Any], dict[str, Any], str, str, int]],
    limit: int,
    risk_type_filter: str,
) -> list[dict[str, Any]]:
    out = []
    for row, (chosen, rejected, risk_type, weight_bucket, copies) in zip(rows, prefs):
        if risk_type != risk_type_filter:
            continue
        out.append(
            {
                "input_redacted": redacted_input(row),
                "chosen_redacted": redacted_target(chosen),
                "rejected_redacted": redacted_target(rejected),
                "risk_type": risk_type,
                "weight_bucket": weight_bucket,
                "oversample_copies_in_control": copies,
            }
        )
        if len(out) >= limit:
            break
    return out


def sft_dataset_entry(file_name: str) -> dict[str, Any]:
    return {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {"messages": "messages"},
        "tags": OPENAI_TAGS,
    }


def dpo_dataset_entry(file_name: str) -> dict[str, Any]:
    return {
        "file_name": file_name,
        "formatting": "sharegpt",
        "ranking": True,
        "columns": {"messages": "messages", "chosen": "chosen", "rejected": "rejected"},
        "tags": OPENAI_TAGS,
    }


def dataset_info_snippet() -> dict[str, Any]:
    return {
        DATASET_NAMES["r1"]: sft_dataset_entry("data/edubench_r1_score_only_train.json"),
        DATASET_NAMES["r2"]: sft_dataset_entry("data/edubench_r2_reason_score_train.json"),
        DATASET_NAMES["r3"]: sft_dataset_entry("data/edubench_r3_reason_rationale_train.json"),
        DATASET_NAMES["r4"]: sft_dataset_entry("data/edubench_r4_shuffled_reason_control_train.json"),
        DATASET_NAMES["r2_balanced"]: sft_dataset_entry("data/edubench_r2_reason_score_balanced_train.json"),
        DATASET_NAMES["r3_balanced"]: sft_dataset_entry("data/edubench_r3_reason_rationale_balanced_train.json"),
        DATASET_NAMES["r2_clean_balanced"]: sft_dataset_entry("data/edubench_r2_clean_reason_score_balanced_train.json"),
        DATASET_NAMES["r5"]: dpo_dataset_entry("data/edubench_r5_risk_balanced_dpo_train.json"),
        DATASET_NAMES["r5_high_control"]: dpo_dataset_entry("data/edubench_r5_high_protection_dpo_control_train.json"),
    }


def yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def yaml_from_mapping(mapping: dict[str, Any]) -> str:
    return "\n".join(f"{key}: {yaml_scalar(value)}" for key, value in mapping.items()) + "\n"


def sft_config(dataset_name: str, output_subdir: str, dataset_dir: Path) -> str:
    return yaml_from_mapping(
        {
            "model_name_or_path": MODEL_PATH,
            "trust_remote_code": True,
            "stage": "sft",
            "do_train": True,
            "finetuning_type": "lora",
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_target": "all",
            "dataset_dir": str(dataset_dir),
            "dataset": dataset_name,
            "template": "qwen3_nothink",
            "cutoff_len": 4096,
            "overwrite_cache": True,
            "preprocessing_num_workers": 16,
            "output_dir": f"saves/edubench/qwen3-4b/{output_subdir}",
            "logging_steps": 10,
            "save_steps": 100,
            "save_only_model": True,
            "plot_loss": True,
            "report_to": "none",
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": "1.0e-4",
            "num_train_epochs": 3,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.05,
            "bf16": True,
            "gradient_checkpointing": True,
        }
    )


def dpo_config(dataset_name: str, output_subdir: str, dataset_dir: Path) -> str:
    return yaml_from_mapping(
        {
            "model_name_or_path": MODEL_PATH,
            "trust_remote_code": True,
            "stage": "dpo",
            "do_train": True,
            "finetuning_type": "lora",
            "adapter_name_or_path": "saves/edubench/qwen3-4b/r2_reason_score_lora",
            "ref_model": MODEL_PATH,
            "ref_model_adapters": "saves/edubench/qwen3-4b/r2_reason_score_lora",
            "dataset_dir": str(dataset_dir),
            "dataset": dataset_name,
            "template": "qwen3_nothink",
            "cutoff_len": 4096,
            "overwrite_cache": True,
            "preprocessing_num_workers": 16,
            "pref_loss": "sigmoid",
            "pref_beta": 0.1,
            "pref_ftx": 0.1,
            "output_dir": f"saves/edubench/qwen3-4b/{output_subdir}",
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
    )


def write_configs(config_dir: Path, dataset_dir: Path) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    write_text(
        config_dir / "llamafactory_qwen3_4b_r1_score_only_lora.yaml",
        sft_config(DATASET_NAMES["r1"], "r1_score_only_lora", dataset_dir),
    )
    write_text(
        config_dir / "llamafactory_qwen3_4b_r2_reason_score_lora.yaml",
        sft_config(DATASET_NAMES["r2"], "r2_reason_score_lora", dataset_dir),
    )
    write_text(
        config_dir / "llamafactory_qwen3_4b_r3_reason_rationale_lora.yaml",
        sft_config(DATASET_NAMES["r3"], "r3_reason_rationale_lora", dataset_dir),
    )
    write_text(
        config_dir / "llamafactory_qwen3_4b_r4_shuffled_reason_lora.yaml",
        sft_config(DATASET_NAMES["r4"], "r4_shuffled_reason_lora", dataset_dir),
    )
    write_text(
        config_dir / "llamafactory_qwen3_4b_r2_reason_score_balanced_lora.yaml",
        sft_config(DATASET_NAMES["r2_balanced"], "r2_reason_score_balanced_lora", dataset_dir),
    )
    write_text(
        config_dir / "llamafactory_qwen3_4b_r3_reason_rationale_balanced_lora.yaml",
        sft_config(DATASET_NAMES["r3_balanced"], "r3_reason_rationale_balanced_lora", dataset_dir),
    )
    write_text(
        config_dir / "llamafactory_qwen3_4b_r2_clean_reason_score_balanced_lora.yaml",
        sft_config(DATASET_NAMES["r2_clean_balanced"], "r2_clean_reason_score_balanced_lora", dataset_dir),
    )
    write_text(
        config_dir / "llamafactory_qwen3_4b_r5_risk_balanced_dpo.yaml",
        dpo_config(DATASET_NAMES["r5"], "r5_risk_balanced_dpo", dataset_dir),
    )
    write_text(
        config_dir / "llamafactory_qwen3_4b_r5_high_protection_dpo_control.yaml",
        dpo_config(DATASET_NAMES["r5_high_control"], "r5_high_protection_dpo_control", dataset_dir),
    )


def check_reason_leakage(rows: list[dict[str, Any]], candidates: dict[str, dict[str, str]]) -> tuple[int, int]:
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


def distribution(rows: list[dict[str, Any]], key_fn) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter.update([key_fn(row)])
    return counter


def max_rate(counter: Counter[str]) -> float:
    total = sum(counter.values())
    return max(counter.values()) / total if total else 0.0


def write_counter_rows(path: Path, metric: str, counter: Counter[str]) -> None:
    rows = [{"metric": metric, "value": key, "count": count} for key, count in counter.most_common()]
    write_csv(path, rows, fieldnames=["metric", "value", "count"])


def build_qc_report(
    out_dir: Path,
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    r2_targets: list[dict[str, Any]],
    r3_targets: list[dict[str, Any]],
    r4_targets: list[dict[str, Any]],
    candidates: dict[str, dict[str, str]],
    high_controls: dict[str, dict[str, str]],
    dataset_rows: list[dict[str, Any]],
    dpo_count_sets: dict[str, tuple[Counter[str], Counter[str]]],
    sft_label_count_sets: dict[str, Counter[str]],
) -> dict[str, Any]:
    label_counts = distribution(train_rows, lambda r: str(clamp_score(r.get("label_5"))))
    language_counts = distribution(train_rows, language)
    metric_counts = distribution(train_rows, metric_name)
    qgroup_counts = distribution(train_rows, question_group)
    failures: Counter[str] = Counter()
    score_caps: Counter[str] = Counter()
    rubric_satisfied: Counter[str] = Counter()
    for target in r2_targets:
        failures.update(target.get("major_failures") or [""])
        score_caps.update([str(target.get("score_cap"))])
        rubric_satisfied.update([str(target.get("rubric_satisfied"))])
    low_count = sum(1 for row in train_rows if clamp_score(row.get("label_5")) <= 2)
    high_count = sum(1 for row in train_rows if clamp_score(row.get("label_5")) >= 4)
    high_protection_count = sum(
        1
        for target in r2_targets
        if target.get("score", 0) >= 4
        and target.get("major_failures") == ["no_major_failure"]
        and target.get("score_cap") is None
        and target.get("rubric_satisfied") is True
    )
    checked_reasons, leakage_count = check_reason_leakage(train_rows, candidates)
    matched_candidates = sum(1 for row in train_rows if sample_id(row) in candidates)
    matched_high_controls = sum(1 for row in train_rows if sample_id(row) in high_controls)
    usable_failure_targets = sum(
        1
        for row, target in zip(train_rows, r2_targets)
        if clamp_score(row.get("label_5")) <= 2 and target.get("major_failures") not in (["unclear"], [])
    )
    safe_for_sft = (
        leakage_count == 0
        and len(train_rows) > 0
        and high_protection_count >= 100
        and usable_failure_targets >= 50
        and max_rate(qgroup_counts) <= 0.5
    )
    main_dpo_original, main_dpo_expanded = dpo_count_sets[DATASET_NAMES["r5"]]
    safe_for_dpo = safe_for_sft and main_dpo_expanded.get("low_to_high", 0) > 0 and main_dpo_expanded.get(
        "high_to_low_protection", 0
    ) > 0

    tables_dir = out_dir / "tables"
    write_counter_rows(tables_dir / "exp19_s0_label_distribution.csv", "label_5", label_counts)
    write_counter_rows(tables_dir / "exp19_s0_major_failures_distribution.csv", "major_failures", failures)
    write_counter_rows(tables_dir / "exp19_s0_score_cap_distribution.csv", "score_cap", score_caps)
    write_counter_rows(tables_dir / "exp19_s0_rubric_satisfied_distribution.csv", "rubric_satisfied", rubric_satisfied)
    write_counter_rows(tables_dir / "exp19_s0_language_distribution.csv", "language", language_counts)
    write_counter_rows(tables_dir / "exp19_s0_metric_distribution.csv", "metric", metric_counts)
    dpo_rows: list[dict[str, Any]] = []
    for dataset_name, (original_counts, expanded_counts) in dpo_count_sets.items():
        for risk_type in sorted(set(original_counts) | set(expanded_counts)):
            dpo_rows.append(
                {
                    "dataset": dataset_name,
                    "risk_type": risk_type,
                    "original_count": original_counts.get(risk_type, 0),
                    "expanded_count": expanded_counts.get(risk_type, 0),
                }
            )
    write_csv(tables_dir / "exp19_s0_dpo_risk_counts.csv", dpo_rows)
    sft_label_rows: list[dict[str, Any]] = []
    for dataset_name, counts in sft_label_count_sets.items():
        total = sum(counts.values())
        for label, count in sorted(counts.items()):
            sft_label_rows.append(
                {
                    "dataset": dataset_name,
                    "label_5": label,
                    "count": count,
                    "rate": count / total if total else 0.0,
                }
            )
    write_csv(tables_dir / "exp19_s0_sft_label_distribution_by_dataset.csv", sft_label_rows)
    write_csv(tables_dir / "exp19_s0_dataset_counts.csv", dataset_rows)

    summary = {
        "train_count": len(train_rows),
        "dev_count_read_for_guardrail": len(dev_rows),
        "low_label_count": low_count,
        "high_label_count": high_count,
        "low_example_fraction": low_count / len(train_rows) if train_rows else 0.0,
        "high_score_protection_count": high_protection_count,
        "matched_a0_candidate_count": matched_candidates,
        "matched_a0_high_control_count": matched_high_controls,
        "usable_failure_targets": usable_failure_targets,
        "max_question_group_rate": max_rate(qgroup_counts),
        "leakage_reasons_checked": checked_reasons,
        "leakage_count": leakage_count,
        "safe_for_sft": safe_for_sft,
        "safe_for_dpo": safe_for_dpo,
    }
    write_json(out_dir / "reports" / "exp19_s0_summary.json", summary)

    report = [
        "# Exp19-S0 SFT/DPO Dataset QC Report",
        "",
        "Exp19-S0 prepares reason-aware SFT and DPO datasets for Qwen3-4B. No model is trained, and the test split is not read.",
        "",
        "## Counts",
        "",
        f"- train samples: {len(train_rows)}",
        f"- dev samples read for guardrail/evaluation prep only: {len(dev_rows)}",
        f"- low-label samples (y<=2): {low_count}",
        f"- high-label samples (y>=4): {high_count}",
        f"- low example fraction: {summary['low_example_fraction']:.4f}",
        f"- matched A0 hidden-failure candidates: {matched_candidates}",
        f"- matched A0 clean high controls: {matched_high_controls}",
        f"- usable low failure targets: {usable_failure_targets}",
        f"- high-score protection count: {high_protection_count}",
        "",
        "## Dataset Counts",
        "",
    ]
    for row in dataset_rows:
        report.append(f"- {row['dataset']}: {row['count']} ({row['kind']})")
    report.extend(
        [
            "",
            "## DPO Risk Counts",
            "",
        ]
    )
    for row in dpo_rows:
        report.append(
            f"- {row['dataset']} / {row['risk_type']}: original={row['original_count']}, expanded={row['expanded_count']}"
        )
    report.extend(
        [
            "",
            "## Distribution Summary",
            "",
            f"- label distribution: {dict(sorted(label_counts.items()))}",
            f"- SFT train distribution note: natural datasets preserve the original long-tail distribution; balanced datasets use explicit risk-aware sampling and must not be interpreted as the raw data distribution.",
            f"- major failures distribution: {dict(failures.most_common())}",
            f"- score cap distribution: {dict(score_caps.most_common())}",
            f"- rubric_satisfied distribution: {dict(rubric_satisfied.most_common())}",
            f"- max question_group/question_key rate: {summary['max_question_group_rate']:.4f}",
            f"- language distribution: {dict(language_counts.most_common())}",
            f"- top metrics: {dict(metric_counts.most_common(12))}",
            "",
            "## Leakage Check",
            "",
            f"- recovered human rationales checked against user prompts: {checked_reasons}",
            f"- exact human-rationale leakage count in user prompts: {leakage_count}",
            "- user prompts are built only from question, answer, metric, rubric, and metadata.",
            "- human rationale-derived fields appear only in assistant targets.",
            "",
            "## Recommendation",
            "",
            f"- safe for SFT: `{safe_for_sft}`",
            f"- safe for DPO: `{safe_for_dpo}`",
            "",
            "Full raw JSON datasets are intentionally written under the gitignored output/data directory. Redacted samples replace raw answers and rationales with previews/hashes.",
        ]
    )
    write_text(out_dir / "reports" / "exp19_sft_dpo_dataset_qc_report.md", "\n".join(report))
    return summary


def build_datasets(args: argparse.Namespace) -> dict[str, Any]:
    train_rows = read_jsonl(args.train_jsonl)
    dev_rows = read_jsonl(args.dev_jsonl)
    candidates, high_controls = build_signal_maps(args.a0_candidates, args.a0_high_controls)
    rng = random.Random(args.seed)

    r1_targets = [score_only_target(row) for row in train_rows]
    r2_targets = [evidence_target(row, candidates.get(sample_id(row)), include_reason=False) for row in train_rows]
    r3_targets = [evidence_target(row, candidates.get(sample_id(row)), include_reason=True) for row in train_rows]
    reason_bundles = [
        {
            "major_failures": target["major_failures"],
            "brief_reason": target.get("brief_reason", ""),
        }
        for target in r3_targets
    ]
    shuffled_bundles = list(reason_bundles)
    rng.shuffle(shuffled_bundles)
    r4_targets = []
    for row, original_target, bundle in zip(train_rows, r2_targets, shuffled_bundles):
        r4_targets.append(
            {
                "score": clamp_score(row.get("label_5")),
                "major_failures": bundle["major_failures"],
                "score_cap": original_target["score_cap"],
                "rubric_satisfied": original_target["rubric_satisfied"],
                "brief_reason": bundle["brief_reason"],
            }
        )

    r1 = [sft_example(row, target) for row, target in zip(train_rows, r1_targets)]
    r2 = [sft_example(row, target) for row, target in zip(train_rows, r2_targets)]
    r3 = [sft_example(row, target) for row, target in zip(train_rows, r3_targets)]
    r4 = [sft_example(row, target) for row, target in zip(train_rows, r4_targets)]
    r2_balanced_indices = build_balanced_indices(train_rows, r2_targets, rng, args.balanced_total, clean_low_only=False)
    r3_balanced_indices = build_balanced_indices(train_rows, r3_targets, rng, args.balanced_total, clean_low_only=False)
    r2_clean_balanced_indices = build_balanced_indices(
        train_rows, r2_targets, rng, args.balanced_total, clean_low_only=True
    )
    r2_balanced = build_sft_from_indices(train_rows, r2_targets, r2_balanced_indices)
    r3_balanced = build_sft_from_indices(train_rows, r3_targets, r3_balanced_indices)
    r2_clean_balanced = build_sft_from_indices(train_rows, r2_targets, r2_clean_balanced_indices)

    dpo_prefs = [dpo_chosen_rejected(row, candidates.get(sample_id(row))) for row in train_rows]
    r5_high_control, dpo_control_original_counts, dpo_control_expanded_counts = build_dpo_high_protection_control(
        train_rows, dpo_prefs
    )
    r5, dpo_original_counts, dpo_expanded_counts = build_dpo_risk_balanced(
        train_rows, dpo_prefs, rng, args.dpo_balanced_total
    )

    data_dir = args.out_dir / "data"
    redacted_dir = args.out_dir / "redacted_samples"
    write_json_array(data_dir / "edubench_r1_score_only_train.json", r1)
    write_json_array(data_dir / "edubench_r2_reason_score_train.json", r2)
    write_json_array(data_dir / "edubench_r3_reason_rationale_train.json", r3)
    write_json_array(data_dir / "edubench_r4_shuffled_reason_control_train.json", r4)
    write_json_array(data_dir / "edubench_r2_reason_score_balanced_train.json", r2_balanced)
    write_json_array(data_dir / "edubench_r3_reason_rationale_balanced_train.json", r3_balanced)
    write_json_array(data_dir / "edubench_r2_clean_reason_score_balanced_train.json", r2_clean_balanced)
    write_json_array(data_dir / "edubench_r5_risk_balanced_dpo_train.json", r5)
    write_json_array(data_dir / "edubench_r5_high_protection_dpo_control_train.json", r5_high_control)

    limit = args.max_redacted_samples
    write_json(redacted_dir / "edubench_r1_score_only_train_sample_redacted.json", redacted_sft_sample(train_rows, r1_targets, limit))
    write_json(redacted_dir / "edubench_r2_reason_score_train_sample_redacted.json", redacted_sft_sample(train_rows, r2_targets, limit))
    write_json(
        redacted_dir / "edubench_r3_reason_rationale_train_sample_redacted.json",
        redacted_sft_sample(train_rows, r3_targets, limit),
    )
    write_json(
        redacted_dir / "edubench_r4_shuffled_reason_control_train_sample_redacted.json",
        redacted_sft_sample(train_rows, r4_targets, limit),
    )
    write_json(redacted_dir / "edubench_r5_risk_balanced_dpo_train_sample_redacted.json", redacted_dpo_sample(train_rows, dpo_prefs, limit))
    write_json(
        redacted_dir / "r2_low_failure_samples_redacted.json",
        redacted_sft_filtered_sample(train_rows, r2_targets, limit, lambda row, target: is_low(row)),
    )
    write_json(
        redacted_dir / "r2_mid_samples_redacted.json",
        redacted_sft_filtered_sample(train_rows, r2_targets, limit, lambda row, target: is_mid(row)),
    )
    write_json(
        redacted_dir / "r2_high_protection_samples_redacted.json",
        redacted_sft_filtered_sample(train_rows, r2_targets, limit, lambda row, target: is_high(row)),
    )
    write_json(
        redacted_dir / "r5_low_to_high_dpo_samples_redacted.json",
        redacted_dpo_filtered_sample(train_rows, dpo_prefs, limit, "low_to_high"),
    )
    write_json(
        redacted_dir / "r5_high_to_low_dpo_samples_redacted.json",
        redacted_dpo_filtered_sample(train_rows, dpo_prefs, limit, "high_to_low_protection"),
    )

    write_json(args.out_dir / "dataset_info_snippet.json", dataset_info_snippet())
    write_configs(args.out_dir / "configs", args.out_dir)
    dataset_rows = [
        {"dataset": DATASET_NAMES["r1"], "count": len(r1), "kind": "sft_natural"},
        {"dataset": DATASET_NAMES["r2"], "count": len(r2), "kind": "sft_natural"},
        {"dataset": DATASET_NAMES["r3"], "count": len(r3), "kind": "sft_natural"},
        {"dataset": DATASET_NAMES["r4"], "count": len(r4), "kind": "sft_control"},
        {"dataset": DATASET_NAMES["r2_balanced"], "count": len(r2_balanced), "kind": "sft_balanced"},
        {"dataset": DATASET_NAMES["r3_balanced"], "count": len(r3_balanced), "kind": "sft_balanced"},
        {"dataset": DATASET_NAMES["r2_clean_balanced"], "count": len(r2_clean_balanced), "kind": "sft_clean_balanced"},
        {"dataset": DATASET_NAMES["r5"], "count": len(r5), "kind": "dpo_risk_balanced"},
        {"dataset": DATASET_NAMES["r5_high_control"], "count": len(r5_high_control), "kind": "dpo_control"},
    ]
    dpo_count_sets = {
        DATASET_NAMES["r5"]: (dpo_original_counts, dpo_expanded_counts),
        DATASET_NAMES["r5_high_control"]: (dpo_control_original_counts, dpo_control_expanded_counts),
    }
    sft_label_count_sets = {
        DATASET_NAMES["r2"]: Counter(str(clamp_score(row.get("label_5"))) for row in train_rows),
        DATASET_NAMES["r2_balanced"]: Counter(str(clamp_score(train_rows[idx].get("label_5"))) for idx in r2_balanced_indices),
        DATASET_NAMES["r3_balanced"]: Counter(str(clamp_score(train_rows[idx].get("label_5"))) for idx in r3_balanced_indices),
        DATASET_NAMES["r2_clean_balanced"]: Counter(str(clamp_score(train_rows[idx].get("label_5"))) for idx in r2_clean_balanced_indices),
    }
    summary = build_qc_report(
        args.out_dir,
        train_rows,
        dev_rows,
        r2_targets,
        r3_targets,
        r4_targets,
        candidates,
        high_controls,
        dataset_rows,
        dpo_count_sets,
        sft_label_count_sets,
    )
    summary.update(
        {
            "r1_count": len(r1),
            "r2_count": len(r2),
            "r3_count": len(r3),
            "r4_count": len(r4),
            "r2_balanced_count": len(r2_balanced),
            "r3_balanced_count": len(r3_balanced),
            "r2_clean_balanced_count": len(r2_clean_balanced),
            "r5_count": len(r5),
            "r5_high_control_count": len(r5_high_control),
            "out_dir": str(args.out_dir),
        }
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Exp19-S0 reason-aware SFT and risk-balanced DPO datasets for Qwen3-4B."
    )
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--dev-jsonl", type=Path, default=DEFAULT_DEV_PATH)
    parser.add_argument("--a0-candidates", type=Path, default=DEFAULT_A0_CANDIDATES)
    parser.add_argument("--a0-high-controls", type=Path, default=DEFAULT_A0_HIGH_CONTROLS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-redacted-samples", type=int, default=20)
    parser.add_argument("--balanced-total", type=int, default=3000)
    parser.add_argument("--dpo-balanced-total", type=int, default=3000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in [args.train_jsonl, args.dev_jsonl, args.a0_candidates, args.a0_high_controls]:
        if not path.exists():
            raise FileNotFoundError(path)
    summary = build_datasets(args)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
