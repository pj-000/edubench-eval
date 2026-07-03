"""Prepare Exp19-R5D evidence-consistency DPO data.

R5D separates structured failure-field consistency from score-risk DPO. It
keeps low-score examples whose rejected response has low/mid score but missing
failure evidence, and high-score examples whose rejected response invents false
failure evidence. It does not train, read test, or use dev/D1 annotations as
training labels.
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
DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_r5d_evidence_consistency_dpo_seed42")

DATASET_NAME = "edubench_r5d_evidence_consistency_dpo_train"
DATA_FILE = "data/edubench_r5d_evidence_consistency_dpo_train.json"
RISK_TYPES = ["low_failure_evidence_consistency", "high_no_false_failure_consistency"]
TARGET_FRACTIONS = {
    "low_failure_evidence_consistency": 0.50,
    "high_no_false_failure_consistency": 0.50,
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


def hard_low_inconsistent_rejected(_: random.Random) -> tuple[str, dict[str, Any]]:
    return (
        "hard_synthetic_low_no_failure_inconsistent",
        {
            "score": 2,
            "major_failures": ["no_major_failure"],
            "score_cap": None,
            "rubric_satisfied": True,
        },
    )


def hard_high_false_failure_rejected(rng: random.Random) -> tuple[str, dict[str, Any]]:
    if rng.random() < 0.6:
        return (
            "hard_synthetic_high_false_failure_score4",
            {
                "score": 4,
                "major_failures": ["insufficient_evidence"],
                "score_cap": 3,
                "rubric_satisfied": False,
            },
        )
    return (
        "hard_synthetic_high_false_failure_score3",
        {
            "score": 3,
            "major_failures": ["unclear"],
            "score_cap": 3,
            "rubric_satisfied": False,
        },
    )


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


def low_evidence_inconsistent(target: dict[str, Any]) -> bool:
    score = score_of(target)
    if score is None or score > 3:
        return False
    cap = safe_int(target.get("score_cap"))
    return (
        has_no_major_failure(target)
        or target.get("score_cap") is None
        or (cap is not None and cap > 2)
        or target.get("rubric_satisfied") is True
        or not has_substantive_failure(target)
    )


def high_false_failure_inconsistent(target: dict[str, Any]) -> bool:
    score = score_of(target)
    if score is None or score < 3:
        return False
    cap = safe_int(target.get("score_cap"))
    return has_substantive_failure(target) or cap is not None or target.get("rubric_satisfied") is False


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
) -> dict[str, list[dict[str, Any]]]:
    low_predictions = load_subset_predictions(d2_dir, "train_low_reasoned_subset")
    high_predictions = load_subset_predictions(d2_dir, "train_clean_high_subset")
    pools: dict[str, list[dict[str, Any]]] = {risk_type: [] for risk_type in RISK_TYPES}
    for row in train_rows:
        sid = sample_id(row)
        label = clamp_score(row.get("label_5"))
        if label <= 2 and is_usable_low_candidate(candidates.get(sid)):
            chosen = low_chosen(row, candidates.get(sid))
            actuals = [
                (source, target)
                for source, target in low_predictions.get(sid, [])
                if low_evidence_inconsistent(target)
            ]
            if actuals:
                for source, rejected in actuals:
                    add_pair_candidate(
                        pools["low_failure_evidence_consistency"],
                        row,
                        chosen,
                        rejected,
                        "low_failure_evidence_consistency",
                        source,
                    )
            else:
                source, rejected = hard_low_inconsistent_rejected(rng)
                add_pair_candidate(
                    pools["low_failure_evidence_consistency"],
                    row,
                    chosen,
                    rejected,
                    "low_failure_evidence_consistency",
                    source,
                )
        elif label >= 4 and sid in high_controls:
            chosen = high_chosen(row)
            actuals = [
                (source, target)
                for source, target in high_predictions.get(sid, [])
                if high_false_failure_inconsistent(target)
            ]
            if actuals:
                for source, rejected in actuals:
                    add_pair_candidate(
                        pools["high_no_false_failure_consistency"],
                        row,
                        chosen,
                        rejected,
                        "high_no_false_failure_consistency",
                        source,
                    )
            else:
                source, rejected = hard_high_false_failure_rejected(rng)
                add_pair_candidate(
                    pools["high_no_false_failure_consistency"],
                    row,
                    chosen,
                    rejected,
                    "high_no_false_failure_consistency",
                    source,
                )
    return pools


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
        "low_failure_evidence_consistency": int(round(total_pairs * TARGET_FRACTIONS["low_failure_evidence_consistency"])),
    }
    targets["high_no_false_failure_consistency"] = total_pairs - targets["low_failure_evidence_consistency"]
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


def validity_for_pair(pair: dict[str, Any]) -> bool:
    target = target_from_message(pair["rejected"])
    if pair["risk_type"] == "low_failure_evidence_consistency":
        return low_evidence_inconsistent(target)
    if pair["risk_type"] == "high_no_false_failure_consistency":
        return high_false_failure_inconsistent(target)
    return False


def table_rows(pools: dict[str, list[dict[str, Any]]], pairs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    expanded = Counter(pair["risk_type"] for pair in pairs)
    total = len(pairs)
    count_rows = [
        {
            "risk_type": risk_type,
            "original_count": len(pools.get(risk_type, [])),
            "expanded_count": expanded.get(risk_type, 0),
            "actual_fraction": safe_rate(expanded.get(risk_type, 0), total),
        }
        for risk_type in RISK_TYPES
    ]
    source_counts: Counter[tuple[str, str, str]] = Counter(
        (pair["rejected_source"], pair["rejected_category"], pair["risk_type"]) for pair in pairs
    )
    source_rows = [
        {"rejected_source": source, "rejected_category": category, "risk_type": risk_type, "n": n}
        for (source, category, risk_type), n in sorted(source_counts.items(), key=lambda item: (item[0][2], item[0][1], item[0][0]))
    ]
    quality_rows: list[dict[str, Any]] = []
    validity_rows: list[dict[str, Any]] = []
    for risk_type in RISK_TYPES:
        items = [pair for pair in pairs if pair["risk_type"] == risk_type]
        chosen_targets = [target_from_message(pair["chosen"]) for pair in items]
        rejected_targets = [target_from_message(pair["rejected"]) for pair in items]
        chosen_scores = [score_of(target) for target in chosen_targets]
        rejected_scores = [score_of(target) for target in rejected_targets]
        n = len(items)
        quality_rows.append(
            {
                "risk_type": risk_type,
                "n": n,
                "chosen_score_mean": mean([float(x) for x in chosen_scores if x is not None]),
                "rejected_score_mean": mean([float(x) for x in rejected_scores if x is not None]),
                "actual_model_rejected_rate": safe_rate(sum(1 for pair in items if pair["rejected_category"] == "actual_model"), n),
                "hard_synthetic_rate": safe_rate(sum(1 for pair in items if pair["rejected_category"] == "hard_synthetic"), n),
                "validity_rate": safe_rate(sum(1 for pair in items if validity_for_pair(pair)), n),
                "rejected_has_failure_rate": safe_rate(sum(1 for target in rejected_targets if has_substantive_failure(target)), n),
                "rejected_no_major_failure_rate": safe_rate(sum(1 for target in rejected_targets if has_no_major_failure(target)), n),
            }
        )
        validity_rows.append(
            {
                "risk_type": risk_type,
                "n": n,
                "validity_rate": safe_rate(sum(1 for pair in items if validity_for_pair(pair)), n),
                "rejected_no_major_failure_rate": safe_rate(sum(1 for target in rejected_targets if has_no_major_failure(target)), n),
                "rejected_substantive_failure_rate": safe_rate(sum(1 for target in rejected_targets if has_substantive_failure(target)), n),
                "rejected_score_cap_nonnull_rate": safe_rate(
                    sum(1 for target in rejected_targets if target.get("score_cap") is not None), n
                ),
                "rejected_rubric_satisfied_false_rate": safe_rate(
                    sum(1 for target in rejected_targets if target.get("rubric_satisfied") is False), n
                ),
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
        "risk_type": pair.get("risk_type", ""),
        "rejected_source": pair.get("rejected_source", ""),
        "rejected_category": pair.get("rejected_category", ""),
    }


def write_redacted_samples(out_dir: Path, pairs: list[dict[str, Any]], limit: int) -> None:
    for risk_type in RISK_TYPES:
        selected = [pair for pair in pairs if pair["risk_type"] == risk_type][:limit]
        write_json(out_dir / "redacted_samples" / f"{risk_type}_redacted.json", [redacted_pair(pair) for pair in selected])


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


def report_text(
    pairs: list[dict[str, Any]],
    count_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    quality_rows: list[dict[str, Any]],
    validity_rows: list[dict[str, Any]],
    leakage_checked: int,
    leakage_count: int,
) -> tuple[str, bool]:
    categories = Counter(pair["rejected_category"] for pair in pairs)
    validity = safe_rate(sum(1 for pair in pairs if validity_for_pair(pair)), len(pairs))
    ready = (
        len([pair for pair in pairs if pair["risk_type"] == "low_failure_evidence_consistency"]) >= 300
        and len([pair for pair in pairs if pair["risk_type"] == "high_no_false_failure_consistency"]) >= 300
        and validity >= 0.95
        and leakage_count == 0
    )
    lines = [
        "# Exp19-R5D Evidence-Consistency DPO QC Report",
        "",
        "R5D trains structured score/failure consistency rather than score-level risk. It is an auxiliary DPO dataset.",
        "",
        "## Summary",
        "",
        f"- total DPO pairs constructed: {len(pairs)}",
        f"- actual model-output rejected responses: {categories.get('actual_model', 0)} ({fmt(safe_rate(categories.get('actual_model', 0), len(pairs)))})",
        f"- hard synthetic rejected responses: {categories.get('hard_synthetic', 0)} ({fmt(safe_rate(categories.get('hard_synthetic', 0), len(pairs)))})",
        f"- recovered human rationales checked for prompt leakage: {leakage_checked}",
        f"- exact human-rationale leakage count in prompts: {leakage_count}",
        f"- consistency validity rate: {fmt(validity)}",
        f"- ready_for_evidence_consistency_dpo: `{ready}`",
        "",
        "## Pair Counts",
        "",
        "| risk_type | original | expanded | actual_fraction |",
        "|---|---:|---:|---:|",
    ]
    for row in count_rows:
        lines.append(f"| {row['risk_type']} | {row['original_count']} | {row['expanded_count']} | {fmt(row['actual_fraction'])} |")
    lines.extend(["", "## Rejected Sources", "", "| source | category | risk_type | n |", "|---|---|---|---:|"])
    for row in source_rows:
        lines.append(f"| {row['rejected_source']} | {row['rejected_category']} | {row['risk_type']} | {row['n']} |")
    lines.extend(
        [
            "",
            "## Pair Quality",
            "",
            "| risk_type | n | chosen_mean | rejected_mean | actual_rate | hard_rate | validity | rejected_failure | rejected_no_failure |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in quality_rows:
        lines.append(
            f"| {row['risk_type']} | {row['n']} | {fmt(row['chosen_score_mean'])} | "
            f"{fmt(row['rejected_score_mean'])} | {fmt(row['actual_model_rejected_rate'])} | "
            f"{fmt(row['hard_synthetic_rate'])} | {fmt(row['validity_rate'])} | "
            f"{fmt(row['rejected_has_failure_rate'])} | {fmt(row['rejected_no_major_failure_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Evaluation Focus",
            "",
            "- major_failure_nonempty_rate on D1 hidden cases",
            "- no_major_failure rate on clean high controls",
            "- failure type F1",
            "- score_cap and rubric_satisfied consistency",
            "",
            "Full DPO JSON is gitignored and must not be committed.",
        ]
    )
    return "\n".join(lines), ready


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
    write_json(args.out_dir / "dataset_info_r5d_snippet.json", dataset_info)
    count_rows, source_rows, quality_rows, validity_rows = table_rows(pools, pairs)
    write_csv(args.out_dir / "tables" / "r5d_pair_counts.csv", count_rows)
    write_csv(args.out_dir / "tables" / "r5d_rejected_source_counts.csv", source_rows)
    write_csv(args.out_dir / "tables" / "r5d_pair_quality_stats.csv", quality_rows)
    write_csv(args.out_dir / "tables" / "r5d_consistency_validity.csv", validity_rows)
    write_redacted_samples(args.out_dir, pairs, args.redacted_limit)
    leakage_checked, leakage_count = leakage_check(train_rows, candidates)
    report, ready = report_text(pairs, count_rows, source_rows, quality_rows, validity_rows, leakage_checked, leakage_count)
    write_text(args.out_dir / "reports" / "r5d_evidence_consistency_dpo_qc_report.md", report)
    write_dpo_config(
        args.out_dir / "configs" / "llamafactory_qwen3_4b_r5d_dpo_from_r2c.yaml",
        "saves/edubench/qwen3-4b/r2_clean_reason_score_balanced_lora",
        "saves/edubench/qwen3-4b/r5d_dpo_from_r2c_lora",
        args.out_dir,
    )
    write_dpo_config(
        args.out_dir / "configs" / "llamafactory_qwen3_4b_r5d_dpo_from_r1b.yaml",
        "saves/edubench/qwen3-4b/r1_score_only_balanced_lora",
        "saves/edubench/qwen3-4b/r5d_dpo_from_r1b_lora",
        args.out_dir,
    )
    categories = Counter(pair["rejected_category"] for pair in pairs)
    return {
        "pairs": len(pairs),
        "ready_for_evidence_consistency_dpo": ready,
        "leakage_count": leakage_count,
        "actual_model_rejected": categories.get("actual_model", 0),
        "hard_synthetic_rejected": categories.get("hard_synthetic", 0),
        "counts": {row["risk_type"]: row["expanded_count"] for row in count_rows},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Exp19-R5D evidence-consistency DPO data.")
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
