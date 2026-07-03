"""Prepare Exp19-R5F2 expanded rejection-mining inputs and DPO data.

R5F2 expands the train-only rejection mining attempted in R5F. It does not
train a model, does not read test, and does not use dev/D1 annotations as train
labels. Human rationale-derived A0 signals are used only to choose train
candidates and construct chosen targets; rationale text is never placed in the
model prompt.
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
    load_prediction_records,
    normalize_failure_name,
    safe_int,
)
from thesis_exp.exp17_low_score_evidence.prepare_exp19_r5a_onpolicy_dpo import (  # noqa: E402
    score_of,
    target_from_raw_output,
)
from thesis_exp.exp17_low_score_evidence.prepare_exp19_r5c_score_risk_dpo import (  # noqa: E402
    high_chosen,
    target_from_message,
    write_dpo_config,
    write_json_array,
)
from thesis_exp.exp17_low_score_evidence.prepare_exp19_sft_dpo_datasets import (  # noqa: E402
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
    redacted_target,
    sample_id,
    sha1,
    sft_dataset_entry,
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
DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_r5f2_rejection_mining_seed42")

GEN_DATASET = "edubench_exp19_r5f2_generation_train"
GEN_FILE = "data/edubench_exp19_r5f2_generation_train.json"
MAIN_DATASET = "edubench_r5f2_score_risk_main_dpo_train"
MAIN_FILE = "data/edubench_r5f2_score_risk_main_dpo_train.json"
PLUS_DATASET = "edubench_r5f2_score_risk_plus_evidence_dpo_train"
PLUS_FILE = "data/edubench_r5f2_score_risk_plus_evidence_dpo_train.json"
REAL_ONLY_DATASET = "edubench_r5f2_real_only_small_dpo_train"
REAL_ONLY_FILE = "data/edubench_r5f2_real_only_small_dpo_train.json"

STRICT_FAILURE_MODES = {
    "missing_key_point",
    "surface_fluent_but_hidden_defect",
    "insufficient_evidence",
}
USABLE_USES = {"evidence_positive", "weak_evidence_positive", "pairwise_low", "format_auxiliary"}
USABLE_TYPES = {"strong_evidence_positive", "weak_evidence_positive", "pairwise_low", "format_auxiliary"}

GENERATOR_ORDER = [
    "r0a_direct_qwen3_4b",
    "r1b_score_only_balanced",
    "r2n_reason_score_natural",
    "r2c_clean_reason_score_balanced",
    "r4b_shuffled_reason_balanced",
    "r5c_from_r1b",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def gold_label(row: dict[str, Any]) -> int:
    return clamp_score(row.get("label_5") or row.get("label") or row.get("gold_label"))


def stable_sample_id(row: dict[str, Any], idx: int) -> str:
    sid = sample_id(row)
    return sid if sid else f"train_row_{idx:05d}"


def answer_text(row: dict[str, Any]) -> str:
    return clean(row.get("answer"))


def word_count(text: str) -> int:
    return len([part for part in text.replace("\n", " ").split(" ") if part.strip()])


def signal_failure(signal: dict[str, str] | None) -> str:
    if not signal:
        return "unclear"
    return normalize_failure_name(signal.get("failure_mode_auto")) or clean(signal.get("failure_mode_auto")) or "unclear"


def signal_is_usable(signal: dict[str, str] | None) -> bool:
    if not signal:
        return False
    use = clean(signal.get("recommended_training_use"))
    candidate_type = clean(signal.get("hidden_failure_candidate_type"))
    if use not in USABLE_USES and candidate_type not in USABLE_TYPES:
        return False
    return candidate_type not in {"answer_key_dependent", "conflict_or_exclude", "unclear"}


def pool_for_sample(row: dict[str, Any], signal: dict[str, str] | None, min_answer_chars: int) -> str:
    label = gold_label(row)
    text = answer_text(row)
    failure = signal_failure(signal)
    if label <= 2 and len(text) >= min_answer_chars and signal_is_usable(signal) and failure in STRICT_FAILURE_MODES:
        return "pool_a_strict_d1_like_low"
    if label <= 2:
        return "pool_b_all_train_low"
    if label == 3 and signal_is_usable(signal) and failure in STRICT_FAILURE_MODES:
        return "pool_c_borderline_hidden_mid"
    return ""


def prompt_record(row: dict[str, Any]) -> dict[str, Any]:
    return {"messages": messages_for(row) + [{"role": "assistant", "content": "{}"}]}


def candidate_meta(
    row: dict[str, Any],
    signal: dict[str, str] | None,
    idx: int,
    pool: str,
) -> dict[str, Any]:
    text = answer_text(row)
    return {
        "sample_id": stable_sample_id(row, idx),
        "question_key": clean(row.get("question_key")),
        "question_group_id": question_group(row),
        "metric": metric_name(row),
        "language": language(row),
        "subject": subject(row),
        "gold_label": gold_label(row),
        "answer_chars": len(text),
        "answer_words": word_count(text),
        "candidate_pool": pool,
        "failure_mode_auto": signal_failure(signal),
        "hidden_failure_candidate_type": clean((signal or {}).get("hidden_failure_candidate_type")),
        "recommended_training_use": clean((signal or {}).get("recommended_training_use")),
        "confidence_weight": clean((signal or {}).get("confidence_weight")),
        "prompt_hash": sha1(user_content(row)),
    }


def prepare_inputs(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.train_jsonl)
    signals = {clean(row.get("sample_id")): row for row in read_csv_rows(args.a0_candidates)}
    selected: list[tuple[dict[str, Any], dict[str, str] | None, dict[str, Any]]] = []

    for idx, row in enumerate(rows):
        sid = stable_sample_id(row, idx)
        signal = signals.get(sid)
        pool = pool_for_sample(row, signal, args.min_answer_chars)
        if not pool:
            continue
        if pool == "pool_c_borderline_hidden_mid" and not args.include_borderline_mid:
            continue
        meta = candidate_meta(row, signal, idx, pool)
        selected.append((row, signal, meta))

    pool_priority = {"pool_a_strict_d1_like_low": 0, "pool_b_all_train_low": 1, "pool_c_borderline_hidden_mid": 2}
    selected.sort(key=lambda item: (pool_priority[item[2]["candidate_pool"]], -int(item[2]["answer_chars"])))
    if args.max_candidates > 0:
        selected = selected[: args.max_candidates]

    data_rows = [prompt_record(row) for row, _signal, _meta in selected]
    candidate_rows = [meta for _row, _signal, meta in selected]
    reference_rows = []
    for eval_index, (row, signal, meta) in enumerate(selected):
        reference_rows.append(
            {
                "eval_index": eval_index,
                **meta,
                "recovered_reason_hash": sha1(recovered_reason(signal)),
            }
        )

    write_json_array(args.out_dir / GEN_FILE, data_rows)
    write_json(args.out_dir / "dataset_info.json", {GEN_DATASET: sft_dataset_entry(GEN_FILE)})
    write_json(args.out_dir / "dataset_info_r5f2_generation_snippet.json", {GEN_DATASET: sft_dataset_entry(GEN_FILE)})
    write_csv(args.out_dir / "tables" / "r5f2_candidate_pool.csv", candidate_rows)
    write_csv(args.out_dir / "references" / "r5f2_generation_reference.csv", reference_rows)

    pool_counts = Counter(row["candidate_pool"] for row in candidate_rows)
    failure_counts = Counter(row["failure_mode_auto"] for row in candidate_rows)
    report = [
        "# Exp19-R5F2 Generation Input QC",
        "",
        "R5F2 expands train-only rejection mining. Human rationale is not included in prompts.",
        "",
        f"- train source: `{args.train_jsonl}`",
        f"- candidates selected: {len(candidate_rows)}",
        f"- min answer chars for Pool A: {args.min_answer_chars}",
        f"- include borderline y=3 pool: `{args.include_borderline_mid}`",
        "",
        "## Candidate Pools",
        "",
    ]
    report.extend(f"- {key}: {value}" for key, value in sorted(pool_counts.items()))
    report.extend(["", "## Failure Modes", ""])
    report.extend(f"- {key}: {value}" for key, value in sorted(failure_counts.items()))
    report.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Test split is not read.",
            "- Dev D1 annotations are not used as training labels.",
            "- Human rationale-derived signals are used only for train-side candidate/target construction.",
        ]
    )
    write_text(args.out_dir / "reports" / "r5f2_generation_input_report.md", "\n".join(report))
    return {"stage": "prepare_inputs", "candidates": len(candidate_rows), "dataset": GEN_DATASET}


def generation_dirs(prediction_root: Path, generator: str, total_samples: int) -> list[Path]:
    return [prediction_root / generator / f"sample_{idx}" for idx in range(total_samples)]


def load_generated_targets(
    prediction_root: Path,
    reference_rows: list[dict[str, str]],
    total_samples: int,
) -> dict[str, list[tuple[str, dict[str, Any]]]]:
    out: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for generator in GENERATOR_ORDER:
        for sample_idx, run_dir in enumerate(generation_dirs(prediction_root, generator, total_samples)):
            try:
                pred_file = find_prediction_file(run_dir)
            except FileNotFoundError:
                continue
            records = load_prediction_records(pred_file)
            for ref, record in zip(reference_rows, records):
                sid = clean(ref.get("sample_id"))
                target = target_from_raw_output(str(record.get("raw_output", "")))
                if sid and target is not None:
                    out[sid].append((f"{generator}_sample{sample_idx}", target))
    return out


def is_low_score_risk_rejected(target: dict[str, Any]) -> bool:
    score = score_of(target)
    return score is not None and score >= 4


def is_low_evidence_inconsistency(target: dict[str, Any]) -> bool:
    score = score_of(target)
    return score == 3 and has_no_major_failure(target) and target.get("score_cap") is None and target.get("rubric_satisfied") is True


def low_pair(
    row: dict[str, Any],
    signal: dict[str, str] | None,
    rejected: dict[str, Any],
    rejected_source: str,
    risk_type: str,
    dataset_variant: str,
    copy_index: int,
    candidate_pool: str,
) -> dict[str, Any]:
    return {
        "messages": messages_for(row),
        "chosen": {"role": "assistant", "content": assistant_json(evidence_target(row, signal, include_reason=False))},
        "rejected": {"role": "assistant", "content": assistant_json(rejected)},
        "risk_type": risk_type,
        "rejected_source": rejected_source,
        "rejected_category": "actual_model_generation",
        "source_sample_id": sample_id(row),
        "candidate_pool": candidate_pool,
        "dataset_variant": dataset_variant,
        "oversample_copy_index": copy_index,
    }


def hard_high_protection_pair(
    row: dict[str, Any],
    copy_index: int,
    dataset_variant: str,
    rng: random.Random,
) -> dict[str, Any]:
    rejected = {
        "score": 2 if rng.random() < 0.65 else 3,
        "major_failures": ["missing_key_point"] if rng.random() < 0.5 else ["insufficient_evidence"],
        "score_cap": 2,
        "rubric_satisfied": False,
    }
    return {
        "messages": messages_for(row),
        "chosen": {"role": "assistant", "content": assistant_json(high_chosen(row))},
        "rejected": {"role": "assistant", "content": assistant_json(rejected)},
        "risk_type": "high_to_low_score_risk",
        "rejected_source": "hard_synthetic_high_to_low",
        "rejected_category": "hard_synthetic",
        "source_sample_id": sample_id(row),
        "candidate_pool": "clean_high_control",
        "dataset_variant": dataset_variant,
        "oversample_copy_index": copy_index,
    }


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
        "candidate_pool": pair.get("candidate_pool", ""),
        "dataset_variant": pair.get("dataset_variant", ""),
    }


def sample_high_controls(
    train_rows: dict[str, dict[str, Any]],
    high_controls_path: Path,
    count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    controls = [row for row in read_csv(high_controls_path) if clean(row.get("sample_id")) in train_rows]
    pool = [train_rows[clean(row.get("sample_id"))] for row in controls]
    if not pool or count <= 0:
        return []
    if count <= len(pool):
        return rng.sample(pool, count)
    return [rng.choice(pool) for _ in range(count)]


def write_dataset(
    out_dir: Path,
    dataset_name: str,
    file_name: str,
    pairs: list[dict[str, Any]],
    redacted_limit: int,
    config_prefix: str,
) -> None:
    write_json_array(out_dir / file_name, pairs)
    write_json(out_dir / f"dataset_info_{config_prefix}_snippet.json", {dataset_name: dpo_dataset_entry(file_name)})
    write_json(out_dir / "redacted_samples" / f"{config_prefix}_redacted.json", [redacted_pair(pair) for pair in pairs[:redacted_limit]])
    write_dpo_config(
        out_dir / "configs" / f"llamafactory_qwen3_4b_{config_prefix}_from_r1b.yaml",
        dataset_name,
        "saves/edubench/qwen3-4b/r1_score_only_balanced_lora",
        f"saves/edubench/qwen3-4b/{config_prefix}_from_r1b_lora",
        out_dir,
    )
    write_dpo_config(
        out_dir / "configs" / f"llamafactory_qwen3_4b_{config_prefix}_from_r2c.yaml",
        dataset_name,
        "saves/edubench/qwen3-4b/r2_clean_reason_score_balanced_lora",
        f"saves/edubench/qwen3-4b/{config_prefix}_from_r2c_lora",
        out_dir,
    )


def pair_quality(dataset: str, pairs: list[dict[str, Any]]) -> dict[str, Any]:
    chosen_scores = [score_of(target_from_message(pair["chosen"])) for pair in pairs]
    rejected_scores = [score_of(target_from_message(pair["rejected"])) for pair in pairs]
    risk_counts = Counter(pair.get("risk_type", "") for pair in pairs)
    unique_rejected = {sha1(pair["rejected"]["content"]) for pair in pairs}
    low_pairs = [pair for pair in pairs if pair.get("risk_type") == "low_to_high_score_risk"]
    return {
        "dataset": dataset,
        "pairs": len(pairs),
        "unique_source_samples": len({pair["source_sample_id"] for pair in pairs}),
        "unique_low_samples": len({pair["source_sample_id"] for pair in low_pairs}),
        "unique_rejected_payloads": len(unique_rejected),
        "duplicate_rejected_payload_rate": 1.0 - (len(unique_rejected) / len(pairs)) if pairs else 0.0,
        "chosen_score_mean": sum(float(x) for x in chosen_scores if x is not None) / max(1, sum(x is not None for x in chosen_scores)),
        "rejected_score_mean": sum(float(x) for x in rejected_scores if x is not None) / max(1, sum(x is not None for x in rejected_scores)),
        **{f"risk_type_{key}": value for key, value in sorted(risk_counts.items())},
    }


def build_dpo(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.train_jsonl)
    train_rows = {sample_id(row): row for row in rows}
    signals = {clean(row.get("sample_id")): row for row in read_csv_rows(args.a0_candidates)}
    reference_rows = read_csv(args.out_dir / "references" / "r5f2_generation_reference.csv")
    generated = load_generated_targets(args.prediction_root, reference_rows, args.total_samples_per_generator)
    rng = random.Random(args.seed)

    score_risk_pairs: list[dict[str, Any]] = []
    evidence_pairs: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []
    source_counter: Counter[str] = Counter()
    pool_counter: Counter[str] = Counter()

    for ref in reference_rows:
        sid = clean(ref.get("sample_id"))
        row = train_rows.get(sid)
        if row is None:
            continue
        signal = signals.get(sid)
        targets = generated.get(sid, [])
        score_risks = [(source, target) for source, target in targets if is_low_score_risk_rejected(target)]
        evidence_risks = [(source, target) for source, target in targets if is_low_evidence_inconsistency(target)]
        rng.shuffle(score_risks)
        rng.shuffle(evidence_risks)
        for copy_index, (source, rejected) in enumerate(score_risks[: args.max_score_rejected_per_sample]):
            pair = low_pair(
                row,
                signal,
                rejected,
                source,
                "low_to_high_score_risk",
                "r5f2_score_risk",
                copy_index,
                ref.get("candidate_pool", ""),
            )
            score_risk_pairs.append(pair)
            source_counter.update([source])
            pool_counter.update([ref.get("candidate_pool", "")])
        for copy_index, (source, rejected) in enumerate(evidence_risks[: args.max_evidence_rejected_per_sample]):
            pair = low_pair(
                row,
                signal,
                rejected,
                source,
                "low_failure_evidence_inconsistency",
                "r5f2_score_risk_plus_evidence",
                copy_index,
                ref.get("candidate_pool", ""),
            )
            evidence_pairs.append(pair)
        stats_rows.append(
            {
                "sample_id": sid,
                "candidate_pool": ref.get("candidate_pool", ""),
                "gold_label": ref.get("gold_label", ""),
                "generated_count": len(targets),
                "score_risk_rejected_count": len(score_risks),
                "evidence_inconsistency_rejected_count": len(evidence_risks),
                "used_score_risk_pairs": min(len(score_risks), args.max_score_rejected_per_sample),
                "used_evidence_pairs": min(len(evidence_risks), args.max_evidence_rejected_per_sample),
                "failure_mode_auto": ref.get("failure_mode_auto", ""),
                "metric": ref.get("metric", ""),
                "language": ref.get("language", ""),
            }
        )

    rng.shuffle(score_risk_pairs)
    rng.shuffle(evidence_pairs)

    high_count = min(len(score_risk_pairs), args.max_high_protection_pairs)
    high_rows = sample_high_controls(train_rows, args.a0_high_controls, high_count, rng)
    high_pairs = [
        hard_high_protection_pair(row, idx, "r5f2_score_risk_main", rng)
        for idx, row in enumerate(high_rows)
    ]

    main_pairs = score_risk_pairs + high_pairs
    plus_pairs = score_risk_pairs + evidence_pairs + high_pairs
    real_only_pairs = list(score_risk_pairs)
    rng.shuffle(main_pairs)
    rng.shuffle(plus_pairs)
    rng.shuffle(real_only_pairs)
    if args.max_pairs > 0:
        main_pairs = main_pairs[: args.max_pairs]
        plus_pairs = plus_pairs[: args.max_pairs]
        real_only_pairs = real_only_pairs[: args.max_pairs]

    write_dataset(args.out_dir, MAIN_DATASET, MAIN_FILE, main_pairs, args.redacted_limit, "r5f2_score_risk_main")
    write_dataset(args.out_dir, PLUS_DATASET, PLUS_FILE, plus_pairs, args.redacted_limit, "r5f2_score_risk_plus_evidence")
    write_dataset(args.out_dir, REAL_ONLY_DATASET, REAL_ONLY_FILE, real_only_pairs, args.redacted_limit, "r5f2_real_only_small")
    write_csv(args.out_dir / "tables" / "r5f2_generation_rejection_stats.csv", stats_rows)
    write_csv(
        args.out_dir / "tables" / "r5f2_rejected_source_counts.csv",
        [{"rejected_source": key, "n": value} for key, value in sorted(source_counter.items())],
    )
    write_csv(
        args.out_dir / "tables" / "r5f2_rejected_pool_counts.csv",
        [{"candidate_pool": key, "n": value} for key, value in sorted(pool_counter.items())],
    )
    quality_rows = [
        pair_quality(MAIN_DATASET, main_pairs),
        pair_quality(PLUS_DATASET, plus_pairs),
        pair_quality(REAL_ONLY_DATASET, real_only_pairs),
    ]
    write_csv(args.out_dir / "tables" / "r5f2_dpo_pair_quality.csv", quality_rows)

    real_ready = (
        len(real_only_pairs) >= args.min_ready_pairs
        and len({pair["source_sample_id"] for pair in real_only_pairs}) >= args.min_ready_samples
    )
    main_ready = (
        len(score_risk_pairs) >= args.min_ready_pairs
        and len({pair["source_sample_id"] for pair in score_risk_pairs}) >= args.min_ready_samples
    )
    decision = {
        "ready_for_real_only_scout": real_ready,
        "ready_for_main_scout": main_ready,
        "score_risk_pairs": len(score_risk_pairs),
        "unique_low_samples": len({pair["source_sample_id"] for pair in score_risk_pairs}),
        "min_ready_pairs": args.min_ready_pairs,
        "min_ready_samples": args.min_ready_samples,
        "recommendation": "train_r5f2_scout" if main_ready else "expand_rejection_mining_or_rethink_dpo_data",
        "guardrails": {
            "no_test_read": True,
            "dev_d1_not_used_as_train_label": True,
            "human_rationale_not_in_prompt": True,
        },
    }
    write_json(args.out_dir / "decision" / "r5f2_rejection_mining_decision.json", decision)

    report = [
        "# Exp19-R5F2 Rejection Mining QC Report",
        "",
        "R5F2 keeps actual train-side generated score-risk rejected outputs and reports whether there is enough data for DPO scout.",
        "",
        f"- score-risk actual rejected pairs: {len(score_risk_pairs)}",
        f"- unique low samples with score-risk rejected: {len({pair['source_sample_id'] for pair in score_risk_pairs})}",
        f"- evidence inconsistency pairs: {len(evidence_pairs)}",
        f"- high protection hard pairs in main: {len(high_pairs)}",
        f"- min ready pairs: {args.min_ready_pairs}",
        f"- min ready samples: {args.min_ready_samples}",
        f"- ready_for_main_scout: `{main_ready}`",
        f"- ready_for_real_only_scout: `{real_ready}`",
        "",
        "## Dataset Versions",
        "",
        f"- `{MAIN_DATASET}`: score-risk low rejected plus high-protection pairs.",
        f"- `{PLUS_DATASET}`: main pairs plus low failure-evidence inconsistency pairs.",
        f"- `{REAL_ONLY_DATASET}`: actual low-to-high generated rejected only.",
        "",
        "No test split is read. Dev D1 annotations are not used for training.",
    ]
    write_text(args.out_dir / "reports" / "r5f2_dpo_qc_report.md", "\n".join(report))
    return {
        "stage": "build_dpo",
        "score_risk_pairs": len(score_risk_pairs),
        "unique_low_samples": len({pair["source_sample_id"] for pair in score_risk_pairs}),
        "ready_for_main_scout": main_ready,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Exp19-R5F2 expanded rejection-mining data.")
    parser.add_argument("--stage", choices=["prepare_inputs", "build_dpo"], default="prepare_inputs")
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--a0-candidates", type=Path, default=DEFAULT_A0_CANDIDATES)
    parser.add_argument("--a0-high-controls", type=Path, default=DEFAULT_A0_HIGH_CONTROLS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_OUT_DIR / "predictions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-answer-chars", type=int, default=40)
    parser.add_argument("--include-borderline-mid", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=0, help="0 keeps all selected candidates.")
    parser.add_argument("--total-samples-per-generator", type=int, default=12)
    parser.add_argument("--max-score-rejected-per-sample", type=int, default=12)
    parser.add_argument("--max-evidence-rejected-per-sample", type=int, default=4)
    parser.add_argument("--max-high-protection-pairs", type=int, default=600)
    parser.add_argument("--max-pairs", type=int, default=4000)
    parser.add_argument("--min-ready-pairs", type=int, default=300)
    parser.add_argument("--min-ready-samples", type=int, default=50)
    parser.add_argument("--redacted-limit", type=int, default=40)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stage == "prepare_inputs":
        summary = prepare_inputs(args)
    else:
        summary = build_dpo(args)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
