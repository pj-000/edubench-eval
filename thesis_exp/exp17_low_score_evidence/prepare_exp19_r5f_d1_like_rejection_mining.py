"""Prepare Exp19-R5F D1-like rejection-mining inputs and DPO data.

R5F is train-only targeted rejection mining for hidden-failure-like low-score
samples. The workflow has two stages:

1. ``prepare_inputs`` builds a LLaMA-Factory generation dataset from train
   samples that resemble D1 hidden failures. It does not train or run inference.
2. ``build_dpo`` reads sampled model generations, keeps actual score-risk
   rejected outputs, and writes a DPO dataset. It still does not read test or
   use dev D1 annotations as training labels.

Human rationale is used only to construct train-side weak labels/targets; it is
never placed in the user prompt.
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
    parsed_payload,
    safe_int,
)
from thesis_exp.exp17_low_score_evidence.prepare_exp19_r5a_onpolicy_dpo import (  # noqa: E402
    score_of,
    target_from_raw_output,
)
from thesis_exp.exp17_low_score_evidence.prepare_exp19_r5c_score_risk_dpo import (  # noqa: E402
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
DEFAULT_D2_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_sft_train_dev_diagnostics_seed42")
DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_r5f_d1_like_rejection_mining_seed42")

GEN_DATASET = "edubench_exp19_r5f_d1_like_generation_train"
GEN_FILE = "data/edubench_exp19_r5f_d1_like_generation_train.json"
DPO_DATASET = "edubench_r5f_d1_like_score_risk_dpo_train"
DPO_FILE = "data/edubench_r5f_d1_like_score_risk_dpo_train.json"

USABLE_USES = {"evidence_positive", "weak_evidence_positive", "pairwise_low", "format_auxiliary"}
USABLE_TYPES = {"strong_evidence_positive", "weak_evidence_positive", "pairwise_low", "format_auxiliary"}
DEFAULT_FAILURE_MODES = {
    "missing_key_point",
    "surface_fluent_but_hidden_defect",
    "insufficient_evidence",
}
ADAPTER_ORDER = [
    "r1b_score_only_balanced",
    "r2n_reason_score_natural",
    "r2c_clean_reason_score_balanced",
    "r4b_shuffled_reason_balanced",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def gold_label(row: dict[str, Any]) -> int:
    return clamp_score(row.get("label_5") or row.get("label") or row.get("gold_label"))


def answer_text(row: dict[str, Any]) -> str:
    return clean(row.get("answer"))


def word_count(text: str) -> int:
    return len([part for part in text.replace("\n", " ").split(" ") if part.strip()])


def stable_sample_id(row: dict[str, Any], idx: int) -> str:
    sid = sample_id(row)
    return sid if sid else f"train_row_{idx:05d}"


def is_usable_signal(signal: dict[str, str] | None, failure_modes: set[str]) -> bool:
    if not signal:
        return False
    use = clean(signal.get("recommended_training_use"))
    candidate_type = clean(signal.get("hidden_failure_candidate_type"))
    failure = normalize_failure_name(signal.get("failure_mode_auto")) or clean(signal.get("failure_mode_auto"))
    if use not in USABLE_USES and candidate_type not in USABLE_TYPES:
        return False
    if candidate_type in {"answer_key_dependent", "conflict_or_exclude", "unclear"}:
        return False
    return failure in failure_modes


def prompt_record(row: dict[str, Any]) -> dict[str, Any]:
    return {"messages": messages_for(row) + [{"role": "assistant", "content": "{}"}]}


def load_d2_high_predictions(d2_dir: Path) -> dict[str, list[dict[str, Any]]]:
    refs_path = d2_dir / "references" / "train_low_reasoned_subset_reference.csv"
    if not refs_path.exists():
        return {}
    refs = read_csv(refs_path)
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for adapter in ADAPTER_ORDER:
        run_dir = d2_dir / "predictions" / adapter / "train_low_reasoned_subset"
        try:
            pred_file = find_prediction_file(run_dir)
        except FileNotFoundError:
            continue
        records = load_prediction_records(pred_file)
        for ref, record in zip(refs, records):
            sid = clean(ref.get("sample_id"))
            payload = parsed_payload(str(record.get("raw_output", "")))
            score = safe_int(payload.get("payload", {}).get("score"))
            failures = payload.get("major_failures") or []
            score_cap = payload.get("score_cap")
            rubric_satisfied = payload.get("rubric_satisfied")
            if not sid:
                continue
            model_high = score is not None and score >= 4
            evidence_risk = (
                score == 3
                and ("no_major_failure" in failures or not failures)
                and score_cap is None
            ) or ("no_major_failure" in failures and rubric_satisfied is True)
            out[sid].append(
                {
                    "adapter": adapter,
                    "score": score,
                    "model_high": model_high,
                    "evidence_risk": evidence_risk,
                    "major_failures": failures,
                    "score_cap": score_cap,
                    "rubric_satisfied": rubric_satisfied,
                }
            )
    return out


def candidate_row(
    row: dict[str, Any],
    signal: dict[str, str],
    idx: int,
    d2_predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    text = answer_text(row)
    model_high_adapters = [item["adapter"] for item in d2_predictions if item.get("model_high")]
    evidence_risk_adapters = [item["adapter"] for item in d2_predictions if item.get("evidence_risk")]
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
        "failure_mode_auto": normalize_failure_name(signal.get("failure_mode_auto")) or clean(signal.get("failure_mode_auto")),
        "hidden_failure_candidate_type": clean(signal.get("hidden_failure_candidate_type")),
        "recommended_training_use": clean(signal.get("recommended_training_use")),
        "confidence_weight": clean(signal.get("confidence_weight")),
        "d2_prediction_count": len(d2_predictions),
        "d2_model_high_count": len(model_high_adapters),
        "d2_evidence_risk_count": len(evidence_risk_adapters),
        "d2_model_high_adapters": "|".join(model_high_adapters),
        "d2_evidence_risk_adapters": "|".join(evidence_risk_adapters),
        "prompt_hash": sha1(user_content(row)),
    }


def prepare_inputs(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.train_jsonl)
    candidates = {clean(row.get("sample_id")): row for row in read_csv_rows(args.a0_candidates)}
    failure_modes = {normalize_failure_name(item) or item for item in args.failure_modes.split(",") if item.strip()}
    d2_predictions = load_d2_high_predictions(args.d2_dir)
    selected: list[tuple[dict[str, Any], dict[str, str], dict[str, Any]]] = []

    for idx, row in enumerate(rows):
        sid = stable_sample_id(row, idx)
        signal = candidates.get(sid)
        text = answer_text(row)
        if gold_label(row) > 2:
            continue
        if args.language != "all" and language(row) != args.language:
            continue
        if len(text) < args.min_answer_chars:
            continue
        if not is_usable_signal(signal, failure_modes):
            continue
        pred_items = d2_predictions.get(sid, [])
        has_model_high = any(item.get("model_high") for item in pred_items)
        if args.require_model_high and not has_model_high:
            continue
        meta = candidate_row(row, signal or {}, idx, pred_items)
        selected.append((row, signal or {}, meta))

    selected.sort(
        key=lambda item: (
            int(item[2]["d2_model_high_count"]),
            int(item[2]["d2_evidence_risk_count"]),
            int(item[2]["answer_chars"]),
        ),
        reverse=True,
    )
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
    write_json(args.out_dir / "dataset_info_r5f_generation_snippet.json", {GEN_DATASET: sft_dataset_entry(GEN_FILE)})
    write_csv(args.out_dir / "tables" / "r5f_d1_like_candidate_pool.csv", candidate_rows)
    write_csv(args.out_dir / "references" / "r5f_d1_like_generation_reference.csv", reference_rows)

    mode_counts = Counter(row["failure_mode_auto"] for row in candidate_rows)
    report = [
        "# Exp19-R5F D1-like Generation Input QC",
        "",
        "This train-only dataset selects low-score samples that resemble hidden-failure cases.",
        "Human rationale is not included in the model prompt.",
        "",
        f"- train source: `{args.train_jsonl}`",
        f"- candidates selected: {len(candidate_rows)}",
        f"- language filter: `{args.language}`",
        f"- min answer chars: {args.min_answer_chars}",
        f"- require model-high D2 prediction: `{args.require_model_high}`",
        f"- D2 prediction map available for samples: {len(d2_predictions)}",
        "",
        "## Failure Modes",
        "",
    ]
    report.extend(f"- {key}: {value}" for key, value in sorted(mode_counts.items()))
    report.extend(
        [
            "",
            "## Next Step",
            "",
            "Run `run_exp19_r5f_d1_like_generate.sh` on server to sample rejected outputs.",
            "Then run the same preparation script with `--stage build_dpo` to construct R5F DPO pairs.",
            "",
            "No test split is read.",
        ]
    )
    write_text(args.out_dir / "reports" / "r5f_d1_like_generation_input_report.md", "\n".join(report))
    return {
        "stage": "prepare_inputs",
        "candidates": len(candidate_rows),
        "dataset": GEN_DATASET,
        "d2_prediction_sample_count": len(d2_predictions),
    }


def generation_dirs(prediction_root: Path, adapter: str, num_samples: int) -> list[Path]:
    return [prediction_root / adapter / f"sample_{idx}" for idx in range(num_samples)]


def load_generated_targets(args: argparse.Namespace, reference_rows: list[dict[str, str]]) -> dict[str, list[tuple[str, dict[str, Any]]]]:
    out: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for adapter in ADAPTER_ORDER:
        for sample_idx, run_dir in enumerate(generation_dirs(args.prediction_root, adapter, args.num_samples_per_input)):
            try:
                pred_file = find_prediction_file(run_dir)
            except FileNotFoundError:
                continue
            records = load_prediction_records(pred_file)
            for ref, record in zip(reference_rows, records):
                sid = clean(ref.get("sample_id"))
                target = target_from_raw_output(str(record.get("raw_output", "")))
                if sid and target is not None:
                    out[sid].append((f"{adapter}_sample{sample_idx}", target))
    return out


def is_score_risk_rejected(target: dict[str, Any]) -> bool:
    score = score_of(target)
    return score is not None and score >= 4


def is_evidence_risk_rejected(target: dict[str, Any]) -> bool:
    score = score_of(target)
    if score == 3 and has_no_major_failure(target) and target.get("score_cap") is None:
        return True
    return has_no_major_failure(target) and target.get("rubric_satisfied") is True


def dpo_example(
    row: dict[str, Any],
    signal: dict[str, str],
    rejected: dict[str, Any],
    rejected_source: str,
    copy_index: int,
) -> dict[str, Any]:
    return {
        "messages": messages_for(row),
        "chosen": {"role": "assistant", "content": assistant_json(evidence_target(row, signal, include_reason=False))},
        "rejected": {"role": "assistant", "content": assistant_json(rejected)},
        "risk_type": "d1_like_low_to_high_score_risk",
        "rejected_source": rejected_source,
        "rejected_category": "actual_model_generation",
        "source_sample_id": sample_id(row),
        "dataset_variant": "r5f_d1_like_score_risk",
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
    }


def build_dpo(args: argparse.Namespace) -> dict[str, Any]:
    train_rows = {sample_id(row): row for row in read_jsonl(args.train_jsonl)}
    candidates = {clean(row.get("sample_id")): row for row in read_csv_rows(args.a0_candidates)}
    reference_rows = read_csv(args.out_dir / "references" / "r5f_d1_like_generation_reference.csv")
    generated = load_generated_targets(args, reference_rows)
    rng = random.Random(args.seed)

    pairs: list[dict[str, Any]] = []
    candidate_stats: list[dict[str, Any]] = []
    for ref in reference_rows:
        sid = clean(ref.get("sample_id"))
        row = train_rows.get(sid)
        signal = candidates.get(sid)
        if row is None or signal is None:
            continue
        targets = generated.get(sid, [])
        score_risk = [(source, target) for source, target in targets if is_score_risk_rejected(target)]
        evidence_risk = [(source, target) for source, target in targets if is_evidence_risk_rejected(target)]
        rng.shuffle(score_risk)
        for copy_index, (source, rejected) in enumerate(score_risk[: args.max_rejected_per_sample]):
            pairs.append(dpo_example(row, signal, rejected, source, copy_index))
        candidate_stats.append(
            {
                "sample_id": sid,
                "generated_count": len(targets),
                "score_risk_rejected_count": len(score_risk),
                "evidence_risk_rejected_count": len(evidence_risk),
                "used_score_risk_pairs": min(len(score_risk), args.max_rejected_per_sample),
                "failure_mode_auto": ref.get("failure_mode_auto", ""),
                "metric": ref.get("metric", ""),
                "language": ref.get("language", ""),
            }
        )

    rng.shuffle(pairs)
    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]

    write_json_array(args.out_dir / DPO_FILE, pairs)
    write_json(args.out_dir / "dataset_info_r5f_snippet.json", {DPO_DATASET: dpo_dataset_entry(DPO_FILE)})
    write_csv(args.out_dir / "tables" / "r5f_d1_like_generation_rejection_stats.csv", candidate_stats)
    source_rows = []
    for source, n in sorted(Counter(pair["rejected_source"] for pair in pairs).items()):
        source_rows.append({"rejected_source": source, "n": n})
    write_csv(args.out_dir / "tables" / "r5f_d1_like_dpo_rejected_source_counts.csv", source_rows)

    quality_rows = []
    if pairs:
        chosen_scores = [score_of(target_from_message(pair["chosen"])) for pair in pairs]
        rejected_scores = [score_of(target_from_message(pair["rejected"])) for pair in pairs]
        quality_rows.append(
            {
                "dataset": DPO_DATASET,
                "pairs": len(pairs),
                "unique_low_samples": len({pair["source_sample_id"] for pair in pairs}),
                "chosen_score_mean": sum(float(x) for x in chosen_scores if x is not None) / max(1, sum(x is not None for x in chosen_scores)),
                "rejected_score_mean": sum(float(x) for x in rejected_scores if x is not None) / max(1, sum(x is not None for x in rejected_scores)),
                "score_risk_actual_rejected_rate": 1.0,
            }
        )
    write_csv(args.out_dir / "tables" / "r5f_d1_like_dpo_pair_quality.csv", quality_rows)
    write_json(args.out_dir / "redacted_samples" / "r5f_d1_like_score_risk_redacted.json", [redacted_pair(pair) for pair in pairs[: args.redacted_limit]])

    write_dpo_config(
        args.out_dir / "configs" / "llamafactory_qwen3_4b_r5f_dpo_from_r1b.yaml",
        DPO_DATASET,
        "saves/edubench/qwen3-4b/r1_score_only_balanced_lora",
        "saves/edubench/qwen3-4b/r5f_dpo_from_r1b_lora",
        args.out_dir,
    )
    write_dpo_config(
        args.out_dir / "configs" / "llamafactory_qwen3_4b_r5f_dpo_from_r2c.yaml",
        DPO_DATASET,
        "saves/edubench/qwen3-4b/r2_clean_reason_score_balanced_lora",
        "saves/edubench/qwen3-4b/r5f_dpo_from_r2c_lora",
        args.out_dir,
    )

    ready = len(pairs) >= args.min_ready_pairs and len({pair["source_sample_id"] for pair in pairs}) >= args.min_ready_samples
    report = [
        "# Exp19-R5F D1-like DPO QC Report",
        "",
        "This dataset keeps only actual generated score-risk rejected outputs with score >= 4.",
        "",
        f"- DPO pairs: {len(pairs)}",
        f"- unique low samples: {len({pair['source_sample_id'] for pair in pairs})}",
        f"- min ready pairs: {args.min_ready_pairs}",
        f"- min ready samples: {args.min_ready_samples}",
        f"- ready_for_scout: `{ready}`",
        "",
        "No test split is read. Dev D1 annotations are not used for training.",
    ]
    write_text(args.out_dir / "reports" / "r5f_d1_like_dpo_qc_report.md", "\n".join(report))
    return {
        "stage": "build_dpo",
        "pairs": len(pairs),
        "unique_low_samples": len({pair["source_sample_id"] for pair in pairs}),
        "ready_for_scout": ready,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Exp19-R5F D1-like rejection-mining data.")
    parser.add_argument("--stage", choices=["prepare_inputs", "build_dpo"], default="prepare_inputs")
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--a0-candidates", type=Path, default=DEFAULT_A0_CANDIDATES)
    parser.add_argument("--d2-dir", type=Path, default=DEFAULT_D2_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_OUT_DIR / "predictions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--language", default="all", help="Use `all` to disable language filtering.")
    parser.add_argument("--failure-modes", default="missing_key_point,surface_fluent_but_hidden_defect,insufficient_evidence")
    parser.add_argument("--min-answer-chars", type=int, default=40)
    parser.add_argument("--max-candidates", type=int, default=240)
    parser.add_argument("--require-model-high", action="store_true")
    parser.add_argument("--num-samples-per-input", type=int, default=4)
    parser.add_argument("--max-rejected-per-sample", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=3000)
    parser.add_argument("--min-ready-pairs", type=int, default=300)
    parser.add_argument("--min-ready-samples", type=int, default=50)
    parser.add_argument("--redacted-limit", type=int, default=30)
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
