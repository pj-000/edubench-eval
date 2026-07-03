"""Exp19-SFT-D2 train-vs-dev structured failure behavior diagnostic.

This module prepares small LLaMA-Factory prediction datasets and collects
adapter outputs for diagnosis. It does not train a model and it never reads the
test split. Human-rationale-derived labels are used only as evaluation
references, not as model prompt inputs.
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
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence.collect_exp19_sft_first_round_dev_results import (  # noqa: E402
    FAILURE_EVAL_TYPES,
    find_prediction_file,
    has_no_major_failure,
    has_substantive_failure,
    load_d1_annotations,
    load_prediction_records,
    normalize_failure_list,
    normalize_failure_name,
    parsed_payload,
    safe_float,
    safe_int,
)
from thesis_exp.exp17_low_score_evidence.prepare_exp19_sft_dpo_datasets import (  # noqa: E402
    OPENAI_TAGS,
    clean,
    clamp_score,
    language,
    messages_for,
    metric_name,
    question_group,
    read_csv_rows,
    read_jsonl,
    sample_id,
    sft_dataset_entry,
    subject,
)
from thesis_exp.exp17_low_score_evidence.run_exp19_r0a_qwen4b_direct_baseline import (  # noqa: E402
    LABELS,
    mean,
    parse_score,
    quadratic_weighted_kappa,
    safe_rate,
)
from thesis_exp.src.edujudge.utils.io import write_csv, write_json, write_text  # noqa: E402


DEFAULT_TRAIN_JSONL = Path("thesis_exp/data/splits/question_seed42/train.jsonl")
DEFAULT_DEV_JSONL = Path("thesis_exp/data/splits/question_seed42/dev.jsonl")
DEFAULT_A0_CANDIDATES = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp17_a0_train_hidden_failure_signals_seed42/"
    "train_hidden_failure_candidates.csv"
)
DEFAULT_A0_HIGH_CONTROLS = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp17_a0_train_hidden_failure_signals_seed42/"
    "train_clean_high_controls.csv"
)
DEFAULT_D1_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/d1_hidden_failure_audit_seed42_dev/"
    "summary_human_rationale_recovered"
)
DEFAULT_OUT_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp19_sft_train_dev_diagnostics_seed42"
)

SUBSETS = [
    "train_low_reasoned_subset",
    "train_clean_high_subset",
    "dev_label2_subset",
    "dev_d1_hidden_subset",
    "dev_d1_matched_controls",
    "dev_high_subset",
]
ADAPTERS = {
    "r2n_reason_score_natural": "R2n reason-score natural",
    "r2c_clean_reason_score_balanced": "R2c clean reason-score balanced",
    "r4b_shuffled_reason_balanced": "R4b shuffled reason balanced",
    "r1b_score_only_balanced": "R1b score-only balanced",
}
USABLE_TRAINING_USES = {
    "evidence_positive",
    "weak_evidence_positive",
    "pairwise_low",
    "format_auxiliary",
}
USABLE_CANDIDATE_TYPES = {
    "strong_evidence_positive",
    "weak_evidence_positive",
    "pairwise_low",
    "format_auxiliary",
}
EXCLUDED_FAILURE_MODES = {"", "unclear", "possible_label_conflict", "no_major_failure"}
STRUCTURED_FIELDS = [
    "adapter",
    "subset",
    "n",
    "major_failure_nonempty_rate",
    "no_major_failure_rate",
    "score_cap_nonnull_rate",
    "rubric_satisfied_true_rate",
    "valid_full_schema_rate",
    "score_json_parse_rate",
]
METRIC_FIELDS = [
    "adapter",
    "subset",
    "n",
    "valid_n",
    "MAE",
    "QWK",
    "mean_pred",
    "pred_ge4_rate",
    "pred_le2_rate",
    "low_to_high_rate",
    "label2_recall",
    "label2_pred_ge4_rate",
    "high_to_low_rate",
    "parse_success_rate",
]
FAILURE_TYPE_FIELDS = [
    "adapter",
    "subset",
    "n_gold_failure",
    "failure_type_micro_f1",
    "failure_type_macro_f1",
    "missing_key_point_recall",
    "factual_or_rubric_mismatch_recall",
    "insufficient_evidence_recall",
    "task_constraint_violation_recall",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def gold_label(row: dict[str, Any]) -> int:
    return clamp_score(row.get("label_5") or row.get("label") or row.get("gold_label"))


def stable_sample_id(row: dict[str, Any], idx: int) -> str:
    sid = sample_id(row)
    return sid if sid else f"row_{idx:05d}"


def dataset_name(subset: str) -> str:
    return f"edubench_exp19_d2_{subset}"


def prompt_record(row: dict[str, Any]) -> dict[str, Any]:
    # The dummy assistant content is not included in the prompt context used for
    # generation; gold labels and human rationales live only in reference CSVs.
    return {"messages": messages_for(row) + [{"role": "assistant", "content": "{}"}]}


def reference_row(
    row: dict[str, Any],
    idx: int,
    subset: str,
    split: str,
    failure_gold: str = "",
    failure_source: str = "",
) -> dict[str, Any]:
    return {
        "eval_index": idx,
        "subset": subset,
        "split": split,
        "sample_id": stable_sample_id(row, idx),
        "record_id": clean(row.get("record_id")),
        "question_key": clean(row.get("question_key")),
        "question_group_id": question_group(row),
        "metric": metric_name(row),
        "metric_id": clean(row.get("metric_id")),
        "language": language(row),
        "subject": subject(row),
        "gold_label": gold_label(row),
        "failure_gold": failure_gold,
        "failure_source": failure_source,
    }


def sample_maps(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {stable_sample_id(row, idx): row for idx, row in enumerate(rows)}


def is_usable_train_failure(row: dict[str, str]) -> bool:
    try:
        label = int(float(clean(row.get("gold_label"))))
    except ValueError:
        return False
    if label > 2:
        return False
    use = clean(row.get("recommended_training_use"))
    candidate_type = clean(row.get("hidden_failure_candidate_type"))
    failure_mode = normalize_failure_name(row.get("failure_mode_auto")) or clean(row.get("failure_mode_auto"))
    if use not in USABLE_TRAINING_USES and candidate_type not in USABLE_CANDIDATE_TYPES:
        return False
    if candidate_type in {"answer_key_dependent", "conflict_or_exclude", "unclear"}:
        return False
    return failure_mode not in EXCLUDED_FAILURE_MODES


def stratified_sample(
    rows: list[dict[str, Any]],
    max_rows: int,
    seed: int,
    key_fn: Callable[[dict[str, Any]], str],
) -> list[dict[str, Any]]:
    if max_rows <= 0 or len(rows) <= max_rows:
        return list(rows)
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    for items in groups.values():
        rng.shuffle(items)
    selected: list[dict[str, Any]] = []
    keys = sorted(groups)
    while len(selected) < max_rows and keys:
        next_keys: list[str] = []
        for key in keys:
            items = groups[key]
            if items and len(selected) < max_rows:
                selected.append(items.pop())
            if items:
                next_keys.append(key)
        keys = next_keys
    return selected


def high_strata_key(row: dict[str, Any]) -> str:
    return f"{metric_name(row)}::{language(row)}::{subject(row)}"


def prepare_subsets(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    train_rows = read_jsonl(args.train_jsonl)
    dev_rows = read_jsonl(args.dev_jsonl)
    train_by_id = sample_maps(train_rows)
    dev_by_id = sample_maps(dev_rows)
    a0_candidates = {clean(row.get("sample_id")): row for row in read_csv_rows(args.a0_candidates)}
    a0_high_controls = {clean(row.get("sample_id")): row for row in read_csv_rows(args.a0_high_controls)}
    d1_annotations, _d1_pairs, d1_control_ids = load_d1_annotations(args.d1_dir)

    train_low_rows: list[dict[str, Any]] = []
    train_low_refs: dict[str, str] = {}
    for sid, signal in sorted(a0_candidates.items()):
        row = train_by_id.get(sid)
        if row is None or not is_usable_train_failure(signal):
            continue
        train_low_rows.append(row)
        train_low_refs[sid] = normalize_failure_name(signal.get("failure_mode_auto")) or "unclear"

    train_high_rows = [train_by_id[sid] for sid in sorted(a0_high_controls) if sid in train_by_id]
    train_high_rows = [row for row in train_high_rows if gold_label(row) >= 4]
    train_high_rows = stratified_sample(train_high_rows, args.max_train_clean_high, args.seed, high_strata_key)

    dev_label2_rows = [row for row in dev_rows if gold_label(row) == 2]

    d1_hidden_rows = [dev_by_id[sid] for sid in sorted(d1_annotations) if sid in dev_by_id]
    d1_hidden_refs = {
        sid: normalize_failure_name(row.get("primary_failure_mode_manual")) or "unclear"
        for sid, row in d1_annotations.items()
    }

    d1_control_rows = [dev_by_id[sid] for sid in sorted(d1_control_ids) if sid in dev_by_id]
    dev_high_rows = [row for row in dev_rows if gold_label(row) >= 4]
    dev_high_rows = stratified_sample(dev_high_rows, args.max_dev_high, args.seed, high_strata_key)

    subsets: dict[str, list[dict[str, Any]]] = {
        "train_low_reasoned_subset": train_low_rows,
        "train_clean_high_subset": train_high_rows,
        "dev_label2_subset": dev_label2_rows,
        "dev_d1_hidden_subset": d1_hidden_rows,
        "dev_d1_matched_controls": d1_control_rows,
        "dev_high_subset": dev_high_rows,
    }
    failure_refs: dict[str, dict[str, str]] = {
        "train_low_reasoned_subset": train_low_refs,
        "dev_d1_hidden_subset": d1_hidden_refs,
    }

    data_dir = args.out_dir / "data"
    references_dir = args.out_dir / "references"
    tables_dir = args.out_dir / "tables"
    reports_dir = args.out_dir / "reports"
    dataset_info: dict[str, Any] = {}
    counts: list[dict[str, Any]] = []

    for subset, rows in subsets.items():
        file_name = f"data/{dataset_name(subset)}.json"
        write_json(args.out_dir / file_name, [prompt_record(row) for row in rows])
        entry = sft_dataset_entry(file_name)
        if entry.get("tags") != OPENAI_TAGS:
            raise ValueError("Unexpected ShareGPT tag mapping.")
        dataset_info[dataset_name(subset)] = entry

        refs = []
        for idx, row in enumerate(rows):
            sid = sample_id(row)
            failure_gold = ""
            failure_source = ""
            if subset in {"train_clean_high_subset", "dev_d1_matched_controls", "dev_high_subset"}:
                failure_gold = "no_major_failure"
                failure_source = "clean_high_control"
            elif subset in failure_refs and sid in failure_refs[subset]:
                failure_gold = failure_refs[subset][sid]
                failure_source = "a0_train_signal" if subset.startswith("train_") else "d1_manual_reference"
            refs.append(reference_row(row, idx, subset, "train" if subset.startswith("train_") else "dev", failure_gold, failure_source))
        write_csv(references_dir / f"{subset}_reference.csv", refs)
        counts.append(
            {
                "subset": subset,
                "dataset_name": dataset_name(subset),
                "split": "train" if subset.startswith("train_") else "dev",
                "n": len(rows),
                "source": str(args.train_jsonl if subset.startswith("train_") else args.dev_jsonl),
                "reference_csv": str(references_dir / f"{subset}_reference.csv"),
            }
        )

    write_json(args.out_dir / "dataset_info.json", dataset_info)
    write_csv(tables_dir / "exp19_sft_train_dev_subset_counts.csv", counts)

    report = [
        "# Exp19-SFT-D2 Diagnostic Datasets",
        "",
        "This prepare step builds prompt-only LLaMA-Factory datasets for train-vs-dev structured failure behavior diagnosis.",
        "The user prompt contains only question, answer, metric, rubric, and metadata.",
        "Gold labels, A0 weak labels, and D1 annotations are stored only in reference CSV files for evaluation.",
        "",
        "| subset | split | n | purpose |",
        "|---|---|---:|---|",
    ]
    purposes = {
        "train_low_reasoned_subset": "Check whether R2/R2c learned to emit failure fields on train low-score examples with recovered reasons.",
        "train_clean_high_subset": "Check whether high-score train controls are protected from false failure assignment.",
        "dev_label2_subset": "Check whether dev label-2 samples are still over-scored.",
        "dev_d1_hidden_subset": "Check whether recovered hidden-failure cases generalize on dev.",
        "dev_d1_matched_controls": "Check whether matched high controls remain high and mostly no-failure.",
        "dev_high_subset": "Check high-score protection on a stratified dev-high subset.",
    }
    for row in counts:
        report.append(f"| {row['subset']} | {row['split']} | {row['n']} | {purposes[row['subset']]} |")
    report.extend(
        [
            "",
            "This step does not read test, does not train, and does not use human rationale as prompt input.",
        ]
    )
    write_text(reports_dir / "exp19_sft_train_dev_prepare_report.md", "\n".join(report))
    # Touch directories for predictable server paths.
    for path in (data_dir, references_dir, tables_dir, reports_dir):
        path.mkdir(parents=True, exist_ok=True)
    return subsets


def load_reference(out_dir: Path, subset: str) -> list[dict[str, str]]:
    return read_csv(out_dir / "references" / f"{subset}_reference.csv")


def align_prediction_rows(
    reference: list[dict[str, str]],
    prediction_records: list[dict[str, Any]],
    adapter: str,
    subset: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = min(len(reference), len(prediction_records))
    if len(reference) != len(prediction_records):
        print(
            f"WARNING {adapter}/{subset}: reference rows={len(reference)} prediction rows={len(prediction_records)}; aligned={n}",
            file=sys.stderr,
        )
    for idx in range(n):
        ref = reference[idx]
        raw_output = str(prediction_records[idx].get("raw_output", ""))
        parsed = parse_score(raw_output)
        structured = parsed_payload(raw_output)
        pred = parsed.get("pred_label")
        rows.append(
            {
                "adapter": adapter,
                "subset": subset,
                "eval_index": int(ref["eval_index"]),
                "split": ref.get("split", ""),
                "sample_id": ref.get("sample_id", ""),
                "record_id": ref.get("record_id", ""),
                "question_key": ref.get("question_key", ""),
                "question_group_id": ref.get("question_group_id", ""),
                "metric": ref.get("metric", ""),
                "metric_id": ref.get("metric_id", ""),
                "language": ref.get("language", ""),
                "subject": ref.get("subject", ""),
                "gold_label": int(float(ref["gold_label"])),
                "failure_gold": ref.get("failure_gold", ""),
                "failure_source": ref.get("failure_source", ""),
                "pred_label": pred,
                "parse_success": bool(parsed.get("parse_success")),
                "parse_method": parsed.get("parse_method", "failed"),
                "score_json_parse_ok": bool(structured["score_json_parse_ok"]),
                "major_failures": structured["major_failures"],
                "major_failures_parse_ok": bool(structured["major_failures_parse_ok"]),
                "score_cap": structured["score_cap"],
                "score_cap_present": bool(structured["score_cap_present"]),
                "score_cap_parse_ok": bool(structured["score_cap_parse_ok"]),
                "rubric_satisfied": structured["rubric_satisfied"],
                "rubric_satisfied_present": bool(structured["rubric_satisfied_present"]),
                "rubric_satisfied_parse_ok": bool(structured["rubric_satisfied_parse_ok"]),
                "brief_reason": structured["brief_reason"],
                "valid_full_schema": bool(structured["valid_full_schema"]),
            }
        )
    return rows


def qwk_for(gold: list[int], pred: list[int]) -> float:
    if len(set(gold)) < 2 or len(set(pred)) < 2:
        return float("nan")
    return quadratic_weighted_kappa(gold, pred)


def diagnostic_metric_row(rows: list[dict[str, Any]], adapter: str, subset: str) -> dict[str, Any]:
    valid = [row for row in rows if safe_int(row.get("pred_label")) is not None and row.get("parse_success")]
    gold = [int(row["gold_label"]) for row in valid]
    pred = [int(row["pred_label"]) for row in valid]
    low = [row for row in rows if int(row["gold_label"]) <= 2]
    high = [row for row in rows if int(row["gold_label"]) >= 4]
    label2 = [row for row in rows if int(row["gold_label"]) == 2]
    label2_hits = sum(1 for row in label2 if safe_int(row.get("pred_label")) == 2)
    label2_ge4 = sum(1 for row in label2 if safe_int(row.get("pred_label")) is not None and int(row["pred_label"]) >= 4)
    low_to_high = sum(1 for row in low if safe_int(row.get("pred_label")) is not None and int(row["pred_label"]) >= 4)
    high_to_low = sum(1 for row in high if safe_int(row.get("pred_label")) is not None and int(row["pred_label"]) <= 2)
    return {
        "adapter": adapter,
        "subset": subset,
        "n": len(rows),
        "valid_n": len(valid),
        "MAE": mean([abs(g - p) for g, p in zip(gold, pred)]),
        "QWK": qwk_for(gold, pred),
        "mean_pred": mean(pred),
        "pred_ge4_rate": safe_rate(sum(1 for value in pred if value >= 4), len(valid)),
        "pred_le2_rate": safe_rate(sum(1 for value in pred if value <= 2), len(valid)),
        "low_to_high_rate": safe_rate(low_to_high, len(low)),
        "label2_recall": safe_rate(label2_hits, len(label2)),
        "label2_pred_ge4_rate": safe_rate(label2_ge4, len(label2)),
        "high_to_low_rate": safe_rate(high_to_low, len(high)),
        "parse_success_rate": safe_rate(len(valid), len(rows)),
    }


def structured_behavior_row(rows: list[dict[str, Any]], adapter: str, subset: str) -> dict[str, Any]:
    n = len(rows)
    return {
        "adapter": adapter,
        "subset": subset,
        "n": n,
        "major_failure_nonempty_rate": safe_rate(sum(1 for row in rows if has_substantive_failure(row)), n),
        "no_major_failure_rate": safe_rate(sum(1 for row in rows if has_no_major_failure(row)), n),
        "score_cap_nonnull_rate": safe_rate(sum(1 for row in rows if row.get("score_cap") is not None), n),
        "rubric_satisfied_true_rate": safe_rate(
            sum(1 for row in rows if row.get("rubric_satisfied") is True),
            n,
        ),
        "valid_full_schema_rate": safe_rate(sum(1 for row in rows if row.get("valid_full_schema")), n),
        "score_json_parse_rate": safe_rate(sum(1 for row in rows if row.get("score_json_parse_ok")), n),
    }


def f1_score(tp: int, fp: int, fn: int) -> float:
    den = (2 * tp) + fp + fn
    return safe_rate(2 * tp, den)


def recall_for(counts: dict[str, Counter[str]], name: str) -> float:
    return safe_rate(counts[name]["tp"], counts[name]["tp"] + counts[name]["fn"])


def failure_type_eval_row(rows: list[dict[str, Any]], adapter: str, subset: str) -> dict[str, Any]:
    counts: dict[str, Counter[str]] = {name: Counter() for name in FAILURE_EVAL_TYPES}
    micro = Counter()
    n_gold = 0
    for row in rows:
        gold = normalize_failure_name(row.get("failure_gold")) or "unclear"
        if gold not in FAILURE_EVAL_TYPES:
            continue
        n_gold += 1
        pred_eval = {name for name in normalize_failure_list(row.get("major_failures")) if name in FAILURE_EVAL_TYPES}
        if gold in pred_eval:
            counts[gold]["tp"] += 1
            micro["tp"] += 1
        else:
            counts[gold]["fn"] += 1
            micro["fn"] += 1
        for pred_name in pred_eval:
            if pred_name != gold:
                counts[pred_name]["fp"] += 1
                micro["fp"] += 1
    macro_values = [
        f1_score(counts[name]["tp"], counts[name]["fp"], counts[name]["fn"])
        for name in FAILURE_EVAL_TYPES
        if sum(counts[name].values()) > 0
    ]
    return {
        "adapter": adapter,
        "subset": subset,
        "n_gold_failure": n_gold,
        "failure_type_micro_f1": f1_score(micro["tp"], micro["fp"], micro["fn"]),
        "failure_type_macro_f1": mean(macro_values),
        "missing_key_point_recall": recall_for(counts, "missing_key_point"),
        "factual_or_rubric_mismatch_recall": recall_for(counts, "factual_or_rubric_mismatch"),
        "insufficient_evidence_recall": recall_for(counts, "insufficient_evidence"),
        "task_constraint_violation_recall": recall_for(counts, "task_constraint_violation"),
    }


def load_predictions_for(args: argparse.Namespace, adapter: str, subset: str) -> list[dict[str, Any]]:
    reference = load_reference(args.out_dir, subset)
    run_dir = args.prediction_root / adapter / subset
    prediction_file = find_prediction_file(run_dir)
    records = load_prediction_records(prediction_file)
    return align_prediction_rows(reference, records, adapter, subset)


def row_lookup(rows: list[dict[str, Any]], adapter: str, subset: str) -> dict[str, Any]:
    return next((row for row in rows if row.get("adapter") == adapter and row.get("subset") == subset), {})


def metric_value(rows: list[dict[str, Any]], adapter: str, subset: str, key: str) -> float:
    return safe_float(row_lookup(rows, adapter, subset).get(key))


def decision_json(metric_rows: list[dict[str, Any]], structured_rows: list[dict[str, Any]]) -> dict[str, Any]:
    adapter = "r2c_clean_reason_score_balanced"
    train_nonempty = metric_value(structured_rows, adapter, "train_low_reasoned_subset", "major_failure_nonempty_rate")
    dev_nonempty = metric_value(structured_rows, adapter, "dev_d1_hidden_subset", "major_failure_nonempty_rate")
    dev_d1_pred_ge4 = metric_value(metric_rows, adapter, "dev_d1_hidden_subset", "pred_ge4_rate")
    dev_label2_pred_ge4 = metric_value(metric_rows, adapter, "dev_label2_subset", "label2_pred_ge4_rate")

    category = "redesign_sft_schema"
    recommendation = "Inspect prediction JSON outputs before deciding the next training step."
    if not math.isnan(train_nonempty) and train_nonempty < 0.30:
        category = "failure_not_learned_even_on_train"
        recommendation = "redesign_sft_schema"
    elif not math.isnan(train_nonempty) and train_nonempty >= 0.50 and not math.isnan(dev_nonempty) and dev_nonempty < 0.20:
        category = "failure_learned_on_train_not_dev"
        recommendation = "DPO or data augmentation, but do not run test/all_5536 yet."
    risk_values = [x for x in [dev_d1_pred_ge4, dev_label2_pred_ge4] if not math.isnan(x)]
    if category == "redesign_sft_schema" and risk_values and max(risk_values) > 0.50:
        category = "failure_learned_but_score_not_calibrated"
        recommendation = "risk-balanced DPO"
    elif category == "redesign_sft_schema" and not math.isnan(train_nonempty) and train_nonempty >= 0.30:
        category = "proceed_to_dpo"
        recommendation = "Proceed to a small risk-balanced DPO scout; still keep test/all_5536 held out."

    return {
        "decision_category": category,
        "recommendation": recommendation,
        "r2c_train_low_reasoned_major_failure_nonempty_rate": None if math.isnan(train_nonempty) else train_nonempty,
        "r2c_dev_d1_hidden_major_failure_nonempty_rate": None if math.isnan(dev_nonempty) else dev_nonempty,
        "r2c_dev_d1_hidden_pred_ge4_rate": None if math.isnan(dev_d1_pred_ge4) else dev_d1_pred_ge4,
        "r2c_dev_label2_pred_ge4_rate": None if math.isnan(dev_label2_pred_ge4) else dev_label2_pred_ge4,
        "recommend_test": False,
        "recommend_all_5536": False,
        "guardrail": "No test/all_5536 evaluation is recommended by D2.",
    }


def fmt(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except Exception:
        return str(value)
    if math.isnan(val):
        return "nan"
    return f"{val:.{digits}f}"


def report_markdown(
    metric_rows: list[dict[str, Any]],
    structured_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    decision: dict[str, Any],
) -> str:
    def structured(adapter: str, subset: str, key: str) -> float:
        return metric_value(structured_rows, adapter, subset, key)

    def metric(adapter: str, subset: str, key: str) -> float:
        return metric_value(metric_rows, adapter, subset, key)

    r2c_train_nonempty = structured("r2c_clean_reason_score_balanced", "train_low_reasoned_subset", "major_failure_nonempty_rate")
    r2c_dev_nonempty = structured("r2c_clean_reason_score_balanced", "dev_d1_hidden_subset", "major_failure_nonempty_rate")
    r4b_train_nonempty = structured("r4b_shuffled_reason_balanced", "train_low_reasoned_subset", "major_failure_nonempty_rate")
    r2c_high_false = structured("r2c_clean_reason_score_balanced", "train_clean_high_subset", "major_failure_nonempty_rate")
    r2c_d1_pred_ge4 = metric("r2c_clean_reason_score_balanced", "dev_d1_hidden_subset", "pred_ge4_rate")

    lines = [
        "# Exp19-SFT-D2 Train-vs-Dev Structured Failure Diagnostic",
        "",
        "This diagnostic uses existing SFT adapters for prediction only. It does not train, does not read test, and does not put human rationale or failure labels into prompts.",
        "",
        "## Key Answers",
        "",
        f"- Does R2n/R2c output major failures on train low reasoned examples? R2c nonempty rate is `{fmt(r2c_train_nonempty)}`.",
        f"- If train is learned, does it generalize to dev D1 hidden cases? R2c dev D1 nonempty rate is `{fmt(r2c_dev_nonempty)}`.",
        f"- Does R4b behave similarly to R2c? R4b train low nonempty rate is `{fmt(r4b_train_nonempty)}`; compare this with R2c above.",
        f"- Are high controls protected? R2c train clean-high nonempty failure rate is `{fmt(r2c_high_false)}`.",
        f"- Is score calibration still risky? R2c dev D1 pred>=4 rate is `{fmt(r2c_d1_pred_ge4)}`.",
        f"- Decision: `{decision.get('decision_category')}`; recommendation: {decision.get('recommendation')}.",
        "",
        "## Diagnostic Metrics",
        "",
        "| adapter | subset | n | MAE | QWK | mean pred | pred>=4 | pred<=2 | label2 recall | high-to-low | parse |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            f"| {row['adapter']} | {row['subset']} | {row['n']} | {fmt(row['MAE'])} | {fmt(row['QWK'])} | "
            f"{fmt(row['mean_pred'])} | {fmt(row['pred_ge4_rate'])} | {fmt(row['pred_le2_rate'])} | "
            f"{fmt(row['label2_recall'])} | {fmt(row['high_to_low_rate'])} | {fmt(row['parse_success_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Structured Behavior",
            "",
            "| adapter | subset | n | nonempty failure | no major failure | score cap nonnull | rubric true | full schema |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in structured_rows:
        lines.append(
            f"| {row['adapter']} | {row['subset']} | {row['n']} | {fmt(row['major_failure_nonempty_rate'])} | "
            f"{fmt(row['no_major_failure_rate'])} | {fmt(row['score_cap_nonnull_rate'])} | "
            f"{fmt(row['rubric_satisfied_true_rate'])} | {fmt(row['valid_full_schema_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Failure Type Evaluation",
            "",
            "| adapter | subset | gold failure n | micro-F1 | macro-F1 | missing recall | factual/rubric recall | insufficient recall | task recall |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in failure_rows:
        lines.append(
            f"| {row['adapter']} | {row['subset']} | {row['n_gold_failure']} | {fmt(row['failure_type_micro_f1'])} | "
            f"{fmt(row['failure_type_macro_f1'])} | {fmt(row['missing_key_point_recall'])} | "
            f"{fmt(row['factual_or_rubric_mismatch_recall'])} | {fmt(row['insufficient_evidence_recall'])} | "
            f"{fmt(row['task_constraint_violation_recall'])} |"
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- D2 does not read the test split.",
            "- D2 does not train any model.",
            "- Human rationale is used only through A0/D1 evaluation references, never as prompt input.",
            "- Raw predictions, configs, logs, and prompt datasets remain gitignored.",
            "- This diagnostic should not recommend test/all_5536.",
        ]
    )
    return "\n".join(lines)


def collect(args: argparse.Namespace) -> dict[str, Any]:
    metric_rows: list[dict[str, Any]] = []
    structured_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    adapters = [item for item in args.adapters.split(",") if item]
    subsets = [item for item in args.subsets.split(",") if item]

    for adapter in adapters:
        for subset in subsets:
            rows = load_predictions_for(args, adapter, subset)
            metric_rows.append(diagnostic_metric_row(rows, adapter, subset))
            structured_rows.append(structured_behavior_row(rows, adapter, subset))
            failure_rows.append(failure_type_eval_row(rows, adapter, subset))

    tables_dir = args.out_dir / "tables"
    reports_dir = args.out_dir / "reports"
    decision_dir = args.out_dir / "decision"
    decision = decision_json(metric_rows, structured_rows)

    write_csv(tables_dir / "exp19_sft_train_dev_diagnostic_metrics.csv", metric_rows, METRIC_FIELDS)
    write_csv(tables_dir / "exp19_sft_train_dev_structured_behavior.csv", structured_rows, STRUCTURED_FIELDS)
    write_csv(tables_dir / "exp19_sft_train_dev_failure_type_eval.csv", failure_rows, FAILURE_TYPE_FIELDS)
    write_text(
        reports_dir / "exp19_sft_train_dev_diagnostic_report.md",
        report_markdown(metric_rows, structured_rows, failure_rows, decision),
    )
    write_json(decision_dir / "exp19_sft_train_dev_diagnostic_decision.json", decision)
    return {
        "metric_rows": len(metric_rows),
        "structured_rows": len(structured_rows),
        "failure_rows": len(failure_rows),
        "decision": decision,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exp19-SFT-D2 train-vs-dev diagnostic.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Prepare prompt datasets and reference CSVs.")
    prepare.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    prepare.add_argument("--dev-jsonl", type=Path, default=DEFAULT_DEV_JSONL)
    prepare.add_argument("--a0-candidates", type=Path, default=DEFAULT_A0_CANDIDATES)
    prepare.add_argument("--a0-high-controls", type=Path, default=DEFAULT_A0_HIGH_CONTROLS)
    prepare.add_argument("--d1-dir", type=Path, default=DEFAULT_D1_DIR)
    prepare.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--max-train-clean-high", type=int, default=0, help="0 means all clean high controls.")
    prepare.add_argument("--max-dev-high", type=int, default=400, help="0 means all dev high controls.")

    collect_parser = subparsers.add_parser("collect", help="Collect LLaMA-Factory prediction outputs.")
    collect_parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    collect_parser.add_argument("--prediction-root", type=Path, default=DEFAULT_OUT_DIR / "predictions")
    collect_parser.add_argument("--adapters", default=",".join(ADAPTERS))
    collect_parser.add_argument("--subsets", default=",".join(SUBSETS))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "prepare":
        subsets = prepare_subsets(args)
        summary = {name: len(rows) for name, rows in subsets.items()}
    elif args.command == "collect":
        summary = collect(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
