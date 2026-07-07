"""Prepare Exp19-R7 train-only human-rationale recovered real DPO datasets.

R7 is a stricter reason-aware follow-up to R6:

- the source rows are only from ``question_seed42/train.jsonl``;
- dev/test are read only for ID/question-key leakage guards, never for labels;
- chosen responses use train-side human gold scores and recovered original
  human rationales from ``5-grades``;
- rejected responses are real wrong scalar scores from existing judges or the
  Qwen3-4B direct baseline, never handwritten templates.

The generated DPO data is meant for external review before training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence.diagnostics.build_train_hidden_failure_signals import (  # noqa: E402
    build_reason_indexes,
    clean_text as reason_clean_text,
    load_reason_rows,
    metric_aliases,
    norm_alnum,
    norm_ws,
    preferred_metric_rows,
)
from thesis_exp.exp17_low_score_evidence.prepare_exp19_r6_real_dpo import (  # noqa: E402
    build_pair_pool,
    coverage_rows,
    parse_target,
    read_r0a_predictions,
    risk_type_for,
    score_only_target,
    source_priority,
)
from thesis_exp.exp17_low_score_evidence.prepare_exp19_sft_dpo_datasets import (  # noqa: E402
    MODEL_PATH,
    clean,
    clamp_score,
    dpo_dataset_entry,
    messages_for,
    metric_name,
    read_csv_rows,
    read_jsonl,
    sample_id,
    sha1,
)
from thesis_exp.src.edujudge.utils.io import write_csv, write_json, write_text  # noqa: E402


DEFAULT_TRAIN_JSONL = Path("thesis_exp/data/splits/question_seed42/train.jsonl")
DEFAULT_DEV_JSONL = Path("thesis_exp/data/splits/question_seed42/dev.jsonl")
DEFAULT_TEST_JSONL = Path("thesis_exp/data/splits/question_seed42/test.jsonl")
DEFAULT_REASON_ROOT = Path("5-grades")
DEFAULT_A0_CANDIDATES = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp17_a0_train_hidden_failure_signals_seed42/"
    "train_hidden_failure_candidates.csv"
)
DEFAULT_R0A_PREDICTIONS = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp19_r0a_qwen4b_direct_baseline_seed42/"
    "predictions/full_predictions_internal.jsonl"
)
DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_r7_reason_recovered_real_dpo_seed42")

DATASET_FILES = {
    "r7a_score_only_reason_covered": "data/edubench_r7a_score_only_reason_covered_real_dpo_train.json",
    "r7b_human_reason_chosen": "data/edubench_r7b_human_reason_chosen_real_score_dpo_train.json",
    "r7c_low_high_human_reason_chosen": "data/edubench_r7c_low_high_human_reason_chosen_real_score_dpo_train.json",
    "r7d_strict_label_consistent_reason": "data/edubench_r7d_strict_label_consistent_reason_real_dpo_train.json",
}

DATASET_NAMES = {
    key: file_name.removeprefix("data/").removesuffix(".json") for key, file_name in DATASET_FILES.items()
}

REASON_CHOSEN_VARIANTS = {
    "r7b_human_reason_chosen",
    "r7c_low_high_human_reason_chosen",
    "r7d_strict_label_consistent_reason",
}

STRICT_RISK_TYPES = {
    "low_to_high_real_model_error",
    "low_to_mid_real_model_error",
    "high_to_low_real_model_error",
    "high_to_mid_real_model_error",
}


def write_json_array(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def fmt(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except Exception:
        return str(value)
    if math.isnan(val):
        return "nan"
    return f"{val:.{digits}f}"


def assistant_json_ordered(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def split_identity_rows(path: Path) -> tuple[set[str], set[str], int]:
    sample_ids: set[str] = set()
    question_keys: set[str] = set()
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            n += 1
            sid = clean(row.get("sample_id") or row.get("record_id") or row.get("id"))
            qkey = clean(row.get("question_key") or row.get("question_id"))
            if sid:
                sample_ids.add(sid)
            if qkey:
                question_keys.add(qkey)
    return sample_ids, question_keys, n


def reason_score(row: dict[str, Any]) -> int | None:
    try:
        return clamp_score(row.get("score"))
    except Exception:
        return None


def reason_summary(rows: list[dict[str, Any]], limit: int = 3) -> str:
    seen: list[str] = []
    for row in rows:
        reason = norm_ws(row.get("reason"))
        if reason and reason not in seen:
            seen.append(reason)
    return " / ".join(seen[:limit])


def recover_reason_bundle(
    row: dict[str, Any],
    exact_index: dict[tuple[str, str], list[dict[str, Any]]],
    alnum_index: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    question = norm_ws(row.get("question", ""))
    answer = norm_ws(row.get("answer", ""))
    candidates = exact_index.get((question, answer), [])
    if not candidates:
        candidates = alnum_index.get((question, norm_alnum(answer)), [])

    metric_rows = preferred_metric_rows(candidates, metric_name(row))
    answer_model = clean(row.get("answer_model") or row.get("generator_model"))
    model_rows = [item for item in metric_rows if clean(item.get("model")) == answer_model]
    selected_metric_rows = model_rows or metric_rows

    gold = clamp_score(row.get("label_5"))
    label_consistent_rows = [item for item in selected_metric_rows if reason_score(item) == gold]
    selected_reason_rows = label_consistent_rows or selected_metric_rows
    scores = [score for score in (reason_score(item) for item in selected_metric_rows) if score is not None]

    if label_consistent_rows:
        status = "label_consistent_metric_model_reason_recovered" if model_rows else "label_consistent_metric_reason_recovered"
    elif selected_metric_rows:
        status = "metric_reason_recovered_score_mismatch_or_no_label_match"
    elif candidates:
        status = "question_answer_matched_metric_unmatched"
    else:
        status = "question_answer_unmatched"

    return {
        "reason_match_status": status,
        "qa_candidate_count": len(candidates),
        "metric_candidate_count": len(metric_rows),
        "model_metric_candidate_count": len(model_rows),
        "selected_reason_count": len(selected_reason_rows),
        "label_consistent_reason_count": len(label_consistent_rows),
        "reason_summary": reason_summary(selected_reason_rows),
        "reason_scores": sorted(set(scores)),
        "reason_eval_ids": sorted(
            set(clean(item.get("eval")) or clean(item.get("_source_file")) for item in selected_reason_rows)
        ),
        "reason_source_files": sorted(set(clean(item.get("_source_file")) for item in selected_reason_rows)),
        "reason_principles": sorted(set(reason_clean_text(item.get("principle")) for item in selected_reason_rows)),
    }


def build_reason_bundles(train_rows: list[dict[str, Any]], reason_root: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    reason_rows, missing_files = load_reason_rows(reason_root)
    exact_index, alnum_index = build_reason_indexes(reason_rows)
    bundles: dict[str, dict[str, Any]] = {}
    for row in train_rows:
        bundles[sample_id(row)] = recover_reason_bundle(row, exact_index, alnum_index)
    return bundles, missing_files


def human_reason_target(record: dict[str, Any]) -> dict[str, Any]:
    reason = clean(record["reason_bundle"].get("reason_summary"))
    return {
        "reason": reason,
        "score": clamp_score(record["gold_label"]),
    }


def dpo_pair(record: dict[str, Any], variant: str, copy_index: int = 0) -> dict[str, Any]:
    chosen_schema = "score_only" if variant == "r7a_score_only_reason_covered" else "human_reason_score"
    chosen_payload = score_only_target(record["gold_label"]) if chosen_schema == "score_only" else human_reason_target(record)
    rejected_payload = score_only_target(record["rejected_score"])
    rb = record["reason_bundle"]
    row = record["row"]
    return {
        "messages": messages_for(row),
        "chosen": {"role": "assistant", "content": assistant_json_ordered(chosen_payload)},
        "rejected": {"role": "assistant", "content": assistant_json_ordered(rejected_payload)},
        "split_source": "train",
        "dataset_variant": variant,
        "chosen_schema": chosen_schema,
        "risk_type": record["risk_type"],
        "rejected_origin": "real_model_score_output",
        "rejected_source": "|".join(record["sources"]),
        "supporting_rejected_sources": "|".join(record["sources"]),
        "rejected_source_count": len(record["sources"]),
        "source_sample_id": record["sample_id"],
        "source_question_key": clean(row.get("question_key") or row.get("question_id")),
        "gold_label": record["gold_label"],
        "rejected_score": record["rejected_score"],
        "reason_match_status": rb.get("reason_match_status"),
        "reason_label_consistent": bool(int(rb.get("label_consistent_reason_count") or 0) > 0),
        "reason_source_files": "|".join(rb.get("reason_source_files") or []),
        "reason_eval_ids": "|".join(rb.get("reason_eval_ids") or []),
        "reason_scores": "|".join(str(item) for item in (rb.get("reason_scores") or [])),
        "reason_hash": sha1(clean(rb.get("reason_summary"))),
        "oversample_copy_index": copy_index,
    }


def select_records(pool: list[dict[str, Any]], variant: str) -> list[dict[str, Any]]:
    reason_recovered = [row for row in pool if clean(row["reason_bundle"].get("reason_summary"))]
    if variant == "r7a_score_only_reason_covered":
        return reason_recovered
    if variant == "r7b_human_reason_chosen":
        return reason_recovered
    if variant == "r7c_low_high_human_reason_chosen":
        return [row for row in reason_recovered if row["risk_type"] in STRICT_RISK_TYPES]
    if variant == "r7d_strict_label_consistent_reason":
        return [
            row
            for row in reason_recovered
            if row["risk_type"] in STRICT_RISK_TYPES
            and int(row["reason_bundle"].get("label_consistent_reason_count") or 0) > 0
        ]
    raise ValueError(f"Unknown variant: {variant}")


def build_datasets(pool: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    datasets: dict[str, list[dict[str, Any]]] = {}
    for variant in DATASET_FILES:
        datasets[variant] = [dpo_pair(record, variant, idx) for idx, record in enumerate(select_records(pool, variant))]
    return datasets


def dataset_info() -> dict[str, Any]:
    return {DATASET_NAMES[key]: dpo_dataset_entry(file_name) for key, file_name in DATASET_FILES.items()}


def pool_with_reasons(
    train_rows: list[dict[str, Any]],
    a0_by_id: dict[str, dict[str, str]],
    r0a_by_id: dict[str, dict[str, Any]],
    reason_bundles: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    pool = build_pair_pool(train_rows, a0_by_id, r0a_by_id)
    out: list[dict[str, Any]] = []
    for record in pool:
        sid = record["sample_id"]
        record["reason_bundle"] = reason_bundles.get(sid, {})
        out.append(record)
    return out


def leakage_audit(
    datasets: dict[str, list[dict[str, Any]]],
    train_ids: set[str],
    train_qkeys: set[str],
    dev_ids: set[str],
    dev_qkeys: set[str],
    test_ids: set[str],
    test_qkeys: set[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for variant, pairs in datasets.items():
        sample_ids = {clean(pair.get("source_sample_id")) for pair in pairs}
        question_keys = {clean(pair.get("source_question_key")) for pair in pairs}
        not_train_ids = sorted(sample_ids - train_ids)
        not_train_qkeys = sorted(question_keys - train_qkeys)
        dev_id_hits = sorted(sample_ids & dev_ids)
        dev_q_hits = sorted(question_keys & dev_qkeys)
        test_id_hits = sorted(sample_ids & test_ids)
        test_q_hits = sorted(question_keys & test_qkeys)
        prompt_reason_hits = 0
        for pair in pairs:
            reason = clean(parse_target(pair["chosen"]).get("reason"))
            prompt = json.dumps(pair.get("messages") or [], ensure_ascii=False)
            if reason and reason in prompt:
                prompt_reason_hits += 1
        row = {
            "dataset_variant": variant,
            "dataset_name": DATASET_NAMES[variant],
            "pairs": len(pairs),
            "not_train_sample_id_count": len(not_train_ids),
            "not_train_question_key_count": len(not_train_qkeys),
            "dev_sample_id_overlap": len(dev_id_hits),
            "dev_question_key_overlap": len(dev_q_hits),
            "test_sample_id_overlap": len(test_id_hits),
            "test_question_key_overlap": len(test_q_hits),
            "human_reason_in_prompt_count": prompt_reason_hits,
            "leakage_pass": not any(
                [
                    not_train_ids,
                    not_train_qkeys,
                    dev_id_hits,
                    dev_q_hits,
                    test_id_hits,
                    test_q_hits,
                    prompt_reason_hits,
                ]
            ),
        }
        rows.append(row)
        if not row["leakage_pass"]:
            errors.append(f"{variant}: leakage audit failed: {row}")
    return rows, errors


def recovery_coverage_rows(train_rows: list[dict[str, Any]], bundles: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        grouped[str(clamp_score(row.get("label_5")))].append(row)
    rows: list[dict[str, Any]] = []
    for label, items in sorted(grouped.items(), key=lambda kv: int(kv[0])):
        recovered = 0
        label_consistent = 0
        statuses: Counter[str] = Counter()
        for item in items:
            bundle = bundles.get(sample_id(item), {})
            statuses.update([clean(bundle.get("reason_match_status"))])
            if clean(bundle.get("reason_summary")):
                recovered += 1
            if int(bundle.get("label_consistent_reason_count") or 0) > 0:
                label_consistent += 1
        rows.append(
            {
                "gold_label": label,
                "train_samples": len(items),
                "reason_recovered_samples": recovered,
                "reason_recovered_rate": fmt(recovered / len(items) if items else 0.0),
                "label_consistent_reason_samples": label_consistent,
                "label_consistent_reason_rate": fmt(label_consistent / len(items) if items else 0.0),
                "match_status_counts": json.dumps(statuses, ensure_ascii=False, sort_keys=True),
            }
        )
    return rows


def pair_count_rows(datasets: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant, pairs in datasets.items():
        by_risk = Counter(pair["risk_type"] for pair in pairs)
        by_consistency = Counter(str(pair["reason_label_consistent"]) for pair in pairs)
        for risk_type, count in sorted(by_risk.items()):
            subset = [pair for pair in pairs if pair["risk_type"] == risk_type]
            reason_consistent = sum(1 for pair in subset if pair["reason_label_consistent"])
            rows.append(
                {
                    "dataset_variant": variant,
                    "dataset_name": DATASET_NAMES[variant],
                    "chosen_schema": subset[0]["chosen_schema"] if subset else "",
                    "risk_type": risk_type,
                    "pairs": count,
                    "unique_source_samples": len({pair["source_sample_id"] for pair in subset}),
                    "label_consistent_reason_pairs": reason_consistent,
                    "label_consistent_reason_rate": fmt(reason_consistent / count if count else 0.0),
                    "all_dataset_reason_consistency_counts": json.dumps(by_consistency, sort_keys=True),
                }
            )
    return rows


def rejected_source_rows(datasets: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant, pairs in datasets.items():
        counter: Counter[tuple[str, str]] = Counter()
        for pair in pairs:
            for source in clean(pair.get("supporting_rejected_sources")).split("|"):
                if source:
                    counter[(pair["risk_type"], source)] += 1
        rows.extend(
            {
                "dataset_variant": variant,
                "dataset_name": DATASET_NAMES[variant],
                "risk_type": risk_type,
                "rejected_source": source,
                "n": count,
            }
            for (risk_type, source), count in sorted(counter.items(), key=lambda kv: (kv[0][0], source_priority(kv[0][1])))
        )
    return rows


def manifest_rows(datasets: dict[str, list[dict[str, Any]]], max_rows_per_variant: int = 5000) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant, pairs in datasets.items():
        for pair in pairs[:max_rows_per_variant]:
            chosen = parse_target(pair["chosen"])
            rejected = parse_target(pair["rejected"])
            rows.append(
                {
                    "dataset_variant": variant,
                    "dataset_name": DATASET_NAMES[variant],
                    "sample_id": pair.get("source_sample_id"),
                    "question_key": pair.get("source_question_key"),
                    "risk_type": pair.get("risk_type"),
                    "gold_label": pair.get("gold_label"),
                    "rejected_score": pair.get("rejected_score"),
                    "chosen_score": chosen.get("score"),
                    "chosen_schema": pair.get("chosen_schema"),
                    "chosen_reason_present": bool(clean(chosen.get("reason"))),
                    "chosen_reason_preview": clean(chosen.get("reason"))[:180],
                    "chosen_reason_hash": sha1(clean(chosen.get("reason"))),
                    "reason_match_status": pair.get("reason_match_status"),
                    "reason_label_consistent": pair.get("reason_label_consistent"),
                    "reason_scores": pair.get("reason_scores"),
                    "reason_source_files": pair.get("reason_source_files"),
                    "reason_eval_ids": pair.get("reason_eval_ids"),
                    "rejected_payload": assistant_json_ordered(rejected),
                    "rejected_source": pair.get("rejected_source"),
                    "rejected_source_count": pair.get("rejected_source_count"),
                    "rejected_origin": pair.get("rejected_origin"),
                    "message_hash": sha1(json.dumps(pair.get("messages"), ensure_ascii=False, sort_keys=True)),
                    "chosen_hash": sha1(clean(pair["chosen"].get("content"))),
                    "rejected_hash": sha1(clean(pair["rejected"].get("content"))),
                }
            )
    return rows


def write_configs(out_dir: Path) -> None:
    config_dir = out_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    base = {
        "model_name_or_path": MODEL_PATH,
        "trust_remote_code": True,
        "stage": "dpo",
        "do_train": True,
        "finetuning_type": "lora",
        "adapter_name_or_path": "saves/edubench/qwen3-4b/r2_clean_reason_score_balanced_lora",
        "ref_model": MODEL_PATH,
        "ref_model_adapters": "saves/edubench/qwen3-4b/r2_clean_reason_score_balanced_lora",
        "dataset_dir": str(out_dir),
        "template": "qwen3_nothink",
        "cutoff_len": 4096,
        "overwrite_cache": True,
        "preprocessing_num_workers": 16,
        "pref_loss": "sigmoid",
        "pref_beta": 0.03,
        "pref_ftx": 0.05,
        "logging_steps": 10,
        "save_steps": 100,
        "save_only_model": True,
        "plot_loss": True,
        "report_to": "none",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": "5.0e-6",
        "max_steps": 100,
        "bf16": True,
        "gradient_checkpointing": True,
    }
    run_rows = []
    for variant in ["r7b_human_reason_chosen", "r7c_low_high_human_reason_chosen", "r7d_strict_label_consistent_reason"]:
        run_name = f"{variant}_from_r2c_s100_b0p03_lr5em6"
        output_dir = f"saves/edubench/qwen3-4b/{run_name}"
        config = {**base, "dataset": DATASET_NAMES[variant], "output_dir": output_dir}
        text = "\n".join(f"{key}: {str(value).lower() if isinstance(value, bool) else value}" for key, value in config.items())
        write_text(config_dir / f"llamafactory_qwen3_4b_{run_name}.yaml", text)
        run_rows.append(
            {
                "run_name": run_name,
                "dataset_variant": variant,
                "dataset_name": DATASET_NAMES[variant],
                "max_steps": 100,
                "pref_beta": "0.03",
                "learning_rate": "5.0e-6",
                "init_adapter": "saves/edubench/qwen3-4b/r2_clean_reason_score_balanced_lora",
                "output_dir": output_dir,
            }
        )
    write_csv(out_dir / "tables" / "r7_run_matrix.csv", run_rows)


def write_report(
    out_dir: Path,
    train_rows: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    datasets: dict[str, list[dict[str, Any]]],
    missing_reason_files: list[str],
    leakage_rows: list[dict[str, Any]],
    dev_n: int,
    test_n: int,
) -> None:
    reason_recovered_pool = [row for row in pool if clean(row["reason_bundle"].get("reason_summary"))]
    strict_reason_pool = [
        row for row in reason_recovered_pool if int(row["reason_bundle"].get("label_consistent_reason_count") or 0) > 0
    ]
    lines = [
        "# Exp19-R7 Reason-Recovered Real DPO Dataset Report",
        "",
        "Exp19-R7 rebuilds the DPO data around original human rationales. It is train-only data for review before DPO training.",
        "",
        "## Construction",
        "",
        "- input prompt: question + answer + metric + rubric + metadata.",
        "- chosen response: train gold score, optionally with recovered original human reason.",
        "- rejected response: real wrong scalar score from Qwen3-4B R0A or existing judge scores.",
        "- rejected responses are not synthetic templates.",
        "- human reasons are never inserted into the user prompt.",
        "",
        "## Leakage Guard",
        "",
        f"- train rows used for construction: `{len(train_rows)}`",
        f"- dev rows read only for ID/question-key leakage guard: `{dev_n}`",
        f"- test rows read only for ID/question-key leakage guard: `{test_n}`",
        "- dev/test labels are not used.",
        "- any sample/question overlap with dev/test makes the script fail.",
        "",
        "| dataset | pairs | leakage_pass | human_reason_in_prompt_count |",
        "|---|---:|---|---:|",
    ]
    for row in leakage_rows:
        lines.append(
            f"| `{row['dataset_name']}` | {row['pairs']} | `{row['leakage_pass']}` | {row['human_reason_in_prompt_count']} |"
        )
    lines.extend(
        [
            "",
            "## Reason Recovery",
            "",
            f"- pair pool before reason filtering: `{len(pool)}`",
            f"- pair pool with recovered reason: `{len(reason_recovered_pool)}`",
            f"- pair pool with label-consistent recovered reason: `{len(strict_reason_pool)}`",
            f"- missing reason files: `{'; '.join(missing_reason_files) if missing_reason_files else 'none'}`",
            "",
            "## Dataset Variants",
            "",
        ]
    )
    for variant, pairs in datasets.items():
        role = "score-only control" if variant == "r7a_score_only_reason_covered" else "reason-aware DPO candidate"
        lines.append(f"- `{DATASET_NAMES[variant]}`: {len(pairs)} pairs; {role}.")
    lines.extend(
        [
            "",
            "## Recommended Review Order",
            "",
            "1. Review `edubench_r7d_strict_label_consistent_reason_real_dpo_train` first. It is the cleanest reason-aware candidate.",
            "2. Review `edubench_r7c_low_high_human_reason_chosen_real_score_dpo_train` if the goal is low/high risk control.",
            "3. Keep `edubench_r7a_score_only_reason_covered_real_dpo_train` as a score-only control on the same reason-covered sample pool.",
            "",
            "## Caveats",
            "",
            "- Rejected responses are real scalar score outputs, not full rejected reasons.",
            "- Chosen reasons are recovered from original human rationale files, but some are not label-consistent with the rounded final label.",
            "- The strict R7D variant filters to label-consistent recovered rationales to reduce this risk.",
        ]
    )
    write_text(out_dir / "reports" / "exp19_r7_reason_recovered_real_dpo_report.md", "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Exp19-R7 reason-recovered real DPO data.")
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--dev-jsonl", type=Path, default=DEFAULT_DEV_JSONL)
    parser.add_argument("--test-jsonl", type=Path, default=DEFAULT_TEST_JSONL)
    parser.add_argument("--reason-root", type=Path, default=DEFAULT_REASON_ROOT)
    parser.add_argument("--a0-candidates", type=Path, default=DEFAULT_A0_CANDIDATES)
    parser.add_argument("--r0a-predictions", type=Path, default=DEFAULT_R0A_PREDICTIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    train_rows = read_jsonl(args.train_jsonl)
    train_ids, train_qkeys, train_n = split_identity_rows(args.train_jsonl)
    dev_ids, dev_qkeys, dev_n = split_identity_rows(args.dev_jsonl)
    test_ids, test_qkeys, test_n = split_identity_rows(args.test_jsonl)
    if train_n != len(train_rows):
        raise RuntimeError("Train identity row count does not match train rows.")
    if train_qkeys & dev_qkeys or train_qkeys & test_qkeys:
        raise RuntimeError("question_seed42 split is not question-disjoint; aborting DPO construction.")

    a0_by_id = {clean(row.get("sample_id")): row for row in read_csv_rows(args.a0_candidates)}
    r0a_by_id = read_r0a_predictions(args.r0a_predictions)
    reason_bundles, missing_reason_files = build_reason_bundles(train_rows, args.reason_root)
    pool = pool_with_reasons(train_rows, a0_by_id, r0a_by_id, reason_bundles)
    datasets = build_datasets(pool)

    leakage_rows, leakage_errors = leakage_audit(
        datasets,
        train_ids,
        train_qkeys,
        dev_ids,
        dev_qkeys,
        test_ids,
        test_qkeys,
    )
    if leakage_errors:
        raise RuntimeError("\n".join(leakage_errors))

    for variant, pairs in datasets.items():
        write_json_array(args.out_dir / DATASET_FILES[variant], pairs)
    write_json(args.out_dir / "dataset_info.json", dataset_info())
    write_json(args.out_dir / "dataset_info_r7_snippet.json", dataset_info())
    write_configs(args.out_dir)

    write_csv(args.out_dir / "tables" / "r7_reason_recovery_coverage.csv", recovery_coverage_rows(train_rows, reason_bundles))
    write_csv(args.out_dir / "tables" / "r7_train_label_coverage.csv", coverage_rows(pool, train_rows))
    write_csv(args.out_dir / "tables" / "r7_pair_counts.csv", pair_count_rows(datasets))
    write_csv(args.out_dir / "tables" / "r7_rejected_source_counts.csv", rejected_source_rows(datasets))
    write_csv(args.out_dir / "tables" / "r7_leakage_audit.csv", leakage_rows)
    write_csv(args.out_dir / "review" / "r7_reason_recovered_dpo_review_manifest.csv", manifest_rows(datasets))
    write_report(args.out_dir, train_rows, pool, datasets, missing_reason_files, leakage_rows, dev_n, test_n)

    decision = {
        "safe_for_dpo_data_review": True,
        "recommended_first_review_dataset": DATASET_NAMES["r7d_strict_label_consistent_reason"],
        "recommended_reason_risk_dataset": DATASET_NAMES["r7c_low_high_human_reason_chosen"],
        "score_only_control_dataset": DATASET_NAMES["r7a_score_only_reason_covered"],
        "train_rows": len(train_rows),
        "dev_rows_used_for_id_guard_only": dev_n,
        "test_rows_used_for_id_guard_only": test_n,
        "pair_counts": {DATASET_NAMES[key]: len(value) for key, value in datasets.items()},
        "guardrails": {
            "training_source_split": "question_seed42/train",
            "dev_test_labels_used": False,
            "dev_test_ids_used_for_leakage_guard_only": True,
            "human_rationale_not_used_as_prompt_input": True,
            "rejected_side_is_real_model_score": True,
            "synthetic_or_template_rejected_used": False,
            "leakage_audit_pass": all(bool(row["leakage_pass"]) for row in leakage_rows),
        },
        "caveats": [
            "Rejected side is still a scalar real model score, not a full natural-language rejected rationale.",
            "R7D is strictest because it requires recovered human reason scores to include the final gold label.",
            "R7B/R7C preserve more coverage but may include reasons whose individual rater score differs from the rounded final label.",
        ],
    }
    write_json(args.out_dir / "decision" / "exp19_r7_reason_recovered_real_dpo_decision.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
