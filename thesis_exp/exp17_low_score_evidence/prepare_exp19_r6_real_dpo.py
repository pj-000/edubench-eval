"""Prepare Exp19-R6 real rejected-output DPO datasets.

R6 is intentionally stricter than earlier R5 synthetic/template DPO data:

- train split only;
- no test read;
- no dev/D1 labels for training;
- no hand-written rejected responses;
- every rejected response is a real score emitted by an existing judge or by
  the Qwen3-4B direct baseline.

The main datasets are score-only DPO pairs, because most reusable real rejected
outputs in this repository are scalar score judgments. Structured chosen-target
variants are emitted separately for review, but the report marks them as
auxiliary because their rejected side is still score-only.
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

from thesis_exp.exp17_low_score_evidence.prepare_exp19_sft_dpo_datasets import (  # noqa: E402
    MODEL_PATH,
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
    redacted_input,
    redacted_target,
    sample_id,
    sha1,
    subject,
)
from thesis_exp.src.edujudge.utils.io import write_csv, write_json, write_text  # noqa: E402


DEFAULT_TRAIN_JSONL = Path("thesis_exp/data/splits/question_seed42/train.jsonl")
DEFAULT_A0_CANDIDATES = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp17_a0_train_hidden_failure_signals_seed42/"
    "train_hidden_failure_candidates.csv"
)
DEFAULT_R0A_PREDICTIONS = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp19_r0a_qwen4b_direct_baseline_seed42/"
    "predictions/full_predictions_internal.jsonl"
)
DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_r6_real_dpo_seed42")

DATASET_FILES = {
    "r6a_qwen_score_only": "data/edubench_r6a_qwen_only_score_real_dpo_train.json",
    "r6b_multi_score_only": "data/edubench_r6b_multi_judge_score_real_dpo_train.json",
    "r6c_low_high_score_only": "data/edubench_r6c_low_high_score_real_dpo_train.json",
    "r6d_multi_structured_chosen": "data/edubench_r6d_multi_judge_structured_chosen_real_score_dpo_train.json",
    "r6e_multi_reason_chosen": "data/edubench_r6e_multi_judge_reason_chosen_real_score_dpo_train.json",
}

DATASET_NAMES = {
    key: file_name.removeprefix("data/").removesuffix(".json") for key, file_name in DATASET_FILES.items()
}

SCORE_ONLY_VARIANTS = {"r6a_qwen_score_only", "r6b_multi_score_only", "r6c_low_high_score_only"}
MAIN_TRAINING_VARIANTS = {"r6a_qwen_score_only", "r6b_multi_score_only", "r6c_low_high_score_only"}
LOW_HIGH_RISK_TYPES = {
    "low_to_high_real_model_error",
    "low_to_mid_real_model_error",
    "high_to_low_real_model_error",
    "high_to_mid_real_model_error",
}
STRICT_LOW_HIGH_RISK_TYPES = {
    "low_to_high_real_model_error",
    "high_to_low_real_model_error",
    "high_to_mid_real_model_error",
}
JUDGE_SCORE_SOURCES = {"EduBenchEvaluator", "deepseek-r1", "deepseek-v3", "gpt-4o", "qwq-plus"}


def write_json_array(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def safe_int(value: Any) -> int | None:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


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


def score_only_target(score: int) -> dict[str, int]:
    return {"score": clamp_score(score)}


def risk_type_for(gold: int, pred: int) -> str:
    if gold <= 2 and pred >= 4:
        return "low_to_high_real_model_error"
    if gold <= 2 and pred == 3:
        return "low_to_mid_real_model_error"
    if gold >= 4 and pred <= 2:
        return "high_to_low_real_model_error"
    if gold >= 4 and pred == 3:
        return "high_to_mid_real_model_error"
    if gold == 3 and pred >= 4:
        return "mid_to_high_real_model_error"
    if gold == 3 and pred <= 2:
        return "mid_to_low_real_model_error"
    return ""


def read_r0a_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Qwen3-4B R0A prediction file: {path}")
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if clean(row.get("split")) != "train":
                continue
            sid = clean(row.get("sample_id"))
            pred = safe_int(row.get("pred_label"))
            if sid and pred is not None:
                out[sid] = row
    return out


def source_predictions(row: dict[str, Any], r0a_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    sid = sample_id(row)
    sources: list[dict[str, Any]] = []
    r0a = r0a_by_id.get(sid)
    if r0a is not None:
        pred = safe_int(r0a.get("pred_label"))
        if pred is not None:
            sources.append(
                {
                    "source": "qwen3_4b_r0a_direct",
                    "source_family": "qwen3_4b_direct_baseline",
                    "pred_score": pred,
                    "raw_output_hash": sha1(clean(r0a.get("raw_output_truncated"))),
                    "parse_method": clean(r0a.get("parse_method")),
                }
            )
    for judge_name, value in (row.get("judge_scores") or {}).items():
        if judge_name not in JUDGE_SCORE_SOURCES:
            continue
        pred = safe_int(value)
        if pred is None:
            continue
        sources.append(
            {
                "source": judge_name,
                "source_family": "existing_judge_score",
                "pred_score": pred,
                "raw_output_hash": "",
                "parse_method": "provided_score",
            }
        )
    return sources


def source_priority(source: str) -> tuple[int, str]:
    order = {
        "qwen3_4b_r0a_direct": 0,
        "EduBenchEvaluator": 1,
        "deepseek-v3": 2,
        "deepseek-r1": 3,
        "gpt-4o": 4,
        "qwq-plus": 5,
    }
    return order.get(source, 99), source


def build_pair_pool(
    train_rows: list[dict[str, Any]],
    a0_by_id: dict[str, dict[str, str]],
    r0a_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    keyed: dict[tuple[str, str, int], dict[str, Any]] = {}
    row_by_id = {sample_id(row): row for row in train_rows}
    for row in train_rows:
        sid = sample_id(row)
        gold = clamp_score(row.get("label_5"))
        for src in source_predictions(row, r0a_by_id):
            pred = clamp_score(src["pred_score"])
            if pred == gold:
                continue
            risk_type = risk_type_for(gold, pred)
            if not risk_type:
                continue
            key = (sid, risk_type, pred)
            record = keyed.setdefault(
                key,
                {
                    "sample_id": sid,
                    "row": row_by_id[sid],
                    "gold_label": gold,
                    "rejected_score": pred,
                    "risk_type": risk_type,
                    "sources": [],
                    "source_families": set(),
                    "raw_output_hashes": [],
                    "parse_methods": set(),
                    "a0_signal": a0_by_id.get(sid),
                },
            )
            record["sources"].append(src["source"])
            record["source_families"].add(src["source_family"])
            if src.get("raw_output_hash"):
                record["raw_output_hashes"].append(src["raw_output_hash"])
            if src.get("parse_method"):
                record["parse_methods"].add(src["parse_method"])

    pool = list(keyed.values())
    for record in pool:
        record["sources"] = sorted(set(record["sources"]), key=source_priority)
        record["source_families"] = sorted(record["source_families"])
        record["parse_methods"] = sorted(record["parse_methods"])
        record["raw_output_hashes"] = sorted(set(record["raw_output_hashes"]))
    pool.sort(key=lambda item: (item["risk_type"], item["sample_id"], item["rejected_score"]))
    return pool


def chosen_target(row: dict[str, Any], a0_signal: dict[str, str] | None, schema: str) -> dict[str, Any]:
    gold = clamp_score(row.get("label_5"))
    if schema == "score_only":
        return score_only_target(gold)
    if schema == "structured":
        return evidence_target(row, a0_signal, include_reason=False)
    if schema == "reason":
        return evidence_target(row, a0_signal, include_reason=True)
    raise ValueError(f"Unknown chosen schema: {schema}")


def primary_rejected_source(record: dict[str, Any], dataset_variant: str) -> str:
    if dataset_variant == "r6a_qwen_score_only":
        return "qwen3_4b_r0a_direct"
    return "|".join(record["sources"])


def dpo_pair(record: dict[str, Any], dataset_variant: str, chosen_schema: str, copy_index: int = 0) -> dict[str, Any]:
    row = record["row"]
    chosen = chosen_target(row, record.get("a0_signal"), chosen_schema)
    rejected = score_only_target(record["rejected_score"])
    rejected_source = primary_rejected_source(record, dataset_variant)
    return {
        "messages": messages_for(row),
        "chosen": {"role": "assistant", "content": assistant_json(chosen)},
        "rejected": {"role": "assistant", "content": assistant_json(rejected)},
        "risk_type": record["risk_type"],
        "dataset_variant": dataset_variant,
        "chosen_schema": chosen_schema,
        "rejected_origin": "real_model_score_output",
        "rejected_source": rejected_source,
        "supporting_rejected_sources": "|".join(record["sources"]),
        "rejected_source_count": len(record["sources"]),
        "source_sample_id": record["sample_id"],
        "gold_label": record["gold_label"],
        "rejected_score": record["rejected_score"],
        "oversample_copy_index": copy_index,
    }


def select_records(pool: list[dict[str, Any]], variant: str) -> list[dict[str, Any]]:
    if variant == "r6a_qwen_score_only":
        return [row for row in pool if "qwen3_4b_r0a_direct" in row["sources"]]
    if variant in {"r6b_multi_score_only", "r6d_multi_structured_chosen", "r6e_multi_reason_chosen"}:
        return list(pool)
    if variant == "r6c_low_high_score_only":
        return [row for row in pool if row["risk_type"] in STRICT_LOW_HIGH_RISK_TYPES]
    raise ValueError(f"Unknown dataset variant: {variant}")


def schema_for_variant(variant: str) -> str:
    if variant.endswith("_structured_chosen"):
        return "structured"
    if variant.endswith("_reason_chosen"):
        return "reason"
    return "score_only"


def build_datasets(pool: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    datasets: dict[str, list[dict[str, Any]]] = {}
    for variant in DATASET_FILES:
        schema = schema_for_variant(variant)
        records = select_records(pool, variant)
        pairs = [dpo_pair(record, variant, schema, idx) for idx, record in enumerate(records)]
        rng.shuffle(pairs)
        datasets[variant] = pairs
    return datasets


def dataset_info() -> dict[str, Any]:
    return {DATASET_NAMES[key]: dpo_dataset_entry(file_name) for key, file_name in DATASET_FILES.items()}


def parse_target(message: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = json.loads(clean(message.get("content")))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def dataset_stats(variant: str, pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        grouped[pair["risk_type"]].append(pair)
    rows: list[dict[str, Any]] = []
    for risk_type, risk_pairs in sorted(grouped.items()):
        chosen_scores = [safe_int(parse_target(pair["chosen"]).get("score")) or 0 for pair in risk_pairs]
        rejected_scores = [safe_int(parse_target(pair["rejected"]).get("score")) or 0 for pair in risk_pairs]
        source_counts = [int(pair.get("rejected_source_count") or 0) for pair in risk_pairs]
        reason_present = [
            bool(clean(parse_target(pair["chosen"]).get("brief_reason"))) for pair in risk_pairs
        ]
        rows.append(
            {
                "dataset_variant": variant,
                "dataset_name": DATASET_NAMES[variant],
                "risk_type": risk_type,
                "pairs": len(risk_pairs),
                "unique_source_samples": len({pair["source_sample_id"] for pair in risk_pairs}),
                "chosen_score_mean": fmt(mean([float(x) for x in chosen_scores])),
                "rejected_score_mean": fmt(mean([float(x) for x in rejected_scores])),
                "source_count_mean": fmt(mean([float(x) for x in source_counts])),
                "reason_present_rate": fmt(sum(reason_present) / len(reason_present) if reason_present else 0.0),
                "hard_synthetic_rate": "0.0000",
                "template_rejected_rate": "0.0000",
                "actual_rejected_rate": "1.0000",
            }
        )
    return rows


def source_stats(variant: str, pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter: Counter[tuple[str, str]] = Counter()
    for pair in pairs:
        for source in clean(pair.get("rejected_source")).split("|"):
            if source:
                counter[(pair["risk_type"], source)] += 1
    return [
        {
            "dataset_variant": variant,
            "risk_type": risk_type,
            "rejected_source": source,
            "n": n,
        }
        for (risk_type, source), n in sorted(counter.items())
    ]


def manifest_rows(variant: str, pairs: list[dict[str, Any]], max_rows: int = 5000) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in pairs[:max_rows]:
        chosen = parse_target(pair["chosen"])
        rejected = parse_target(pair["rejected"])
        rows.append(
            {
                "dataset_variant": variant,
                "sample_id": pair.get("source_sample_id"),
                "risk_type": pair.get("risk_type"),
                "gold_label": pair.get("gold_label"),
                "rejected_score": pair.get("rejected_score"),
                "chosen_score": chosen.get("score"),
                "chosen_schema": pair.get("chosen_schema"),
                "chosen_major_failures": "|".join(chosen.get("major_failures") or []),
                "chosen_score_cap": chosen.get("score_cap"),
                "chosen_brief_reason_present": bool(clean(chosen.get("brief_reason"))),
                "rejected_payload": assistant_json(rejected),
                "rejected_source": pair.get("rejected_source"),
                "supporting_rejected_sources": pair.get("supporting_rejected_sources"),
                "rejected_source_count": pair.get("rejected_source_count"),
                "rejected_origin": pair.get("rejected_origin"),
                "message_hash": sha1(json.dumps(pair.get("messages"), ensure_ascii=False, sort_keys=True)),
                "chosen_hash": sha1(clean(pair["chosen"].get("content"))),
                "rejected_hash": sha1(clean(pair["rejected"].get("content"))),
            }
        )
    return rows


def attach_rows_for_review(pairs: list[dict[str, Any]], row_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for pair in pairs:
        copied = dict(pair)
        copied["_row"] = row_by_id.get(clean(pair.get("source_sample_id")))
        out.append(copied)
    return out


def redacted_samples(variant: str, pairs: list[dict[str, Any]], row_by_id: dict[str, dict[str, Any]], limit_per_type: int) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    out: list[dict[str, Any]] = []
    for pair in pairs:
        risk_type = clean(pair.get("risk_type"))
        if counts[risk_type] >= limit_per_type:
            continue
        row = row_by_id.get(clean(pair.get("source_sample_id")))
        if row is None:
            continue
        out.append(
            {
                "input_redacted": redacted_input(row),
                "chosen_redacted": redacted_target(parse_target(pair["chosen"])),
                "rejected_redacted": redacted_target(parse_target(pair["rejected"])),
                "risk_type": risk_type,
                "dataset_variant": variant,
                "rejected_origin": pair.get("rejected_origin"),
                "rejected_source": pair.get("rejected_source"),
                "rejected_source_count": pair.get("rejected_source_count"),
            }
        )
        counts[risk_type] += 1
    return out


def coverage_rows(pool: list[dict[str, Any]], train_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_label = Counter(str(clamp_score(row.get("label_5"))) for row in train_rows)
    pair_by_label = Counter(str(record["gold_label"]) for record in pool)
    sample_by_label: dict[str, set[str]] = defaultdict(set)
    for record in pool:
        sample_by_label[str(record["gold_label"])].add(record["sample_id"])
    rows = []
    for label in sorted(by_label, key=int):
        rows.append(
            {
                "gold_label": label,
                "train_samples": by_label[label],
                "samples_with_real_rejected_pair": len(sample_by_label.get(label, set())),
                "real_rejected_pairs": pair_by_label[label],
                "sample_coverage_rate": fmt(len(sample_by_label.get(label, set())) / by_label[label]),
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
    run_rows: list[dict[str, Any]] = []
    for variant in ["r6a_qwen_score_only", "r6b_multi_score_only", "r6c_low_high_score_only"]:
        run_name = f"{variant}_from_r2c_s100_b0p03_lr5em6"
        output_dir = f"saves/edubench/qwen3-4b/{run_name}"
        mapping = {**base, "dataset": DATASET_NAMES[variant], "output_dir": output_dir}
        text = "\n".join(f"{key}: {str(value).lower() if isinstance(value, bool) else value}" for key, value in mapping.items())
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
    write_csv(out_dir / "tables" / "r6_run_matrix.csv", run_rows)


def write_report(
    out_dir: Path,
    train_rows: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    datasets: dict[str, list[dict[str, Any]]],
) -> None:
    risk_counts = Counter(record["risk_type"] for record in pool)
    source_counts: Counter[str] = Counter()
    for record in pool:
        for source in record["sources"]:
            source_counts[source] += 1
    lines = [
        "# Exp19-R6 Real DPO Dataset Report",
        "",
        "Exp19-R6 builds DPO preference pairs from train split only. The rejected side is always a",
        "real score emitted by an existing judge or by Qwen3-4B direct baseline; no template or hard",
        "synthetic rejected response is used.",
        "",
        "## Construction Rule",
        "",
        "- input prompt: question + answer + metric + rubric + metadata.",
        "- chosen response: human gold score, optionally with structured hidden-failure fields.",
        "- rejected response: real wrong model score from train-side reusable judge outputs.",
        "- strict main variants are score-only because reusable real rejected outputs are scalar scores.",
        "- structured/reason chosen variants are auxiliary review artifacts; their rejected side remains score-only.",
        "",
        "## Guardrails",
        "",
        "- test split is not read.",
        "- dev/D1 annotations are not used for training labels.",
        "- human rationale is not used as prompt input.",
        "- rejected responses are not handwritten templates.",
        "- full DPO JSON is stored under gitignored `data/`; lightweight review/QC files are committed.",
        "",
        "## Train / Pair Counts",
        "",
        f"- train samples: {len(train_rows)}",
        f"- unique real rejected pair pool: {len(pool)}",
    ]
    for risk_type, count in sorted(risk_counts.items()):
        lines.append(f"- {risk_type}: {count}")
    lines.extend(["", "## Rejected Sources", ""])
    for source, count in sorted(source_counts.items()):
        lines.append(f"- {source}: {count}")
    lines.extend(["", "## Dataset Variants", ""])
    for variant, pairs in datasets.items():
        role = "main score-only training candidate" if variant in MAIN_TRAINING_VARIANTS else "auxiliary review candidate"
        lines.append(f"- {DATASET_NAMES[variant]}: {len(pairs)} pairs; {role}.")
    lines.extend(
        [
            "",
            "## Recommended Review Order",
            "",
            "1. Review `r6b_multi_judge_score_real_dpo_train` as the main real-DPO candidate.",
            "2. Review `r6c_low_high_score_real_dpo_train` if the next DPO run should focus only on risk/protection.",
            "3. Treat structured/reason chosen variants as ablations, not as the cleanest real-DPO baseline.",
        ]
    )
    write_text(out_dir / "reports" / "exp19_r6_real_dpo_dataset_report.md", "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Exp19-R6 real rejected-output DPO datasets.")
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--a0-candidates", type=Path, default=DEFAULT_A0_CANDIDATES)
    parser.add_argument("--r0a-predictions", type=Path, default=DEFAULT_R0A_PREDICTIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--redacted-limit-per-type", type=int, default=8)
    args = parser.parse_args()

    train_rows = read_jsonl(args.train_jsonl)
    a0_by_id = {clean(row.get("sample_id")): row for row in read_csv_rows(args.a0_candidates)}
    r0a_by_id = read_r0a_predictions(args.r0a_predictions)
    row_by_id = {sample_id(row): row for row in train_rows}
    pool = build_pair_pool(train_rows, a0_by_id, r0a_by_id)
    datasets = build_datasets(pool, args.seed)

    data_dir = args.out_dir / "data"
    for variant, pairs in datasets.items():
        write_json_array(args.out_dir / DATASET_FILES[variant], pairs)

    write_json(args.out_dir / "dataset_info_r6_snippet.json", dataset_info())
    write_json(args.out_dir / "dataset_info.json", dataset_info())
    write_configs(args.out_dir)

    count_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for variant, pairs in datasets.items():
        quality_rows.extend(dataset_stats(variant, pairs))
        source_rows.extend(source_stats(variant, pairs))
        by_risk = Counter(pair["risk_type"] for pair in pairs)
        count_rows.extend(
            {
                "dataset_variant": variant,
                "dataset_name": DATASET_NAMES[variant],
                "chosen_schema": schema_for_variant(variant),
                "risk_type": risk_type,
                "pairs": count,
                "is_main_training_candidate": variant in MAIN_TRAINING_VARIANTS,
            }
            for risk_type, count in sorted(by_risk.items())
        )
        manifest.extend(manifest_rows(variant, pairs))
        write_json(
            args.out_dir / "redacted_samples" / f"{DATASET_NAMES[variant]}_redacted.json",
            redacted_samples(variant, pairs, row_by_id, args.redacted_limit_per_type),
        )

    write_csv(args.out_dir / "tables" / "r6_pair_counts.csv", count_rows)
    write_csv(args.out_dir / "tables" / "r6_pair_quality.csv", quality_rows)
    write_csv(args.out_dir / "tables" / "r6_rejected_source_counts.csv", source_rows)
    write_csv(args.out_dir / "tables" / "r6_train_label_coverage.csv", coverage_rows(pool, train_rows))
    write_csv(args.out_dir / "review" / "r6_real_dpo_review_manifest.csv", manifest)
    write_report(args.out_dir, train_rows, pool, datasets)

    decision = {
        "safe_for_dpo_data_review": True,
        "recommended_main_dataset": DATASET_NAMES["r6b_multi_score_only"],
        "recommended_risk_focused_dataset": DATASET_NAMES["r6c_low_high_score_only"],
        "train_samples": len(train_rows),
        "unique_real_rejected_pair_pool": len(pool),
        "dataset_pair_counts": {DATASET_NAMES[key]: len(value) for key, value in datasets.items()},
        "guardrails": {
            "no_test_read": True,
            "no_dev_d1_training_labels": True,
            "human_rationale_not_used_as_prompt_input": True,
            "hard_synthetic_rejected_used": False,
            "template_rejected_used": False,
            "rejected_side_is_real_model_score": True,
        },
        "caveats": [
            "Main variants are score-only because reusable real rejected outputs are scalar scores.",
            "Structured/reason chosen variants are auxiliary; they should be reviewed before training.",
            "Existing judge_scores provide real scalar judgments, not full natural-language model outputs.",
        ],
    }
    write_json(args.out_dir / "decision" / "exp19_r6_real_dpo_dataset_decision.json", decision)

    print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
