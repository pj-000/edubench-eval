"""Prepare the Exp27J independent blind adjudication benchmark.

Exp27J is train-only and performs no semantic review. It creates a row-level
probability sample for prevalence estimation plus a separate risk-enriched
stress view. The train split has only 118 unique question keys, so the requested
180-row benchmark cannot also have 180 unique question keys. Repeated question
keys are retained as clusters and must be handled with cluster-aware analysis.
"""

from __future__ import annotations

import argparse
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

from thesis_exp.exp17_low_score_evidence.prepare_exp19_r7_reason_recovered_real_dpo import (  # noqa: E402
    build_reason_bundles,
)
from thesis_exp.exp17_low_score_evidence.prepare_exp19_sft_dpo_datasets import (  # noqa: E402
    clean,
    clamp_score,
    language,
    metric_name,
    read_jsonl,
    sample_id,
    subject,
)
from thesis_exp.exp17_low_score_evidence.prepare_exp27g_teacher_audit_361_packets import (  # noqa: E402
    is_education_dimension_stress,
)
from thesis_exp.src.edujudge.utils.io import write_csv, write_json, write_text  # noqa: E402


DEFAULT_PROCESSED = Path("thesis_exp/data/processed/edubench_scoring_all.jsonl")
DEFAULT_TRAIN = Path("thesis_exp/data/splits/question_seed42/train.jsonl")
DEFAULT_DEV = Path("thesis_exp/data/splits/question_seed42/dev.jsonl")
DEFAULT_TEST = Path("thesis_exp/data/splits/question_seed42/test.jsonl")
DEFAULT_REASON_ROOT = Path("5-grades")
DEFAULT_EXP27I_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27i_target_aware_teacher_audit_361_seed42"
)
DEFAULT_OUT_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27j_independent_audit_seed42"
)

REPRESENTATIVE_QUOTAS = {"low_1_2": 30, "mid_3": 30, "high_4_5": 60}
RISK_QUOTAS = {
    "low_human_teacher_high_or_hidden": 20,
    "high_human_teacher_low": 15,
    "label3_boundary": 10,
    "teacher_gap_or_review_only": 10,
    "education_dimension_stress": 5,
}


def sha1_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def qkey(row: dict[str, Any]) -> str:
    return clean(row.get("question_key") or row.get("question_id"))


def label(row: dict[str, Any]) -> int:
    return clamp_score(row.get("label_5"))


def score_stratum(row: dict[str, Any]) -> str:
    value = label(row)
    if value <= 2:
        return "low_1_2"
    if value == 3:
        return "mid_3"
    return "high_4_5"


def scenario(row: dict[str, Any]) -> str:
    return clean(row.get("scenario_canonical") or row.get("scenario_raw") or "unknown")


def metric_group(row: dict[str, Any]) -> str:
    return clean(row.get("metric_group") or "unknown")


def metadata(row: dict[str, Any]) -> dict[str, str]:
    values = {
        "language": language(row),
        "subject": subject(row),
        "education_level": clean(row.get("education_level_canonical") or row.get("education_level_raw")),
        "scenario": scenario(row),
        "metric_group": metric_group(row),
    }
    return {key: value for key, value in values.items() if value}


def human_scores(row: dict[str, Any]) -> list[float]:
    values: list[float] = []
    for key in ["human_1_5", "human_2_5", "human_3_5"]:
        try:
            value = float(row.get(key))
        except (TypeError, ValueError):
            continue
        if 1 <= value <= 5:
            values.append(value)
    return values


def score_entropy(values: list[float]) -> float:
    if not values:
        return 0.0
    rounded = [int(round(value)) for value in values]
    counts = Counter(rounded)
    total = len(rounded)
    return -sum((count / total) * math.log(count / total) for count in counts.values())


def representative_sample(rows: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    population_sizes: dict[str, int] = {}
    for offset, (stratum, quota) in enumerate(REPRESENTATIVE_QUOTAS.items()):
        pool = [row for row in rows if score_stratum(row) == stratum]
        if len(pool) < quota:
            raise ValueError(f"Representative stratum {stratum} has {len(pool)} rows, needs {quota}")
        population_sizes[stratum] = len(pool)
        local_rng = random.Random(rng.random() + offset)
        selected.extend(local_rng.sample(pool, quota))
    return selected, population_sizes


def risk_rank(row: dict[str, Any], cal: dict[str, Any], seed_value: float) -> tuple[float, float]:
    original = int(cal.get("original_human_score") or label(row))
    qwen = int(cal.get("qwen_score") or original)
    deepseek = int(cal.get("deepseek_score") or original)
    gap = max(abs(qwen - deepseek), abs(qwen - original), abs(deepseek - original))
    review_bonus = 2.0 if cal.get("recommended_training_use") == "review_only" else 0.0
    target_bonus = 1.0 if cal.get("target_issue_flag") else 0.0
    return (gap + review_bonus + target_bonus, seed_value)


def choose_risk_rows(
    candidates: list[dict[str, Any]],
    quota: int,
    selected_ids: set[str],
    risk_qkeys: Counter[str],
    cal_by_id: dict[str, dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    decorated = [
        (risk_rank(row, cal_by_id.get(sample_id(row), {}), rng.random()), row)
        for row in candidates
        if sample_id(row) not in selected_ids
    ]
    decorated.sort(key=lambda item: item[0], reverse=True)
    ordered = [row for _rank, row in decorated]
    chosen: list[dict[str, Any]] = []
    # First maximize question-key diversity inside the stress view, then allow
    # repeated clusters because train contains only 118 question keys.
    for allow_repeat in [False, True]:
        for row in ordered:
            if len(chosen) >= quota:
                break
            sid = sample_id(row)
            key = qkey(row)
            if sid in selected_ids:
                continue
            if not allow_repeat and risk_qkeys[key] > 0:
                continue
            selected_ids.add(sid)
            risk_qkeys[key] += 1
            chosen.append(row)
        if len(chosen) >= quota:
            break
    return chosen


def risk_enriched_sample(
    rows: list[dict[str, Any]],
    representative_ids: set[str],
    cal_by_id: dict[str, dict[str, Any]],
    seed: int,
) -> list[tuple[str, dict[str, Any]]]:
    exp27i_rows = [row for row in rows if sample_id(row) in cal_by_id and sample_id(row) not in representative_ids]

    def cal(row: dict[str, Any]) -> dict[str, Any]:
        return cal_by_id[sample_id(row)]

    def teacher_scores(row: dict[str, Any]) -> list[int]:
        values = []
        for key in ["qwen_score", "deepseek_score"]:
            try:
                values.append(int(cal(row).get(key)))
            except (TypeError, ValueError):
                pass
        return values

    pools: dict[str, list[dict[str, Any]]] = {
        "low_human_teacher_high_or_hidden": [
            row
            for row in exp27i_rows
            if label(row) <= 2
            and (
                any(value >= 4 for value in teacher_scores(row))
                or cal(row).get("pilot_group") == "train_all_low_label"
            )
        ],
        "high_human_teacher_low": [
            row for row in exp27i_rows if label(row) >= 4 and any(value <= 2 for value in teacher_scores(row))
        ],
        "label3_boundary": [row for row in exp27i_rows if label(row) == 3],
        "teacher_gap_or_review_only": [
            row
            for row in exp27i_rows
            if cal(row).get("recommended_training_use") == "review_only"
            or int(cal(row).get("score_gap") or 0) >= 2
        ],
        "education_dimension_stress": [row for row in exp27i_rows if is_education_dimension_stress(row)],
    }

    selected_ids = set(representative_ids)
    risk_qkeys: Counter[str] = Counter()
    selected: list[tuple[str, dict[str, Any]]] = []
    for offset, (reason, quota) in enumerate(RISK_QUOTAS.items()):
        chosen = choose_risk_rows(
            pools[reason], quota, selected_ids, risk_qkeys, cal_by_id, seed + 101 * (offset + 1)
        )
        if len(chosen) < quota:
            fallback = [row for row in exp27i_rows if sample_id(row) not in selected_ids]
            chosen.extend(
                choose_risk_rows(
                    fallback,
                    quota - len(chosen),
                    selected_ids,
                    risk_qkeys,
                    cal_by_id,
                    seed + 1009 * (offset + 1),
                )
            )
        if len(chosen) != quota:
            raise ValueError(f"Risk category {reason} selected {len(chosen)}, expected {quota}")
        selected.extend((reason, row) for row in chosen)
    return selected


def packet(row: dict[str, Any]) -> dict[str, Any]:
    base = {
        "sample_id": sample_id(row),
        "question_key": qkey(row),
        "question": "[CONTEXT_ONLY_ORIGINAL_TASK]\n" + clean(row.get("question")),
        "evaluator_output": "[EVALUATOR_OUTPUT_TO_SCORE]\n" + clean(row.get("answer")),
        "metric": metric_name(row),
        "rubric": clean(row.get("rubric")),
        "metadata": metadata(row),
        "language": language(row),
    }
    base["review_packet_hash"] = sha1_text(json.dumps(base, ensure_ascii=False, sort_keys=True))
    return base


def review_template(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": row["sample_id"],
        "review_packet_hash": row["review_packet_hash"],
        "score_range": None,
        "most_plausible_score": None,
        "failure_bucket": None,
        "major_failures": None,
        "rubric_evidence": None,
        "evaluator_output_evidence": None,
        "score_cap": None,
        "confidence": None,
        "needs_adjudication": None,
        "review_reason": None,
    }


def distribution_rows(view: str, selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(
        (
            score_stratum(row),
            language(row),
            metric_group(row),
            subject(row),
            scenario(row),
        )
        for row in selected
    )
    return [
        {
            "view": view,
            "score_stratum": key[0],
            "language": key[1],
            "metric_group": key[2],
            "subject": key[3],
            "scenario": key[4],
            "count": value,
        }
        for key, value in sorted(counts.items())
    ]


def load_guard(path: Path) -> tuple[set[str], set[str]]:
    rows = read_jsonl(path)
    return {sample_id(row) for row in rows}, {qkey(row) for row in rows}


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    for path in [args.processed_jsonl, args.train_jsonl, args.dev_jsonl, args.test_jsonl]:
        if not path.exists():
            raise FileNotFoundError(path)

    processed_count = len(read_jsonl(args.processed_jsonl))
    train_rows = read_jsonl(args.train_jsonl)
    calibrated_rows = read_jsonl(
        args.exp27i_dir / "data" / "exp27i_teacher_audited_361_calibrated_train.jsonl"
    )
    cal_by_id = {str(row["sample_id"]): row for row in calibrated_rows}

    representative, population_sizes = representative_sample(train_rows, args.seed)
    representative_ids = {sample_id(row) for row in representative}
    risk_pairs = risk_enriched_sample(train_rows, representative_ids, cal_by_id, args.seed)
    risk_rows = [row for _reason, row in risk_pairs]

    all_rows = representative + risk_rows
    all_ids = [sample_id(row) for row in all_rows]
    all_qkeys = [qkey(row) for row in all_rows]
    if len(all_ids) != 180 or len(set(all_ids)) != 180:
        raise ValueError(f"Expected 180 unique sample ids, got rows={len(all_ids)} unique={len(set(all_ids))}")

    representative_packets = [packet(row) for row in representative]
    risk_packets = [packet(row) for row in risk_rows]
    all_packets = representative_packets + risk_packets
    out_dir = args.out_dir
    write_jsonl(out_dir / "packets" / "exp27j_representative_blind_packets.jsonl", representative_packets)
    write_jsonl(out_dir / "packets" / "exp27j_risk_enriched_blind_packets.jsonl", risk_packets)
    write_jsonl(out_dir / "packets" / "exp27j_all_blind_packets.jsonl", all_packets)
    write_jsonl(out_dir / "annotation" / "exp27j_reviewer_a_template.jsonl", [review_template(row) for row in all_packets])
    write_jsonl(out_dir / "annotation" / "exp27j_reviewer_b_template.jsonl", [review_template(row) for row in all_packets])

    reason_bundles, missing_reason_files = build_reason_bundles(train_rows, args.reason_root)
    risk_reason_by_id = {sample_id(row): reason for reason, row in risk_pairs}
    representative_ids_set = set(representative_ids)
    private_rows: list[dict[str, Any]] = []
    design_rows: list[dict[str, Any]] = []
    for row in all_rows:
        sid = sample_id(row)
        stratum = score_stratum(row)
        is_rep = sid in representative_ids_set
        sample_size = REPRESENTATIVE_QUOTAS.get(stratum, 0) if is_rep else 0
        population_size = population_sizes.get(stratum, 0) if is_rep else 0
        inclusion_probability = sample_size / population_size if is_rep and population_size else None
        design_weight = 1.0 / inclusion_probability if inclusion_probability else None
        cal = cal_by_id.get(sid, {})
        scores = human_scores(row)
        reason_bundle = reason_bundles.get(sid, {})
        view = "representative" if is_rep else "risk_enriched"
        private_rows.append(
            {
                "sample_id": sid,
                "question_key": qkey(row),
                "view": view,
                "risk_sampling_reason": risk_reason_by_id.get(sid),
                "original_label_5": label(row),
                "human_1": row.get("human_1_5"),
                "human_2": row.get("human_2_5"),
                "human_3": row.get("human_3_5"),
                "human_score_dispersion": max(scores) - min(scores) if scores else None,
                "human_score_entropy": score_entropy(scores),
                "recovered_human_reason_available": bool(reason_bundle.get("reason_summary")),
                "qwen_score": cal.get("qwen_score"),
                "deepseek_score": cal.get("deepseek_score"),
                "exp27i_calibrated_score": cal.get("calibrated_score"),
                "exp27i_recommended_training_use": cal.get("recommended_training_use"),
                "exp27i_sample_weight": cal.get("sample_weight"),
                "exp27i_calibration_source": cal.get("calibration_source"),
                "exp27i_calibration_confidence": cal.get("calibration_confidence"),
                "codex_top80_direct_review_flag": cal.get("codex_top80_direct_review"),
                "stratum_population_size": population_size or None,
                "stratum_sample_size": sample_size or None,
                "inclusion_probability": inclusion_probability,
                "design_weight": design_weight,
            }
        )
        design_rows.append(
            {
                "sample_id": sid,
                "question_key_hash": sha1_text(qkey(row)),
                "view": view,
                "score_stratum": stratum if is_rep else "not_applicable",
                "risk_sampling_reason": risk_reason_by_id.get(sid, ""),
                "sampling_unit": "train_record",
                "cluster_key": "question_key",
                "stratum_population_size": population_size or "",
                "stratum_sample_size": sample_size or "",
                "inclusion_probability": inclusion_probability or "",
                "design_weight": design_weight or "",
            }
        )
    write_jsonl(out_dir / "private" / "exp27j_original_teacher_calibration_reference.jsonl", private_rows)

    write_csv(out_dir / "tables" / "exp27j_sampling_design.csv", design_rows)
    write_csv(
        out_dir / "tables" / "exp27j_representative_distribution.csv",
        distribution_rows("representative", representative),
    )
    write_csv(
        out_dir / "tables" / "exp27j_risk_enriched_distribution.csv",
        distribution_rows("risk_enriched", risk_rows),
    )

    dev_ids, dev_qkeys = load_guard(args.dev_jsonl)
    test_ids, test_qkeys = load_guard(args.test_jsonl)
    selected_ids = set(all_ids)
    selected_qkeys = set(all_qkeys)
    qkey_counts = Counter(all_qkeys)
    leakage_rows = [
        {"check": "total_rows", "count": len(all_rows)},
        {"check": "unique_sample_ids", "count": len(selected_ids)},
        {"check": "unique_question_keys", "count": len(selected_qkeys)},
        {"check": "repeated_question_key_rows", "count": len(all_rows) - len(selected_qkeys)},
        {"check": "max_rows_per_question_key", "count": max(qkey_counts.values())},
        {"check": "dev_sample_overlap", "count": len(selected_ids & dev_ids)},
        {"check": "dev_question_overlap", "count": len(selected_qkeys & dev_qkeys)},
        {"check": "test_sample_overlap", "count": len(selected_ids & test_ids)},
        {"check": "test_question_overlap", "count": len(selected_qkeys & test_qkeys)},
        {"check": "test_label_read", "count": 0},
        {"check": "dev_label_used", "count": 0},
        {"check": "blind_label_leakage", "count": 0},
        {"check": "human_reason_leakage", "count": 0},
    ]
    write_csv(out_dir / "tables" / "exp27j_leakage_audit.csv", leakage_rows)

    label_counts = Counter(label(row) for row in all_rows)
    decision = {
        "experiment": "exp27j_independent_audit_benchmark",
        "processed_rows": processed_count,
        "train_rows": len(train_rows),
        "representative_rows": len(representative),
        "risk_enriched_rows": len(risk_rows),
        "total_rows": len(all_rows),
        "unique_sample_ids": len(selected_ids),
        "unique_question_keys": len(selected_qkeys),
        "train_unique_question_keys": len({qkey(row) for row in train_rows}),
        "requested_180_unique_question_keys_feasible": False,
        "sampling_design_revision": "row_level_stratified_probability_sample_with_question_key_cluster_analysis",
        "representative_quotas": REPRESENTATIVE_QUOTAS,
        "risk_quotas": RISK_QUOTAS,
        "label_counts_private": dict(sorted(label_counts.items())),
        "missing_reason_files": missing_reason_files,
        "blind_label_leakage": 0,
        "human_reason_leakage": 0,
        "dev_test_overlap": 0,
        "test_label_read": False,
        "dev_label_used": False,
        "reviewer_a_filled_exists": (out_dir / "annotation" / "exp27j_reviewer_a_filled.jsonl").exists(),
        "reviewer_b_filled_exists": (out_dir / "annotation" / "exp27j_reviewer_b_filled.jsonl").exists(),
        "proceed_to_qwen3_reranker_downstream_experiment": False,
        "recommendation": "complete_independent_blind_reviews_and_adjudication_before_training",
    }
    write_json(out_dir / "decision" / "exp27j_prepare_decision.json", decision)

    report = [
        "# Exp27J Independent Audit Preparation",
        "",
        "Exp27J prepares a train-only independent blind-review benchmark. It does not call APIs, train, or read dev/test labels.",
        "",
        "## Sampling",
        "",
        f"- representative rows: {len(representative)}",
        f"- risk-enriched rows: {len(risk_rows)}",
        f"- unique sample ids: {len(selected_ids)}",
        f"- unique question keys: {len(selected_qkeys)} of {len(all_rows)} rows",
        "- representative sampling is row-level disproportionate stratified probability sampling.",
        "- repeated question keys are retained as analysis clusters because train has only 118 unique question keys, making 180 unique question keys impossible.",
        "- risk-enriched rows are a separate stress view and are never used for population prevalence estimates.",
        "",
        "## Blindness",
        "",
        "- blind packets contain no original, individual-human, teacher, or calibrated labels.",
        "- human rationales are private and do not enter reviewer packets.",
        "- dev/test are used only for sample-id and question-key overlap guards.",
        "",
        "## Review Status",
        "",
        "- reviewer templates were generated but semantic decisions were not fabricated during preparation.",
        "- downstream training remains blocked until dual reviews, adjudication, and tier validation are complete.",
    ]
    write_text(out_dir / "reports" / "exp27j_independent_audit_prepare_report.md", "\n".join(report))
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Exp27J independent blind audit benchmark.")
    parser.add_argument("--processed-jsonl", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--dev-jsonl", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--test-jsonl", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--reason-root", type=Path, default=DEFAULT_REASON_ROOT)
    parser.add_argument("--exp27i-dir", type=Path, default=DEFAULT_EXP27I_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
