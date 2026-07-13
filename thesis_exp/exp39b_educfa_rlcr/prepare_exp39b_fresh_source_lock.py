"""Select and freeze 60 fresh train-only Exp39B source anchors, or fail before API use."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp39b_educfa_rlcr.common import (  # noqa: E402
    BAND_COUNTS,
    BANDS,
    EXP39A_LOCK,
    EXP39A_PACKETS,
    PROMPT_DIR,
    ROOT,
    SCHEMA_DIR,
    TRAIN_PATH,
    ensure_layout,
    read_jsonl,
    sample_id,
    score_value,
    sha256_file,
    stable_hash,
    token_count,
    write_csv,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, default=TRAIN_PATH)
    parser.add_argument("--exp39a-lock", type=Path, default=EXP39A_LOCK)
    parser.add_argument("--exp39a-packets", type=Path, default=EXP39A_PACKETS)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--rows", type=int, default=60)
    parser.add_argument("--seed", type=int, default=43)
    return parser.parse_args()


def eligible(row: dict[str, Any]) -> bool:
    human = [score_value(row, key) for key in ("human_1", "human_2", "human_3")]
    return (
        sum(score >= 4 for score in human) >= 2
        and all(score > 2 for score in human)
        and int(row["label_5"]) in {4, 5}
        and bool(str(row.get("answer") or "").strip())
        and bool(str(row.get("question_key") or "").strip())
        and bool(row.get("rubric"))
        and token_count(str(row.get("answer") or "")) >= 20
    )


def stratum(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(row.get("language") or "unknown"),
        str(row.get("metric_group") or "unknown"),
        int(row["label_5"]),
    )


def select_balanced(pool: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    available = Counter(stratum(row) for row in pool)
    selected_strata: Counter[tuple[str, str, int]] = Counter()
    qkey_counts: Counter[str] = Counter()
    remaining = {sample_id(row): row for row in pool}
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        candidates = [row for row in remaining.values() if qkey_counts[str(row["question_key"])] < 2]
        if not candidates:
            raise RuntimeError("Could not satisfy the two-source-per-question_key cap")

        def rank(row: dict[str, Any]) -> tuple[Any, ...]:
            key = stratum(row)
            qkey = str(row["question_key"])
            return (
                1 if qkey_counts[qkey] else 0,
                qkey_counts[qkey],
                selected_strata[key] / available[key],
                stable_hash({"seed": seed, "sample_id": sample_id(row)}),
            )

        chosen = min(candidates, key=rank)
        selected.append(chosen)
        selected_strata[stratum(chosen)] += 1
        qkey_counts[str(chosen["question_key"])] += 1
        del remaining[sample_id(chosen)]
    return selected


def assign_bands(rows: list[dict[str, Any]], seed: int) -> dict[str, str]:
    remaining = Counter(BAND_COUNTS)
    assigned: dict[str, str] = {}
    strata: dict[str, Counter[tuple[str, str, int]]] = {band: Counter() for band in BAND_COUNTS}
    ordered = sorted(rows, key=lambda row: (stratum(row), stable_hash({"seed": seed, "id": sample_id(row)})))
    rng = random.Random(seed)
    tie_order = list(BAND_COUNTS)
    rng.shuffle(tie_order)
    for row in ordered:
        key = stratum(row)
        candidates = [band for band in tie_order if remaining[band] > 0]
        band = min(
            candidates,
            key=lambda name: (
                strata[name][key] / BAND_COUNTS[name],
                (BAND_COUNTS[name] - remaining[name]) / BAND_COUNTS[name],
                tie_order.index(name),
            ),
        )
        assigned[sample_id(row)] = band
        remaining[band] -= 1
        strata[band][key] += 1
    if any(remaining.values()):
        raise RuntimeError(f"Band assignment incomplete: {dict(remaining)}")
    return assigned


def packet(row: dict[str, Any], band_name: str, seed: int) -> dict[str, Any]:
    source_id = sample_id(row)
    exp_id = "exp39b_" + stable_hash({"source_sample_id": source_id, "band": band_name, "seed": seed})[:24]
    return {
        "sample_id": exp_id,
        "source_sample_id": source_id,
        "question_key": str(row["question_key"]),
        "question": str(row.get("question") or ""),
        "original_answer": str(row.get("answer") or ""),
        "metric": str(row.get("metric_canonical") or row.get("metric_raw") or row.get("metric_id") or ""),
        "metric_group": str(row.get("metric_group") or "unknown"),
        "rubric": row.get("rubric"),
        "metadata": {
            "language": row.get("language"),
            "subject": row.get("subject_canonical") or row.get("subject_raw"),
            "education_level": row.get("education_level_canonical") or row.get("education_level_raw"),
            "scenario": row.get("scenario_canonical") or row.get("scenario_raw"),
            "metric_group": row.get("metric_group"),
        },
        "language": str(row.get("language") or "unknown"),
        "subject": str(row.get("subject_canonical") or row.get("subject_raw") or "unknown"),
        "source_label_5": int(row["label_5"]),
        "target_band_name": band_name,
        "target_band": BANDS[band_name],
    }


def freeze(path: Path, value: dict[str, Any]) -> None:
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != value:
        raise RuntimeError(f"Frozen Exp39B artifact changed: {path}")
    write_json(path, value)


def main() -> None:
    args = parse_args()
    ensure_layout(args.out_dir)
    rows = read_jsonl(args.train_jsonl)
    exp39a_rows = read_jsonl(args.exp39a_packets)
    if len(rows) != 2654 or len({sample_id(row) for row in rows}) != 2654:
        raise ValueError("Paper-like train must contain 2654 unique rows")
    exp39a_lock = json.loads(args.exp39a_lock.read_text(encoding="utf-8"))
    if int(exp39a_lock.get("source_rows", -1)) != len(exp39a_rows):
        raise ValueError("Exp39A source lock and private packets disagree")

    used_ids = {str(row["source_sample_id"]) for row in exp39a_rows}
    used_qkeys = {str(row["question_key"]) for row in exp39a_rows}
    eligible_before_exclusion = [row for row in rows if eligible(row)]
    pool = [
        row for row in eligible_before_exclusion
        if sample_id(row) not in used_ids and str(row["question_key"]) not in used_qkeys
    ]
    remaining_qkeys = {str(row["question_key"]) for row in pool}
    can_prepare = len(pool) >= args.rows and len(remaining_qkeys) >= 40
    selected = select_balanced(pool, args.rows, args.seed) if can_prepare else []
    assigned = assign_bands(selected, args.seed) if selected else {}
    packets = [packet(row, assigned[sample_id(row)], args.seed) for row in selected]
    packet_path = args.out_dir / "private/source_packets/exp39b_source_anchor_packets.jsonl"
    write_jsonl(packet_path, packets)

    overlap = [{
        "train_rows": len(rows),
        "train_question_keys": len({str(row["question_key"]) for row in rows}),
        "exp39a_source_rows": len(used_ids),
        "exp39a_question_keys": len(used_qkeys),
        "eligible_before_exclusion": len(eligible_before_exclusion),
        "eligible_after_source_id_exclusion": sum(sample_id(row) not in used_ids for row in eligible_before_exclusion),
        "eligible_after_source_and_qkey_exclusion": len(pool),
        "remaining_question_keys": len(remaining_qkeys),
        "selected_source_overlap": sum(row["source_sample_id"] in used_ids for row in packets),
        "selected_qkey_overlap": sum(row["question_key"] in used_qkeys for row in packets),
        "dev_access_count": 0,
        "test_access_count": 0,
    }]
    write_csv(args.out_dir / "tables/exp39b_exp39a_overlap_audit.csv", overlap)
    distribution = []
    for dimension, getter in {
        "language": lambda row: row["language"],
        "metric_group": lambda row: row["metric_group"],
        "source_label": lambda row: row["source_label_5"],
        "target_band": lambda row: row["target_band_name"],
    }.items():
        for value, count in sorted(Counter(str(getter(row)) for row in packets).items()):
            distribution.append({"dimension": dimension, "value": value, "count": count, "rate": count / len(packets)})
    write_csv(
        args.out_dir / "tables/exp39b_source_distribution.csv",
        distribution,
        fieldnames=["dimension", "value", "count", "rate"],
    )
    qcounts = Counter(row["question_key"] for row in packets)
    write_csv(
        args.out_dir / "tables/exp39b_source_qkey_distribution.csv",
        [{"sources_per_question_key": count, "question_key_count": number} for count, number in sorted(Counter(qcounts.values()).items())],
        fieldnames=["sources_per_question_key", "question_key_count"],
    )

    prompt_paths = sorted(PROMPT_DIR.glob("exp39b_*.md"))
    schema_paths = sorted(SCHEMA_DIR.glob("exp39b_*.json"))
    for path in prompt_paths:
        shutil.copy2(path, args.out_dir / "prompts" / path.name)
    for path in schema_paths:
        shutil.copy2(path, args.out_dir / "schemas" / path.name)
    source_lock = {
        "experiment": "Exp39B EduCFA-RLCR Pilot",
        "status": "SOURCE_LOCKED" if can_prepare else "SOURCE_POOL_NO_GO",
        "train_path": str(args.train_jsonl),
        "train_sha256": sha256_file(args.train_jsonl),
        "train_rows": len(rows),
        "train_question_keys": len({str(row["question_key"]) for row in rows}),
        "exp39a_source_rows_excluded": len(used_ids),
        "exp39a_question_keys_excluded": len(used_qkeys),
        "eligible_after_exclusion": len(pool),
        "eligible_question_keys_after_exclusion": len(remaining_qkeys),
        "source_rows": len(packets),
        "source_question_keys": len(qcounts),
        "selection_seed": args.seed,
        "selection_uses_teacher_scores": False,
        "selection_uses_model_predictions": False,
        "dev_access_count": 0,
        "test_access_count": 0,
    }
    protocol_lock = {
        "experiment": "Exp39B EduCFA-RLCR Pilot",
        "seed": args.seed,
        "qwen_model": "qwen3.7-max",
        "deepseek_model": "deepseek-v4-pro",
        "temperature": 0,
        "thinking": "disabled",
        "target_band_counts": BAND_COUNTS,
        "target_bands": BANDS,
        "planner_valid_min": 54,
        "accepted_min": 45,
        "acceptance_rules_frozen": True,
        "dev_access_count": 0,
        "test_access_count": 0,
    }
    acceptance_gate = {
        "fresh_sources": 60, "source_qkeys_min": 40, "plan_valid_min": 54,
        "source_original_high_min": 57, "accepted_min": 45, "target_band_coverage_min": 48,
        "severe_low_accepted_min": 14, "moderate_low_accepted_min": 18, "boundary_accepted_min": 11,
        "accepted_final_score_le2_min": 25, "accepted_final_score_eq3_min": 12,
        "rubric_failure_rate_min": 0.90, "non_target_preservation_rate_min": 0.90,
        "outside_span_identity_required": 1.0, "one_span_only_required": 1.0,
        "duplicate_rate_max": 0.01, "low_band_entirely_high_max": 0,
        "target_coverage_improvement_min": 0.10, "revision_rescue_rate_min": 0.30,
        "dev_access_count": 0, "test_access_count": 0,
    }
    freeze(args.out_dir / "configs/exp39b_source_lock.json", source_lock)
    freeze(args.out_dir / "configs/exp39b_protocol_lock.json", protocol_lock)
    freeze(args.out_dir / "configs/exp39b_acceptance_gate.json", acceptance_gate)
    write_json(args.out_dir / "hashes/exp39b_source_lock_hashes.json", {
        "exp39a_source_ids_sha256": stable_hash(sorted(used_ids)),
        "exp39a_question_keys_sha256": stable_hash(sorted(used_qkeys)),
        "selected_source_ids_sha256": stable_hash(sorted(row["source_sample_id"] for row in packets)),
        "selected_question_keys_sha256": stable_hash(sorted(row["question_key"] for row in packets)),
        "private_packet_sha256": sha256_file(packet_path),
    })
    write_json(args.out_dir / "hashes/exp39b_prompt_schema_hashes.json", {
        "prompts": {path.name: sha256_file(path) for path in prompt_paths},
        "schemas": {path.name: sha256_file(path) for path in schema_paths},
    })
    decision = {
        "status": "PROTOCOL_PREPARE_GO" if can_prepare else "SOURCE_POOL_NO_GO",
        "fresh_source_count": len(packets),
        "fresh_question_key_count": len(qcounts),
        "eligible_after_exclusion": len(pool),
        "eligible_question_keys_after_exclusion": len(remaining_qkeys),
        "recommend_api_calls": can_prepare,
        "failure_reason": None if can_prepare else "Exp39A covers every paper-like train question_key",
        "dev_access_count": 0,
        "test_access_count": 0,
    }
    write_json(args.out_dir / "decision/exp39b_protocol_prepare_decision.json", decision)
    report = [
        "# Exp39B EduCFA-RLCR protocol preparation", "",
        f"- Status: **{decision['status']}**",
        f"- Train rows / question keys: `{len(rows)} / {source_lock['train_question_keys']}`",
        f"- Exp39A excluded source rows / question keys: `{len(used_ids)} / {len(used_qkeys)}`",
        f"- Eligible before exclusion: `{len(eligible_before_exclusion)}`",
        f"- Eligible after source and question-key exclusion: `{len(pool)}` across `{len(remaining_qkeys)}` question keys",
        f"- Selected fresh sources: `{len(packets)}`",
        "- No teacher/model/dev/test signal was used for source selection.",
        "- Qwen/DeepSeek API calls are forbidden unless this preparation decision is GO.",
    ]
    if not can_prepare:
        report.extend([
            "", "## Pre-registered stop", "",
            "Exp39A source packets cover all paper-like train question keys. The required 60 fresh rows with at least 40 fresh question keys do not exist under the frozen isolation rule.",
            "The question-key exclusion was not relaxed and no API was called.",
        ])
    (args.out_dir / "reports/exp39b_protocol_prepare_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(decision, sort_keys=True))
    if not can_prepare:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
