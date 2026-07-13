"""Select and freeze 240 reliable paper-like train source anchors for Exp39A."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp39_educfa.common import (  # noqa: E402
    OPERATORS,
    PROMPT_DIR,
    ROOT,
    SCHEMA_DIR,
    TARGET_COUNTS,
    TRAIN_PATH,
    human_distribution,
    read_jsonl,
    sample_id,
    score_value,
    sha256_file,
    stable_hash,
    write_csv,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, default=TRAIN_PATH)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--rows", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def stratum(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row.get("language") or "unknown"),
        str(row.get("metric_group") or "unknown"),
        str(row.get("subject_canonical") or row.get("subject_raw") or "unknown"),
        int(row["label_5"]),
    )


def eligible(row: dict[str, Any]) -> bool:
    scores = [score_value(row, key) for key in ("human_1", "human_2", "human_3")]
    return (
        sum(score >= 4 for score in scores) >= 2
        and all(score > 2 for score in scores)
        and int(row["label_5"]) in {4, 5}
        and bool(str(row.get("answer") or "").strip())
        and bool(row.get("rubric"))
        and bool(str(row.get("question_key") or "").strip())
    )


def select_rows(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    pool = [row for row in rows if eligible(row)]
    if len(pool) < count:
        raise ValueError(f"Only {len(pool)} source anchors satisfy the locked criteria; need {count}")
    available = Counter(stratum(row) for row in pool)
    selected_strata: Counter[tuple[str, str, str, int]] = Counter()
    qkey_counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    remaining = {sample_id(row): row for row in pool}
    while len(selected) < count:
        candidates = [row for row in remaining.values() if qkey_counts[str(row["question_key"])] < 3]
        if not candidates:
            raise RuntimeError("Could not satisfy the three-source-per-question_key cap")

        def rank(row: dict[str, Any]) -> tuple[Any, ...]:
            key = stratum(row)
            qkey = str(row["question_key"])
            representation = selected_strata[key] / available[key]
            return (
                1 if qkey_counts[qkey] else 0,
                qkey_counts[qkey],
                representation,
                stable_hash({"seed": seed, "sample_id": sample_id(row)}),
            )

        chosen = min(candidates, key=rank)
        selected.append(chosen)
        selected_strata[stratum(chosen)] += 1
        qkey_counts[str(chosen["question_key"])] += 1
        del remaining[sample_id(chosen)]
    return selected


def assigned_payloads(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    targets = [score for score, count in TARGET_COUNTS.items() for _ in range(count)]
    operators = [operator for operator in OPERATORS for _ in range(len(rows) // len(OPERATORS))]
    operators.extend(OPERATORS[: len(rows) - len(operators)])
    rng = random.Random(seed)
    rng.shuffle(targets)
    rng.shuffle(operators)
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row.get("language") or "unknown"),
            str(row.get("metric_group") or "unknown"),
            str(row.get("subject_canonical") or "unknown"),
            int(row["label_5"]),
            stable_hash({"seed": seed, "sample_id": sample_id(row)}),
        ),
    )
    output = []
    for row, target, operator in zip(ordered, targets, operators):
        source_id = sample_id(row)
        synthetic_id = "exp39a_" + stable_hash(
            {"source_sample_id": source_id, "target_score": target, "operator": operator, "seed": seed}
        )[:24]
        metadata = {
            "language": row.get("language"),
            "subject": row.get("subject_canonical") or row.get("subject_raw"),
            "education_level": row.get("education_level_canonical") or row.get("education_level_raw"),
            "scenario": row.get("scenario_canonical") or row.get("scenario_raw"),
            "metric_group": row.get("metric_group"),
        }
        output.append(
            {
                "sample_id": synthetic_id,
                "source_sample_id": source_id,
                "question_key": str(row["question_key"]),
                "question": row.get("question", ""),
                "original_answer": row.get("answer", ""),
                "metric": row.get("metric_canonical") or row.get("metric_raw") or row.get("metric_id", ""),
                "metric_group": row.get("metric_group"),
                "rubric": row.get("rubric"),
                "metadata": metadata,
                "language": row.get("language"),
                "subject": metadata["subject"],
                "source_label_5": int(row["label_5"]),
                "source_human_distribution_5": human_distribution(row),
                "assigned_operator": operator,
                "assigned_target_score": target,
            }
        )
    return output


def ensure_dirs(out_dir: Path) -> None:
    for name in (
        "configs", "tables", "reports", "decision", "hashes", "raw_api",
        "private/source_packets", "private/generated_candidates", "private/verified_counterfactuals",
        "private/data", "private/groupcv_predictions", "private/checkpoints", "logs_private",
    ):
        (out_dir / name).mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    ensure_dirs(args.out_dir)
    rows = read_jsonl(args.train_jsonl)
    if len(rows) != 2654 or len({sample_id(row) for row in rows}) != 2654:
        raise ValueError("Paper-like train must contain 2654 unique rows")
    selected = select_rows(rows, args.rows, args.seed)
    packets = assigned_payloads(selected, args.seed)
    if Counter(row["assigned_target_score"] for row in packets) != Counter(TARGET_COUNTS):
        raise RuntimeError("Target assignment does not match the frozen 40/120/80 distribution")
    if max(Counter(row["question_key"] for row in packets).values()) > 3:
        raise RuntimeError("Source selection violated the three-row question_key cap")

    packet_path = args.out_dir / "private/source_packets/exp39a_source_anchor_packets.jsonl"
    write_jsonl(packet_path, packets)
    distribution = []
    dimensions = {
        "language": lambda row: row["language"],
        "metric_group": lambda row: row["metric_group"],
        "subject": lambda row: row["subject"],
        "source_label": lambda row: row["source_label_5"],
        "target_score": lambda row: row["assigned_target_score"],
        "operator": lambda row: row["assigned_operator"],
    }
    for dimension, getter in dimensions.items():
        for value, count in sorted(Counter(str(getter(row)) for row in packets).items()):
            distribution.append({"dimension": dimension, "value": value, "count": count, "rate": count / len(packets)})
    qkey_counts = Counter(row["question_key"] for row in packets)
    write_csv(args.out_dir / "tables/exp39a_source_anchor_distribution.csv", distribution)
    write_csv(
        args.out_dir / "tables/exp39a_source_anchor_qkey_distribution.csv",
        [{"sources_per_question_key": value, "question_key_count": count} for value, count in sorted(Counter(qkey_counts.values()).items())],
    )

    generator_prompt = PROMPT_DIR / "exp39a_qwen_counterfactual_generator.md"
    verifier_prompt = PROMPT_DIR / "exp39a_deepseek_blind_counterfactual_verifier.md"
    generation_schema = SCHEMA_DIR / "exp39a_counterfactual_generation_schema.json"
    verification_schema = SCHEMA_DIR / "exp39a_counterfactual_verification_schema.json"
    lock = {
        "experiment": "Exp39A EduCFA",
        "train_path": str(args.train_jsonl),
        "train_sha256": sha256_file(args.train_jsonl),
        "train_rows": len(rows),
        "source_rows": len(packets),
        "source_question_keys": len(qkey_counts),
        "max_sources_per_question_key": max(qkey_counts.values()),
        "selection_seed": args.seed,
        "selection_uses_teacher_scores": False,
        "selection_uses_model_predictions": False,
        "full_processed_dataset_accessed": False,
        "dev_access_count": 0,
        "test_access_count": 0,
    }
    protocol = {
        "qwen_model": "qwen3.7-max",
        "deepseek_model": "deepseek-v4-pro",
        "temperature": 0,
        "qwen_max_tokens": 2048,
        "deepseek_max_tokens": 1024,
        "operators": list(OPERATORS),
        "target_counts": {str(score): count for score, count in TARGET_COUNTS.items()},
        "generator_prompt_sha256": sha256_file(generator_prompt),
        "verifier_prompt_sha256": sha256_file(verifier_prompt),
        "generation_schema_sha256": sha256_file(generation_schema),
        "verification_schema_sha256": sha256_file(verification_schema),
        "acceptance_rules_frozen": True,
        "dev_access_count": 0,
        "test_access_count": 0,
    }
    lock_path = args.out_dir / "configs/exp39a_source_lock.json"
    protocol_path = args.out_dir / "configs/exp39a_generation_protocol_lock.json"
    for path, value in ((lock_path, lock), (protocol_path, protocol)):
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != value:
            raise RuntimeError(f"Frozen Exp39A lock changed: {path}")
        write_json(path, value)
    write_json(args.out_dir / "hashes/exp39a_source_anchor_hashes.json", {
        "source_sample_ids_sha256": stable_hash(sorted(row["source_sample_id"] for row in packets)),
        "source_question_keys_sha256": stable_hash(sorted(row["question_key"] for row in packets)),
        "assignment_sha256": stable_hash([
            [row["source_sample_id"], row["assigned_target_score"], row["assigned_operator"]] for row in packets
        ]),
        "private_packet_sha256": sha256_file(packet_path),
    })
    print(json.dumps({
        "status": "PREPARED", "source_rows": len(packets), "source_question_keys": len(qkey_counts),
        "target_counts": dict(Counter(row["assigned_target_score"] for row in packets)),
        "operator_counts": dict(Counter(row["assigned_operator"] for row in packets)),
        "dev_access_count": 0, "test_access_count": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
