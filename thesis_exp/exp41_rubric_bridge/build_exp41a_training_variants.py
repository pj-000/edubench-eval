"""Build the six locked Exp41A train-only rubric representation variants."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp41_rubric_bridge.common import (  # noqa: E402
    ROOT, TRAIN_PATH, VARIANTS, canonical_model_text, deterministic_checklist, human_stats,
    raw_rubric_text, read_jsonl, rubric_unit_id, sample_id, sha256_file, stable_hash,
    write_csv, write_json, write_jsonl,
)

DEFAULT_TOKENIZER = "/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, default=TRAIN_PATH)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--tokenizer-name-or-path", default=DEFAULT_TOKENIZER)
    return parser.parse_args()


def percentile(values: list[int], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else float("nan")


def build_shuffle_map(units: dict[str, dict[str, Any]]) -> dict[str, str]:
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for unit_id, row in units.items():
        strata[(str(row["language"]), str(row["metric_group"]))].append(unit_id)
    mapping: dict[str, str] = {}
    for key, unit_ids in sorted(strata.items()):
        ordered = sorted(unit_ids, key=lambda uid: (int(units[uid]["compiled_token_length"]), len(units[uid]["criteria"]), uid))
        if len(ordered) < 2:
            mapping[ordered[0]] = ordered[0]
            continue
        shift = max(1, len(ordered) // 2)
        if math.gcd(shift, len(ordered)) != 1:
            shift = 1
        for index, unit_id in enumerate(ordered):
            mapping[unit_id] = ordered[(index + shift) % len(ordered)]
    return mapping


def main() -> None:
    args = parse_args()
    decision_path = args.out_dir / "decision/exp41a_compiler_qualification_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if not decision.get("recommend_groupcv_training"):
        raise RuntimeError("Variant construction blocked by compiler qualification NO-GO")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, local_files_only=True)
    rows = read_jsonl(args.train_jsonl)
    if len(rows) != 2654:
        raise ValueError(f"Expected 2654 train rows, found {len(rows)}")
    compiled_rows = read_jsonl(args.out_dir / "private/compiled_rubrics/exp41a_fitted_compiled_rubrics.jsonl")
    compiled = {str(row["rubric_unit_id"]): row for row in compiled_rows}
    if len(compiled) != 1044:
        raise ValueError(f"Expected 1044 fitted compiled rubrics, found {len(compiled)}")
    unit_meta = {
        str(row["rubric_unit_id"]): row
        for row in read_jsonl(args.out_dir / "private/rubric_units/exp41a_rubric_units.jsonl")
    }
    compiled_units = {
        uid: {**row, "language": unit_meta[uid]["language"], "metric_group": unit_meta[uid]["metric_group"]}
        for uid, row in compiled.items()
    }
    shuffle_map = build_shuffle_map(compiled_units)
    mapping_change = sum(source != target for source, target in shuffle_map.items()) / len(shuffle_map)
    if mapping_change < 0.90:
        raise RuntimeError(f"V4 shuffled mapping change below 0.90: {mapping_change:.4f}")

    class_counts = Counter(int(row["label_5"]) for row in rows)
    priors = [class_counts[label] / len(rows) for label in range(1, 6)]
    variants: dict[str, list[dict[str, Any]]] = {variant: [] for variant in VARIANTS}
    core_identity = []
    for position, row in enumerate(rows):
        sid = sample_id(row)
        uid = rubric_unit_id(row)
        if uid not in compiled:
            raise RuntimeError(f"Missing compiled rubric unit: {uid}")
        stats = human_stats(row)
        soft = stats["human_distribution_5"]
        hard = [1.0 if label == int(row["label_5"]) else 0.0 for label in range(1, 6)]
        deterministic = deterministic_checklist(raw_rubric_text(row))
        representations: dict[str, str | None] = {
            "v0_original_hard": None,
            "v0h_human_soft": None,
            "v1_raw_rubric": raw_rubric_text(row),
            "v2_deterministic_checklist": deterministic,
            "v3_rubric_bridge": compiled[uid]["serialized_text"],
            "v4_shuffled_compiled_rubric": compiled[shuffle_map[uid]]["serialized_text"],
            "v5_human_soft_logit_adjustment": None,
        }
        core_hash = stable_hash((sid, row.get("question"), row.get("answer"), row.get("metric_canonical")))
        core_identity.append(core_hash)
        for variant in VARIANTS:
            text, audit = canonical_model_text(tokenizer, row, representations[variant])
            target = hard if variant == "v0_original_hard" else soft
            output = {
                "position": position, "variant": variant, "sample_id": sid,
                "question_key": str(row["question_key"]), "rubric_unit_id": uid,
                "source_rubric_unit_id": shuffle_map[uid] if variant == "v4_shuffled_compiled_rubric" else uid,
                "text": text, "soft_target_5": target, "human_distribution_5": soft,
                "gold_label_5": int(row["label_5"]), "expected_human_score": stats["expected_human_score"],
                "human_entropy": stats["human_entropy"], "human_score_range": stats["human_score_range"],
                "sample_weight": 1.0,
                "language": row.get("language"), "metric_group": row.get("metric_group"),
                "subject_canonical": row.get("subject_canonical"), "core_input_hash": core_hash,
                **audit,
            }
            variants[variant].append(output)

    reference_ids = [row["sample_id"] for row in variants["v0_original_hard"]]
    reference_soft = [row["human_distribution_5"] for row in variants["v0h_human_soft"]]
    rubric_variants = [variants[name] for name in (
        "v1_raw_rubric", "v2_deterministic_checklist", "v3_rubric_bridge", "v4_shuffled_compiled_rubric"
    )]
    if any(
        [(row["answer_tokens"], row["answer_truncated"], row["answer_token_budget"]) for row in candidate]
        != [(row["answer_tokens"], row["answer_truncated"], row["answer_token_budget"]) for row in rubric_variants[0]]
        for candidate in rubric_variants[1:]
    ):
        raise RuntimeError("V1/V2/V3/V4 answer truncation is not identical")
    equivalence = []
    summaries = []
    hashes: dict[str, Any] = {}
    for variant, variant_rows in variants.items():
        if len(variant_rows) != 2654 or [row["sample_id"] for row in variant_rows] != reference_ids:
            raise RuntimeError(f"Variant row/order mismatch: {variant}")
        if variant != "v0_original_hard" and [row["soft_target_5"] for row in variant_rows] != reference_soft:
            raise RuntimeError(f"Human target mismatch: {variant}")
        if any(abs(sum(row["soft_target_5"]) - 1.0) > 1e-8 or min(row["soft_target_5"]) < 0 for row in variant_rows):
            raise RuntimeError(f"Invalid target distribution: {variant}")
        if any(float(row["sample_weight"]) != 1.0 for row in variant_rows):
            raise RuntimeError(f"Invalid sample weight: {variant}")
        path = args.out_dir / f"private/data/exp41a_{variant}.jsonl"
        write_jsonl(path, variant_rows)
        input_lengths = [int(row["input_tokens"]) for row in variant_rows]
        summaries.append({
            "variant": variant, "rows": len(variant_rows), "unique_sample_ids": len(set(reference_ids)),
            "unique_question_keys": len({row["question_key"] for row in variant_rows}),
            "target_sum_valid_rate": 1.0, "sample_weight_mean": 1.0,
            "mean_input_tokens": float(np.mean(input_lengths)), "p95_input_tokens": percentile(input_lengths, 95),
            "answer_truncation_count": sum(bool(row["answer_truncated"]) for row in variant_rows),
            "answer_truncation_rate": float(np.mean([bool(row["answer_truncated"]) for row in variant_rows])),
            "rubric_truncation_count": sum(bool(row["rubric_truncated"]) for row in variant_rows),
            "rubric_truncation_rate": float(np.mean([bool(row["rubric_truncated"]) for row in variant_rows])),
        })
        equivalence.append({
            "variant": variant, "rows_equal": len(variant_rows) == 2654,
            "sample_id_order_equal": [row["sample_id"] for row in variant_rows] == reference_ids,
            "core_hash_equal": [row["core_input_hash"] for row in variant_rows] == core_identity,
            "human_targets_equal": [row["human_distribution_5"] for row in variant_rows] == reference_soft,
            "sample_weights_all_one": all(float(row["sample_weight"]) == 1.0 for row in variant_rows),
        })
        hashes[variant] = {
            "path": str(path), "rows": len(variant_rows), "sha256": sha256_file(path),
            "sample_order_sha256": stable_hash(reference_ids), "core_identity_sha256": stable_hash(core_identity),
            "human_target_sha256": stable_hash([row["human_distribution_5"] for row in variant_rows]),
        }
    v3_lengths = sorted(int(compiled[uid]["compiled_token_length"]) for uid in compiled)
    v4_lengths = sorted(int(compiled[shuffle_map[uid]]["compiled_token_length"]) for uid in compiled)
    shuffled_audit = [{
        "rubric_units": len(shuffle_map), "mapping_change_count": sum(a != b for a, b in shuffle_map.items()),
        "mapping_change_rate": mapping_change, "token_length_multiset_equal": v3_lengths == v4_lengths,
        "criterion_count_multiset_equal": sorted(len(compiled[uid]["criteria"]) for uid in compiled) == sorted(len(compiled[shuffle_map[uid]]["criteria"]) for uid in compiled),
        "same_language_metric_group_rate": float(np.mean([
            (compiled_units[source]["language"], compiled_units[source]["metric_group"]) ==
            (compiled_units[target]["language"], compiled_units[target]["metric_group"])
            for source, target in shuffle_map.items()
        ])),
    }]
    write_csv(args.out_dir / "tables/exp41a_variant_summary.csv", summaries)
    write_csv(args.out_dir / "tables/exp41a_variant_equivalence.csv", equivalence)
    write_csv(args.out_dir / "tables/exp41a_input_length_audit.csv", [
        {key: value for key, value in row.items() if key in {
            "variant", "rows", "mean_input_tokens", "p95_input_tokens", "answer_truncation_count",
            "answer_truncation_rate", "rubric_truncation_count", "rubric_truncation_rate",
        }} for row in summaries
    ])
    write_csv(args.out_dir / "tables/exp41a_shuffled_control_audit.csv", shuffled_audit)
    write_json(args.out_dir / "hashes/exp41a_dataset_hashes.json", {
        "variants": hashes, "class_priors_5": priors, "shuffle_mapping_sha256": stable_hash(shuffle_map),
        "tokenizer_name_or_path": args.tokenizer_name_or_path, "dev_access_count": 0, "test_access_count": 0,
    })
    print(json.dumps({"status": "BUILT", "variants": len(variants), "rows_per_variant": 2654,
                      "mapping_change_rate": mapping_change, "dev_access_count": 0, "test_access_count": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
