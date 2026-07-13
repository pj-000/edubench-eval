"""Build the six locked Exp39A train-only supervision variants after data GO."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp39_educfa.common import (  # noqa: E402
    ROOT,
    TRAIN_PATH,
    VARIANTS,
    half_up,
    human_distribution,
    interval_distribution,
    read_jsonl,
    sample_id,
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
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def original_row(row: dict[str, Any]) -> dict[str, Any]:
    target = human_distribution(row)
    sid = sample_id(row)
    return {
        **row,
        "sample_id": sid,
        "source_sample_id": sid,
        "original_label_5": int(row["label_5"]),
        "soft_target_5": target,
        "sample_weight": 1.0,
        "synthetic": False,
        "original_evaluation_row": True,
        "exp39a_payload_type": "original_human_empirical",
    }


def synthetic_row(source: dict[str, Any], answer: str, target: list[float], sid: str, payload_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    expected = sum((index + 1) * value for index, value in enumerate(target))
    return {
        **source,
        "sample_id": sid,
        "source_sample_id": sample_id(source),
        "answer": answer,
        "original_label_5": int(source["label_5"]),
        "label_5": half_up(expected),
        "soft_target_5": target,
        "sample_weight": 1.0,
        "synthetic": True,
        "original_evaluation_row": False,
        "exp39a_payload_type": payload_type,
        **payload,
    }


def select_matched_low(train: list[dict[str, Any]], accepted: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    low = [row for row in train if int(row["label_5"]) <= 2]
    if not low:
        raise ValueError("No real label1/2 rows available for V1")
    output = []
    for index, candidate in enumerate(accepted):
        desired = int(candidate["assigned_target_score"])

        def rank(row: dict[str, Any]) -> tuple[Any, ...]:
            return (
                0 if row.get("language") == candidate.get("language") else 1,
                0 if row.get("metric_group") == candidate.get("metric_group") else 1,
                0 if (row.get("subject_canonical") or row.get("subject_raw")) == candidate.get("subject") else 1,
                abs(int(row["label_5"]) - min(desired, 2)),
                stable_hash({"seed": seed, "index": index, "sample_id": sample_id(row)}),
            )

        output.append(min(low, key=rank))
    return output


def generic_corruption(answer: str, index: int) -> tuple[str, str]:
    sentences = [item.strip() for item in re.split(r"(?<=[.!?。！？])\s*", answer) if item.strip()]
    mode = index % 3
    if mode == 0 and len(sentences) >= 2:
        drop = max(0, len(sentences) // 2)
        return " ".join(sentences[:drop] + sentences[drop + 1 :]), "sentence_deletion"
    if mode == 1 and len(sentences) >= 2:
        swapped = list(sentences)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        return " ".join(swapped), "sentence_swap"
    corrupted = re.sub(r"[\n\r:;：；,，]", " ", answer)
    corrupted = re.sub(r"\s+", " ", corrupted).strip()
    if corrupted == answer.strip():
        corrupted = answer.strip()[:-1] if len(answer.strip()) > 1 else answer.strip() + "?"
    return corrupted, "format_corruption"


def deranged_payloads(rows: list[dict[str, Any]], seed: int) -> tuple[dict[str, dict[str, Any]], float]:
    strata: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[(int(row["assigned_target_score"]), str(row.get("language")), str(row.get("metric_group")))].append(row)
    mapping: dict[str, dict[str, Any]] = {}
    changed = 0
    for key in sorted(strata):
        members = sorted(strata[key], key=lambda row: stable_hash({"seed": seed, "sample_id": row["sample_id"]}))
        donors = members[1:] + members[:1] if len(members) > 1 else members
        for recipient, donor in zip(members, donors):
            mapping[str(recipient["sample_id"])] = donor
            changed += donor["sample_id"] != recipient["sample_id"]
    return mapping, changed / max(len(rows), 1)


def main() -> None:
    args = parse_args()
    decision = json.loads((args.out_dir / "decision/exp39a_data_qualification_decision.json").read_text(encoding="utf-8"))
    if not decision.get("recommend_groupcv_training"):
        raise RuntimeError("Exp39A variant construction is blocked by data qualification NO-GO")
    train = read_jsonl(args.train_jsonl)
    accepted = read_jsonl(args.out_dir / "private/verified_counterfactuals/exp39a_accepted_counterfactuals.jsonl")
    generated = read_jsonl(args.out_dir / "private/generated_candidates/exp39a_qwen_generated_candidates.jsonl")
    if len(train) != 2654 or not accepted:
        raise ValueError("Expected 2654 original rows and at least one accepted counterfactual")
    source_by_id = {sample_id(row): row for row in train}
    original = [original_row(row) for row in train]
    accepted_count = len(accepted)

    v1_extra = []
    for index, row in enumerate(select_matched_low(train, accepted, args.seed)):
        target = human_distribution(row)
        v1_extra.append(synthetic_row(
            row, str(row["answer"]), target,
            "exp39a_v1_" + stable_hash({"source": sample_id(row), "copy": index, "seed": args.seed})[:24],
            "real_low_oversample", {"oversample_copy_index": index},
        ))

    generated_sorted = sorted(generated, key=lambda row: stable_hash({"seed": args.seed, "sample_id": row["sample_id"]}))[:accepted_count]
    packet_by_id = {row["sample_id"]: row for row in read_jsonl(args.out_dir / "private/source_packets/exp39a_source_anchor_packets.jsonl")}
    v2_extra = []
    for row in generated_sorted:
        packet = packet_by_id[row["sample_id"]]
        source = source_by_id[packet["source_sample_id"]]
        target = interval_distribution(int(row["target_score_range"][0]), int(row["target_score_range"][1]), int(row["target_score"]))
        v2_extra.append(synthetic_row(
            source, row["counterfactual_answer"], target, "exp39a_v2_" + str(row["sample_id"]),
            "unverified_counterfactual", {"assigned_target_score": row["target_score"], "assigned_operator": row["operator"]},
        ))

    v3_extra = []
    for index, row in enumerate(accepted):
        source = source_by_id[row["source_sample_id"]]
        answer, mode = generic_corruption(str(source["answer"]), index)
        v3_extra.append(synthetic_row(
            source, answer, row["soft_target_5"], "exp39a_v3_" + stable_hash({"source": row["source_sample_id"], "index": index})[:24],
            "generic_corruption", {"assigned_target_score": row["assigned_target_score"], "generic_corruption_mode": mode},
        ))

    v4_extra = []
    for row in accepted:
        source = source_by_id[row["source_sample_id"]]
        v4_extra.append(synthetic_row(
            source, row["counterfactual_answer"], row["soft_target_5"], "exp39a_v4_" + str(row["sample_id"]),
            "verified_rubric_counterfactual", {
                "assigned_target_score": row["assigned_target_score"], "assigned_operator": row["assigned_operator"],
                "edit_ratio": row["edit_ratio"], "length_ratio": row["length_ratio"],
            },
        ))

    shuffled, change_rate = deranged_payloads(accepted, args.seed)
    v5_extra = []
    for recipient in accepted:
        donor = shuffled[recipient["sample_id"]]
        source = source_by_id[recipient["source_sample_id"]]
        v5_extra.append(synthetic_row(
            source, donor["counterfactual_answer"], donor["soft_target_5"], "exp39a_v5_" + str(recipient["sample_id"]),
            "shuffled_counterfactual", {
                "assigned_target_score": donor["assigned_target_score"], "assigned_operator": donor["assigned_operator"],
                "shuffled_from_sample_id_hash": stable_hash(donor["sample_id"]),
                "shuffled_pair_changed": donor["sample_id"] != recipient["sample_id"],
            },
        ))
    if change_rate < 0.80:
        raise RuntimeError(f"V5 actual-change rate {change_rate:.4f} is below the locked 0.80 minimum")

    datasets = {
        "v0h_human_soft": original,
        "v1_matched_real_low_oversampling": original + v1_extra,
        "v2_unverified_counterfactual": original + v2_extra,
        "v3_generic_corruption": original + v3_extra,
        "v4_educfa": original + v4_extra,
        "v5_shuffled_counterfactual": original + v5_extra,
    }
    data_dir = args.out_dir / "private/data"
    hashes = {}
    summaries = []
    equivalence = []
    original_ids = [sample_id(row) for row in train]
    for variant in VARIANTS:
        rows = datasets[variant]
        path = data_dir / f"exp39a_{variant}.jsonl"
        write_jsonl(path, rows)
        original_prefix = [row["sample_id"] for row in rows[:2654]]
        targets_valid = all(len(row["soft_target_5"]) == 5 and min(row["soft_target_5"]) >= 0 and abs(sum(row["soft_target_5"]) - 1) <= 1e-8 for row in rows)
        summaries.append({
            "variant": variant, "rows": len(rows), "original_rows": sum(not row["synthetic"] for row in rows),
            "additional_rows": sum(row["synthetic"] for row in rows), "unique_question_keys": len({row["question_key"] for row in rows}),
            **{f"target_mass_{score}": sum(row["soft_target_5"][score - 1] for row in rows) for score in range(1, 6)},
        })
        equivalence.append({
            "variant": variant, "same_original_2654_prefix": original_prefix == original_ids,
            "all_original_rows_unchanged": all(not row["synthetic"] for row in rows[:2654]),
            "all_sample_weights_one": all(float(row["sample_weight"]) == 1.0 for row in rows),
            "all_targets_valid": targets_valid,
            "synthetic_never_evaluation": all(not row["original_evaluation_row"] for row in rows[2654:]),
        })
        hashes[variant] = {"path": str(path), "rows": len(rows), "sha256": sha256_file(path)}
    if not all(all(value for key, value in row.items() if key != "variant") for row in equivalence):
        raise RuntimeError("Variant equivalence validation failed")
    write_csv(args.out_dir / "tables/exp39a_variant_summary.csv", summaries)
    write_csv(args.out_dir / "tables/exp39a_variant_equivalence.csv", equivalence)
    write_csv(args.out_dir / "tables/exp39a_shuffled_control_audit.csv", [{
        "rows": accepted_count, "actual_change_rate": change_rate,
        "target_marginal_preserved": Counter(row["assigned_target_score"] for row in accepted) == Counter(row["assigned_target_score"] for row in shuffled.values()),
        "target_entropy_multiset_preserved": sorted(tuple(row["soft_target_5"]) for row in accepted) == sorted(tuple(row["soft_target_5"]) for row in shuffled.values()),
        "strata": len({(row["assigned_target_score"], row.get("language"), row.get("metric_group")) for row in accepted}),
    }])
    write_json(args.out_dir / "hashes/exp39a_training_config_hashes.json", hashes)
    write_json(args.out_dir / "configs/exp39a_training_matrix_lock.json", {
        "variants": VARIANTS, "original_rows_per_variant": 2654, "accepted_synthetic_rows": accepted_count,
        "sample_weight": 1.0, "loss": "standard_soft_cross_entropy", "epochs": 10,
        "learning_rate": 2e-5, "micro_batch": 4, "gradient_accumulation": 32,
        "weight_decay": 0.01, "warmup_ratio": 0.05, "max_length": 2048,
        "method_attribution_rule": {
            "no_material_harm": {"MAE": 0.005, "Exact_Match": -0.005, "QWK": -0.01, "label5_recall": -0.02, "high_to_low_rate": 0.01},
            "at_least_one_material_gain": {"MAE": 0.005, "low_to_high_relative_reduction": 0.10, "label2_recall": 0.05},
        },
        "seed": args.seed, "custom_loss": False, "model_architecture_change": False,
        "dev_access_count": 0, "test_access_count": 0,
    })
    print(json.dumps({
        "status": "BUILT", "accepted_synthetic_rows": accepted_count,
        "variant_rows": {key: len(value) for key, value in datasets.items()},
        "shuffled_actual_change_rate": change_rate, "dev_access_count": 0, "test_access_count": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
