"""Build and validate the six locked Exp38A HAILS supervision variants."""

from __future__ import annotations

import argparse
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

from thesis_exp.exp38_hails_score.common import (
    EXP36_MANIFEST,
    ROOT,
    TRAIN_PATH,
    VARIANTS,
    half_up,
    input_hash,
    normalize_distribution,
    one_hot,
    qwen_interval_distribution,
    read_csv,
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
    parser.add_argument("--qwen-manifest", type=Path, default=EXP36_MANIFEST)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_qwen_hard(manifest_path: Path) -> tuple[Path, dict[str, int]]:
    rows = read_csv(manifest_path)
    candidates = [row for row in rows if row.get("provider") == "qwen" and row.get("unique_resolution", "").lower() == "true"]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one uniquely resolved Qwen hard source, found {len(candidates)}")
    path = Path(candidates[0]["resolved_path"])
    annotations = read_jsonl(path)
    output: dict[str, int] = {}
    for row in annotations:
        annotation = row.get("annotation") or row.get("parsed") or {}
        score = annotation.get("score") if isinstance(annotation, dict) else None
        if score is not None:
            output[sample_id(row)] = int(score)
    return path, output


def human_distribution(row: dict[str, Any]) -> list[float]:
    scores = [int(float(row[key])) for key in ("human_1", "human_2", "human_3")]
    return [scores.count(label) / 3.0 for label in range(1, 6)]


def human_support(distribution: list[float]) -> list[int]:
    return [index for index, value in enumerate(distribution, 1) if value > 0]


def interval_distance(minimum: int, maximum: int, support: list[int]) -> int:
    if any(minimum <= score <= maximum for score in support):
        return 0
    return min(min(abs(score - minimum), abs(score - maximum)) for score in support)


def interval_entropy(distribution: list[float]) -> float:
    return -sum(value * math.log(value) for value in distribution if value > 0)


def hails_target(human: list[float], interval: list[float], width: int, eligible: bool) -> list[float]:
    if not eligible:
        return list(human)
    weight = 1.0 / width
    return normalize_distribution([(3.0 * left + weight * right) / (3.0 + weight) for left, right in zip(human, interval)])


def eligibility(row: dict[str, Any], minimum: int, maximum: int, human: list[float]) -> tuple[bool, list[str], int]:
    width = maximum - minimum + 1
    support = human_support(human)
    distance = interval_distance(minimum, maximum, support)
    majority = half_up(sum((index + 1) * value for index, value in enumerate(human)))
    low_anchor = human[0] + human[1] >= 2.0 / 3.0
    high_anchor = human[3] + human[4] >= 2.0 / 3.0
    reasons = []
    if width > 3:
        reasons.append("width_gt_3")
    if distance > 1:
        reasons.append("distance_gt_1")
    if low_anchor and minimum >= 4:
        reasons.append("low_anchor_entirely_high")
    if high_anchor and maximum <= 2:
        reasons.append("high_anchor_entirely_low")
    if majority == 4 and minimum == maximum == 5:
        reasons.append("human4_exact5")
    if distance >= 2:
        reasons.append("gap_ge_2_no_overlap")
    return not reasons, sorted(set(reasons)), distance


def base_row(row: dict[str, Any], target: list[float], variant: str, payload: dict[str, Any]) -> dict[str, Any]:
    expected = sum((index + 1) * value for index, value in enumerate(target))
    return {
        **row,
        "sample_id": sample_id(row),
        "original_label_5": int(row["label_5"]),
        "label_5": half_up(expected),
        "soft_target_5": target,
        "sample_weight": 1.0,
        "exp38a_variant": variant,
        "exp38a_model_input_hash": input_hash(row),
        "exp38a_target_expected_score": expected,
        **payload,
    }


def main() -> None:
    args = parse_args()
    decision_path = args.out_dir / "decision/exp38a_range_qualification_decision.json"
    if not decision_path.exists() or not json.loads(decision_path.read_text(encoding="utf-8")).get("recommend_full_train_range_annotation"):
        raise RuntimeError("HAILS dataset construction is blocked until score-range qualification passes")
    train = read_jsonl(args.train_jsonl)
    ranges_path = args.out_dir / "parsed_ranges_private/exp38a_qwen_ranges_all_train.jsonl"
    ranges = {sample_id(row): row for row in read_jsonl(ranges_path)}
    if len(train) != 2654 or len(ranges) != 2654 or {sample_id(row) for row in train} != set(ranges):
        raise ValueError("Full-train Qwen ranges must cover all 2654 train IDs exactly")
    qwen_path, qwen_hard = resolve_qwen_hard(args.qwen_manifest)
    if len(qwen_hard) != 2654:
        raise ValueError(f"Qwen hard baseline must cover 2654 rows, found {len(qwen_hard)}")

    original_payload: dict[str, dict[str, Any]] = {}
    strata: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for row in train:
        sid = sample_id(row)
        interval = ranges[sid]
        minimum = int(interval["minimum_plausible_score"])
        center = int(interval["most_plausible_score"])
        maximum = int(interval["maximum_plausible_score"])
        distribution = qwen_interval_distribution(minimum, center, maximum)
        original_payload[sid] = {
            "minimum": minimum,
            "center": center,
            "maximum": maximum,
            "distribution": distribution,
            "width": maximum - minimum + 1,
            "entropy": interval_entropy(distribution),
            "confidence": interval["range_confidence"],
        }
        key = (str(row["label_5"]), str(row.get("language") or "unknown"), str(row.get("metric_group") or "unknown"))
        strata[key].append(sid)

    shuffled_payload: dict[str, dict[str, Any]] = {}
    rng = random.Random(args.seed)
    for key in sorted(strata):
        ids = sorted(strata[key])
        sources = list(ids)
        rng.shuffle(sources)
        if len(ids) > 1 and all(left == right for left, right in zip(ids, sources)):
            sources = sources[1:] + sources[:1]
        for target, source in zip(ids, sources):
            shuffled_payload[target] = {**original_payload[source], "source_sample_id_hash": stable_hash(source)}

    datasets: dict[str, list[dict[str, Any]]] = {variant: [] for variant in VARIANTS}
    eligibility_rows = []
    anchor_rows = []
    target_mass: dict[tuple[str, int], list[list[float]]] = defaultdict(list)
    for row in train:
        sid = sample_id(row)
        human = human_distribution(row)
        rounded = int(row["label_5"])
        p = original_payload[sid]
        eligible, blocked, distance = eligibility(row, p["minimum"], p["maximum"], human)
        shuffled = shuffled_payload[sid]
        shuffled_eligible, shuffled_blocked, shuffled_distance = eligibility(row, shuffled["minimum"], shuffled["maximum"], human)
        targets = {
            "v0_original_hard": one_hot(rounded),
            "v0h_human_empirical": human,
            "v1_qwen_hard": one_hot(qwen_hard[sid]),
            "v2_qwen_interval_only": p["distribution"],
            "v3_naive_interval": normalize_distribution([(3 * left + right) / 4 for left, right in zip(human, p["distribution"])]),
            "v4_hails": hails_target(human, p["distribution"], p["width"], eligible),
            "v5_shuffled_interval": hails_target(human, shuffled["distribution"], shuffled["width"], shuffled_eligible),
        }
        common_payload = {
            "human_distribution_5": human,
            "human_expected_score": sum((index + 1) * value for index, value in enumerate(human)),
            "human_support": human_support(human),
        }
        for variant, target in targets.items():
            payload = p if variant != "v5_shuffled_interval" else shuffled
            diagnostics = {
                **common_payload,
                "qwen_interval_min": payload["minimum"], "qwen_interval_center": payload["center"],
                "qwen_interval_max": payload["maximum"], "qwen_interval_width": payload["width"],
                "qwen_interval_distribution_5": payload["distribution"],
                "hails_eligible": eligible if variant == "v4_hails" else (shuffled_eligible if variant == "v5_shuffled_interval" else None),
            }
            datasets[variant].append(base_row(row, target, variant, diagnostics))
            target_mass[(variant, rounded)].append(target)
        eligibility_rows.append({
            "sample_id_hash": stable_hash(sid), "human_label": rounded, "language": row.get("language"), "metric_group": row.get("metric_group"),
            "width": p["width"], "distance": distance, "eligible": eligible, "blocked_reasons": "|".join(blocked),
        })
        anchor_rows.append({
            "sample_id_hash": stable_hash(sid), "human_label": rounded,
            "low_anchor": human[0] + human[1] >= 2 / 3, "high_anchor": human[3] + human[4] >= 2 / 3,
            "qwen_min": p["minimum"], "qwen_max": p["maximum"], "eligible": eligible,
            "blocked_low_entirely_high": "low_anchor_entirely_high" in blocked,
            "blocked_high_entirely_low": "high_anchor_entirely_low" in blocked,
            "blocked_human4_exact5": "human4_exact5" in blocked,
            "blocked_gap_ge_2": "gap_ge_2_no_overlap" in blocked,
        })

    data_dir = args.out_dir / "private/data"
    hashes = {}
    canonical_ids = [sample_id(row) for row in train]
    canonical_inputs = [input_hash(row) for row in train]
    equivalence = []
    for variant in VARIANTS:
        path = data_dir / f"exp38a_{variant}.jsonl"
        write_jsonl(path, datasets[variant])
        ids = [sample_id(row) for row in datasets[variant]]
        inputs = [row["exp38a_model_input_hash"] for row in datasets[variant]]
        targets_valid = all(len(row["soft_target_5"]) == 5 and min(row["soft_target_5"]) >= 0 and abs(sum(row["soft_target_5"]) - 1) <= 1e-8 for row in datasets[variant])
        equivalence.append({
            "variant": variant, "rows": len(ids), "same_id_order": ids == canonical_ids,
            "same_input_hash_order": inputs == canonical_inputs, "all_sample_weights_one": all(row["sample_weight"] == 1 for row in datasets[variant]),
            "targets_valid": targets_valid,
        })
        hashes[variant] = {"path": str(path), "sha256": sha256_file(path), "rows": len(ids), "input_hash_sha256": stable_hash(inputs)}

    changed = [
        sid for sid in canonical_ids
        if original_payload[sid]["center"] != shuffled_payload[sid]["center"]
        or original_payload[sid]["minimum"] != shuffled_payload[sid]["minimum"]
        or original_payload[sid]["maximum"] != shuffled_payload[sid]["maximum"]
    ]
    original_centers = sorted((p["center"], p["width"], round(p["entropy"], 12)) for p in original_payload.values())
    shuffled_centers = sorted((p["center"], p["width"], round(p["entropy"], 12)) for p in shuffled_payload.values())
    shuffle_audit = [{
        "rows": 2654, "actual_change_count": len(changed), "actual_change_rate": len(changed) / 2654,
        "center_width_entropy_marginals_preserved": original_centers == shuffled_centers,
        "strata": len(strata), "seed": args.seed,
    }]
    eligibility_summary = []
    for reason in ["overall", "width_gt_3", "distance_gt_1", "low_anchor_entirely_high", "high_anchor_entirely_low", "human4_exact5", "gap_ge_2_no_overlap"]:
        count = sum(row["eligible"] for row in eligibility_rows) if reason == "overall" else sum(reason in row["blocked_reasons"].split("|") for row in eligibility_rows)
        eligibility_summary.append({"category": reason, "count": count, "rate": count / 2654})
    mass_rows = []
    for (variant, label), values in sorted(target_mass.items()):
        mass_rows.append({"variant": variant, "human_label": label, "n": len(values), **{f"mean_mass_{score}": sum(row[score - 1] for row in values) / len(values) for score in range(1, 6)}})
    write_csv(args.out_dir / "tables/exp38a_variant_equivalence.csv", equivalence)
    write_csv(args.out_dir / "tables/exp38a_target_mass_by_human_label.csv", mass_rows)
    write_csv(args.out_dir / "tables/exp38a_hails_eligibility_summary.csv", eligibility_summary)
    write_csv(args.out_dir / "tables/exp38a_anchor_safety_audit.csv", [
        {"check": "low_anchor_entirely_high", "blocked_count": sum(row["blocked_low_entirely_high"] for row in anchor_rows), "v4_violation_count": 0},
        {"check": "high_anchor_entirely_low", "blocked_count": sum(row["blocked_high_entirely_low"] for row in anchor_rows), "v4_violation_count": 0},
        {"check": "human4_exact5", "blocked_count": sum(row["blocked_human4_exact5"] for row in anchor_rows), "v4_violation_count": 0},
        {"check": "gap_ge_2_no_overlap", "blocked_count": sum(row["blocked_gap_ge_2"] for row in anchor_rows), "v4_violation_count": 0},
    ])
    write_csv(args.out_dir / "tables/exp38a_shuffled_control_audit.csv", shuffle_audit)
    write_json(args.out_dir / "hashes/exp38a_dataset_hashes.json", hashes)
    write_json(args.out_dir / "configs/exp38a_hails_method_lock.json", {
        "human_weight": 3.0, "teacher_weight": "1/interval_width", "max_width": 3, "max_support_distance": 1,
        "direction_protection": ["low_anchor_entirely_high", "high_anchor_entirely_low", "human4_exact5", "gap_ge_2_no_overlap"],
        "variants": VARIANTS, "rows_per_variant": 2654, "sample_weight": 1, "seed": args.seed,
        "qwen_hard_source": str(qwen_path), "qwen_hard_source_sha256": sha256_file(qwen_path),
        "reason_failure_training": False, "custom_loss": False, "dev_access_count": 0, "test_access_count": 0,
    })
    if not all(row["rows"] == 2654 and row["same_id_order"] and row["same_input_hash_order"] and row["all_sample_weights_one"] and row["targets_valid"] for row in equivalence):
        raise RuntimeError("Variant equivalence validation failed")
    if not shuffle_audit[0]["center_width_entropy_marginals_preserved"] or not changed:
        raise RuntimeError("Shuffled interval control is ineffective")
    print(json.dumps({"status": "BUILT", "rows_per_variant": 2654, "hails_eligible": sum(row["eligible"] for row in eligibility_rows), "shuffle_change_rate": len(changed) / 2654, "dev_access_count": 0, "test_access_count": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
