"""Build deterministic train-fold-only TACO triplets and qualification audits."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.exp44_taco_score.common import (
    FOLDS,
    ROOT,
    SEED,
    atomic_json,
    ensure_dirs,
    fold_map,
    ordinal_distribution_distance,
    read_jsonl,
    stable_hash,
    stable_seed,
    stratum,
    triplet_path,
    write_csv,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--partner-cap", type=int, default=8)
    return parser.parse_args()


def ordered_candidates(
    anchor: dict[str, Any],
    rows: list[dict[str, Any]],
    role: str,
    *,
    exclude: set[str],
    fold: int,
    epoch: int,
) -> list[tuple[dict[str, Any], float]]:
    values = []
    anchor_distribution = anchor["human_distribution_5"]
    for candidate in rows:
        sample_id = str(candidate["sample_id"])
        if sample_id in exclude or candidate["question_key"] == anchor["question_key"] and role == "positive":
            continue
        distance = ordinal_distribution_distance(anchor_distribution, candidate["human_distribution_5"])
        if role == "positive" and int(candidate["gold_label_5"]) != int(anchor["gold_label_5"]):
            continue
        if role == "near" and not (0.5 < distance < 2.0):
            continue
        if role == "far" and distance < 2.0:
            continue
        language_penalty = 0 if candidate["language"] == anchor["language"] else 1
        if role == "positive":
            priority = (language_penalty, distance)
        elif role == "near":
            priority = (language_penalty, abs(distance - 1.0))
        else:
            priority = (language_penalty, -distance)
        tie = stable_hash(("exp44a", fold, epoch, role, anchor["sample_id"], sample_id, SEED))
        values.append((priority, tie, candidate, distance))
    values.sort(key=lambda item: (item[0], item[1]))
    return [(item[2], item[3]) for item in values]


def choose_partner(
    candidates: list[tuple[dict[str, Any], float]],
    reuse: Counter[str],
    cap: int,
) -> tuple[dict[str, Any] | None, float | None]:
    for candidate, distance in candidates:
        if reuse[str(candidate["sample_id"])] < cap:
            return candidate, distance
    return None, None


def build_stratum_triplets(
    anchors: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    fold: int,
    epoch: int,
    cap: int,
    reuse: Counter[str],
) -> list[dict[str, Any]]:
    built = []
    for anchor in anchors:
        anchor_id = str(anchor["sample_id"])
        positive_candidates = ordered_candidates(anchor, train_rows, "positive", exclude={anchor_id}, fold=fold, epoch=epoch)
        positive, positive_distance = choose_partner(positive_candidates, reuse, cap)
        positive_type = "distinct_row"
        if positive is None:
            positive = anchor
            positive_distance = 0.0
            positive_type = "self_view"
        positive_id = str(positive["sample_id"])
        near_candidates = ordered_candidates(anchor, train_rows, "near", exclude={anchor_id, positive_id}, fold=fold, epoch=epoch)
        near, near_distance = choose_partner(near_candidates, reuse, cap)
        if near is None:
            continue
        near_id = str(near["sample_id"])
        far_candidates = ordered_candidates(anchor, train_rows, "far", exclude={anchor_id, positive_id, near_id}, fold=fold, epoch=epoch)
        far, far_distance = choose_partner(far_candidates, reuse, cap)
        if far is None:
            continue
        far_id = str(far["sample_id"])
        if positive_type == "distinct_row":
            reuse[positive_id] += 1
        reuse[near_id] += 1
        reuse[far_id] += 1
        built.append(
            {
                "triplet_id": stable_hash(("exp44a", fold, epoch, anchor_id, positive_id, near_id, far_id)),
                "fold": fold,
                "epoch": epoch,
                "anchor_id": anchor_id,
                "anchor_label": int(anchor["gold_label_5"]),
                "anchor_stratum": stratum(int(anchor["gold_label_5"])),
                "positive_id": positive_id,
                "positive_type": positive_type,
                "positive_distance": float(positive_distance),
                "near_id": near_id,
                "near_distance": float(near_distance),
                "far_id": far_id,
                "far_distance": float(far_distance),
                "anchor_question_key": anchor["question_key"],
                "positive_question_key": positive["question_key"],
                "near_question_key": near["question_key"],
                "far_question_key": far["question_key"],
            }
        )
    return built


def shuffle_margins(rows: list[dict[str, Any]], fold: int, epoch: int) -> float:
    changed = 0
    total = 0
    for group_name in ("low", "mid", "high"):
        indices = [index for index, row in enumerate(rows) if row["anchor_stratum"] == group_name]
        original_near = [rows[index]["near_distance"] for index in indices]
        original_far = [rows[index]["far_distance"] for index in indices]
        if len(original_near) < 2:
            for index in indices:
                rows[index]["shuffled_near_distance"] = rows[index]["near_distance"]
                rows[index]["shuffled_far_distance"] = rows[index]["far_distance"]
            continue
        rng = np.random.default_rng(stable_seed("exp44a-margin", fold, epoch, group_name, SEED))
        best_near = best_far = list(range(len(original_near)))
        best_changes = -1
        # Preserve each complete near/far margin distribution while avoiding
        # an impossible derangement when repeated (near, far) tuples dominate.
        for _ in range(512):
            near_permutation = rng.permutation(len(original_near)).tolist()
            far_permutation = rng.permutation(len(original_far)).tolist()
            changes = sum(
                original_near[index] != original_near[near_permutation[index]]
                or original_far[index] != original_far[far_permutation[index]]
                for index in range(len(original_near))
            )
            if changes > best_changes:
                best_near, best_far, best_changes = near_permutation, far_permutation, changes
        for local_index, row_index in enumerate(indices):
            near = original_near[best_near[local_index]]
            far = original_far[best_far[local_index]]
            rows[row_index]["shuffled_near_distance"] = near
            rows[row_index]["shuffled_far_distance"] = far
            changed += int(original_near[local_index] != near or original_far[local_index] != far)
            total += 1
    return changed / total if total else 0.0


def main() -> None:
    args = parse_args()
    ensure_dirs(args.out_dir)
    rows = read_jsonl(args.out_dir / "private/data/exp44a_train_e4.jsonl")
    folds = fold_map(args.out_dir / "private/data/exp44a_groupcv_fold_assignment.csv")
    pool_rows: list[dict[str, Any]] = []
    distance_rows: list[dict[str, Any]] = []
    anchor_rows: list[dict[str, Any]] = []
    reuse_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    all_go = True
    for fold in FOLDS:
        train_rows = [row for row in rows if folds[row["sample_id"]] != fold]
        heldout_ids = {row["sample_id"] for row in rows if folds[row["sample_id"]] == fold}
        low_all = [row for row in train_rows if int(row["gold_label_5"]) <= 2]
        mid_all = [row for row in train_rows if int(row["gold_label_5"]) == 3]
        high_all = [row for row in train_rows if int(row["gold_label_5"]) >= 4]
        for epoch in range(1, args.epochs + 1):
            count = len(low_all)
            low = sorted(low_all, key=lambda row: stable_hash(("low", fold, epoch, SEED, row["sample_id"])))
            mid = sorted(mid_all, key=lambda row: stable_hash(("mid", fold, epoch, SEED, row["sample_id"])))[:count]
            high = sorted(high_all, key=lambda row: stable_hash(("high", fold, epoch, SEED, row["sample_id"])))[:count]
            reuse: Counter[str] = Counter()
            by_stratum = {
                name: build_stratum_triplets(selected, train_rows, fold, epoch, args.partner_cap, reuse)
                for name, selected in (("low", low), ("mid", mid), ("high", high))
            }
            equal_count = min(len(value) for value in by_stratum.values())
            triplets = []
            for name in ("low", "mid", "high"):
                selected = sorted(by_stratum[name], key=lambda row: stable_hash((row["triplet_id"], "equalize")))[:equal_count]
                triplets.extend(selected)
                anchor_rows.append({"fold": fold, "epoch": epoch, "stratum": name, "available": count, "valid_before_equalize": len(by_stratum[name]), "selected": len(selected)})
            triplets.sort(key=lambda row: stable_hash(("order", fold, epoch, row["triplet_id"])))
            shuffle_rate = shuffle_margins(triplets, fold, epoch)
            write_jsonl(triplet_path(args.out_dir, fold, epoch), triplets)
            endpoints = {
                endpoint
                for row in triplets
                for endpoint in (row["anchor_id"], row["positive_id"], row["near_id"], row["far_id"])
            }
            leakage = len(endpoints & heldout_ids)
            low_coverage = equal_count / count if count else 0.0
            cap_observed = max(reuse.values(), default=0)
            near_count = sum(0.5 < float(row["near_distance"]) < 2.0 for row in triplets)
            far_count = sum(float(row["far_distance"]) >= 2.0 for row in triplets)
            epoch_go = (
                low_coverage >= 0.90
                and equal_count > 0
                and near_count == len(triplets)
                and far_count == len(triplets)
                and leakage == 0
                and cap_observed <= args.partner_cap
                and shuffle_rate >= 0.80
            )
            all_go &= epoch_go
            pool_rows.append({"fold": fold, "epoch": epoch, "train_rows": len(train_rows), "low_available": count, "triplets": len(triplets), "per_stratum": equal_count, "low_anchor_coverage": low_coverage, "near_count": near_count, "far_count": far_count, "shuffle_assignment_change_rate": shuffle_rate, "status": "GO" if epoch_go else "NO_GO"})
            leakage_rows.append({"fold": fold, "epoch": epoch, "heldout_endpoint_overlap": leakage, "question_key_train_heldout_overlap": len({row["question_key"] for row in train_rows} & {row["question_key"] for row in rows if folds[row["sample_id"]] == fold}), "status": "PASS" if leakage == 0 else "FAIL"})
            for role in ("positive", "near", "far"):
                values = [float(row[f"{role}_distance"]) for row in triplets]
                distance_rows.append({"fold": fold, "epoch": epoch, "role": role, "n": len(values), "mean": float(np.mean(values)), "p50": float(np.quantile(values, 0.5)), "p90": float(np.quantile(values, 0.9)), "min": min(values), "max": max(values)})
            for sample_id, uses in sorted(reuse.items()):
                reuse_rows.append({"fold": fold, "epoch": epoch, "partner_id_hash": stable_hash(sample_id), "reuse_count": uses, "cap": args.partner_cap, "within_cap": uses <= args.partner_cap})
    write_csv(args.out_dir / "tables/exp44a_triplet_pool_summary.csv", pool_rows)
    write_csv(args.out_dir / "tables/exp44a_triplet_distance_distribution.csv", distance_rows)
    write_csv(args.out_dir / "tables/exp44a_anchor_class_distribution.csv", anchor_rows)
    write_csv(args.out_dir / "tables/exp44a_triplet_partner_reuse.csv", reuse_rows)
    write_csv(args.out_dir / "tables/exp44a_triplet_leakage_audit.csv", leakage_rows)
    decision = {
        "status": "TRIPLET_DATA_GO" if all_go else "TRIPLET_DATA_NO_GO",
        "fold_epochs": len(pool_rows),
        "minimum_low_anchor_coverage": min(float(row["low_anchor_coverage"]) for row in pool_rows),
        "minimum_margin_shuffle_rate": min(float(row["shuffle_assignment_change_rate"]) for row in pool_rows),
        "maximum_partner_reuse": max(int(row["reuse_count"]) for row in reuse_rows),
        "endpoint_leakage": sum(int(row["heldout_endpoint_overlap"]) for row in leakage_rows),
        "dev_access_count": 0,
        "test_access_count": 0,
    }
    atomic_json(args.out_dir / "decision/exp44a_data_decision.json", decision)
    report = [
        "# Exp44A Data Preparation",
        "",
        f"Decision: **{decision['status']}**",
        "",
        f"- Fold-epochs: {decision['fold_epochs']}",
        f"- Minimum low-anchor coverage: {decision['minimum_low_anchor_coverage']:.4f}",
        f"- Minimum shuffled-margin reassignment: {decision['minimum_margin_shuffle_rate']:.4f}",
        f"- Maximum partner reuse: {decision['maximum_partner_reuse']}",
        f"- Held-out endpoint leakage: {decision['endpoint_leakage']}",
        "- Dev/test access: 0/0",
    ]
    (args.out_dir / "reports/exp44a_data_prepare_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(decision, sort_keys=True))
    if not all_go:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
