"""Build deterministic train-only RubiMOR pair pools and fold subsets."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from thesis_exp.exp43_rubimor.common import ROOT, TRAIN_PATH, canonical_metric, human_stats, read_jsonl, sample_id, stable_hash, write_csv, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    return parser.parse_args()


def select_pairs(group: list[dict[str, Any]]) -> list[tuple[dict, dict, float, str]]:
    eligible = []
    for left_index, left in enumerate(group):
        for right in group[left_index + 1 :]:
            gap = abs(float(left["mu"]) - float(right["mu"]))
            if gap < 1.0:
                continue
            positive, negative = (left, right) if left["mu"] > right["mu"] else (right, left)
            eligible.append((positive, negative, gap, "near" if gap < 2 else "far"))
    eligible.sort(key=lambda item: (stable_hash((item[0]["sample_id"], item[1]["sample_id"], 42)), item[0]["sample_id"], item[1]["sample_id"]))
    if len(group) <= 6:
        return eligible
    selected, degree = [], Counter()
    targets = {"near": 7, "far": 5}
    for gap_bin in ("near", "far"):
        for item in [value for value in eligible if value[3] == gap_bin]:
            a, b = item[0]["sample_id"], item[1]["sample_id"]
            if degree[a] >= 4 or degree[b] >= 4 or sum(value[3] == gap_bin for value in selected) >= targets[gap_bin]:
                continue
            selected.append(item)
            degree[a] += 1
            degree[b] += 1
    return selected[:12]


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.train)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    mismatch_candidates = mismatch_excluded = 0
    for row in rows:
        stats = human_stats(row)
        groups[(str(row["question_key"]), canonical_metric(row))].append({
            "sample_id": sample_id(row), "question_key": str(row["question_key"]), "metric": canonical_metric(row),
            "mu": stats["expected_human_score"], "language": row.get("language") or "unknown",
            "subject": row.get("subject_canonical") or row.get("subject_raw") or "unknown",
            "scenario": row.get("scenario_canonical") or row.get("scenario_raw") or "unknown",
        })
    pairs = []
    for (qkey, metric), group in sorted(groups.items()):
        metadata_groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        for row in group:
            metadata_groups[(row["language"], row["subject"], row["scenario"])].append(row)
        if len(metadata_groups) > 1:
            mismatch_candidates += sum(len(part) for part in metadata_groups.values())
        for metadata_key, part in metadata_groups.items():
            for positive, negative, gap, gap_bin in select_pairs(part):
                pair_id = stable_hash((positive["sample_id"], negative["sample_id"], qkey, metric))
                pairs.append({"pair_id": pair_id, "question_key": qkey, "metric": metric, "positive_id": positive["sample_id"], "negative_id": negative["sample_id"], "positive_mu": positive["mu"], "negative_mu": negative["mu"], "gap": gap, "gap_bin": gap_bin, "language": metadata_key[0], "subject": metadata_key[1], "scenario": metadata_key[2]})
    # Count cross-metadata eligible pairs that were intentionally excluded.
    for group in groups.values():
        for i, left in enumerate(group):
            for right in group[i + 1 :]:
                if abs(left["mu"] - right["mu"]) >= 1 and (left["language"], left["subject"], left["scenario"]) != (right["language"], right["subject"], right["scenario"]):
                    mismatch_excluded += 1
    mismatch_rate = mismatch_excluded / max(mismatch_excluded + len(pairs), 1)
    write_jsonl(args.out_dir / "private/pairs/exp43_train_pairs.jsonl", pairs)
    with (args.out_dir / "private/data/exp43_groupcv_fold_assignment.csv").open("r", encoding="utf-8", newline="") as handle:
        fold_by_id = {row["sample_id"]: int(row["fold"]) for row in csv.DictReader(handle)}
    fold_summary = []
    for fold in range(5):
        subset = [pair for pair in pairs if fold_by_id[pair["positive_id"]] != fold and fold_by_id[pair["negative_id"]] != fold]
        write_jsonl(args.out_dir / f"private/pairs/exp43_train_pairs_fold{fold}.jsonl", subset)
        fold_summary.append({"fold": fold, "pairs": len(subset), "near": sum(p["gap_bin"] == "near" for p in subset), "far": sum(p["gap_bin"] == "far" for p in subset), "endpoint_fold_leakage": 0})
    gaps = Counter((pair["gap_bin"], f"{pair['gap']:.6f}") for pair in pairs)
    write_csv(args.out_dir / "tables/exp43_pair_pool_summary.csv", [{"scope": "all_train", "pairs": len(pairs), "groups": len({(p['question_key'], p['metric']) for p in pairs}), "near": sum(p["gap_bin"] == "near" for p in pairs), "far": sum(p["gap_bin"] == "far" for p in pairs), "metadata_mismatch_excluded": mismatch_excluded, "mismatch_rate": mismatch_rate}, *fold_summary])
    write_csv(args.out_dir / "tables/exp43_pair_gap_distribution.csv", [{"gap_bin": key[0], "gap": key[1], "count": count} for key, count in sorted(gaps.items())])
    write_csv(args.out_dir / "tables/exp43_pair_leakage_audit.csv", fold_summary)
    status = "GO" if mismatch_rate <= 0.01 else "PAIR_DATA_NO_GO"
    write_json(args.out_dir / "configs/exp43_pair_protocol_lock.json", {"same_question_key_metric": True, "minimum_expected_score_gap": 1.0, "small_group_all_pairs_max_size": 6, "large_group_max_degree": 4, "large_group_max_pairs": 12, "near_target": 7, "far_target": 5, "stable_seed": 42, "e6n_flip_seed": 4242, "e6n_flip_range": [0.48, 0.52]})
    write_json(args.out_dir / "decision/exp43_pair_data_decision.json", {"status": status, "pairs": len(pairs), "metadata_mismatch_rate": mismatch_rate, "test_access_count": 0})
    print(json.dumps({"status": status, "pairs": len(pairs), "metadata_mismatch_rate": mismatch_rate}, sort_keys=True))


if __name__ == "__main__":
    main()

