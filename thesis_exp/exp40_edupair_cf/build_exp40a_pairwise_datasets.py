"""Build the four matched-count Exp40A pair variants after pairwise GO."""

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

from thesis_exp.exp40_edupair_cf.common import (  # noqa: E402
    PAIR_VARIANTS,
    ROOT,
    TRAIN_PATH,
    human_distribution,
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


def expected_human_score(row: dict[str, Any]) -> float:
    distribution = human_distribution(row)
    return sum((index + 1) * value for index, value in enumerate(distribution))


def pair_payload(
    pair_id: str, question_key: str, better: dict[str, Any], worse: dict[str, Any],
    pair_type: str, **metadata: Any,
) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "question_key": question_key,
        "pair_type": pair_type,
        "better": better,
        "worse": worse,
        **metadata,
    }


def model_row(source: dict[str, Any], answer: str, synthetic_id: str) -> dict[str, Any]:
    return {
        **source,
        "sample_id": synthetic_id,
        "answer": answer,
    }


def build_human_pairs(train: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in train:
        groups[(str(row["question_key"]), str(row.get("metric_canonical") or row.get("metric_raw") or row.get("metric_id")))].append(row)
    candidates = []
    for (qkey, metric), rows in groups.items():
        scores = {sample_id(row): expected_human_score(row) for row in rows}
        for left_index, left in enumerate(rows):
            for right in rows[left_index + 1 :]:
                gap = scores[sample_id(left)] - scores[sample_id(right)]
                if abs(gap) < 1.0:
                    continue
                better, worse = (left, right) if gap > 0 else (right, left)
                candidates.append((qkey, metric, better, worse, abs(gap)))
    candidates.sort(key=lambda row: stable_hash({"seed": seed, "better": sample_id(row[2]), "worse": sample_id(row[3])}))
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} same-question/same-metric human pairs satisfy gap>=1; need {count}")
    selected = candidates[:count]
    return [
        pair_payload(
            "exp40a_v1_" + stable_hash({"better": sample_id(better), "worse": sample_id(worse), "seed": seed})[:24],
            qkey, better, worse, "human_real_pair", human_expected_score_gap=gap, metric=metric,
        )
        for qkey, metric, better, worse, gap in selected
    ]


def resolved_pair_to_payload(resolved: dict[str, Any], source: dict[str, Any], prefix: str, pair_type: str) -> dict[str, Any]:
    worse = model_row(
        source,
        str(resolved["counterfactual_answer"]),
        prefix + "_cf_" + stable_hash({"pair": resolved["pair_id"]})[:20],
    )
    return pair_payload(
        prefix + "_" + stable_hash({"pair": resolved["pair_id"]})[:24],
        str(source["question_key"]), source, worse, pair_type,
        exp40a_source_pair_id=resolved["pair_id"], operator=resolved.get("operator"),
        language=resolved.get("language"), metric_group=resolved.get("metric_group"),
    )


def deranged_donors(rows: list[dict[str, Any]], seed: int) -> tuple[dict[str, dict[str, Any]], float]:
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[(str(row.get("language", "unknown")), str(row.get("metric_group", "unknown")))].append(row)
    mapping: dict[str, dict[str, Any]] = {}
    changed = 0
    for stratum in sorted(strata):
        members = sorted(strata[stratum], key=lambda row: stable_hash({"seed": seed, "pair": row["pair_id"]}))
        donors = members[1:] + members[:1] if len(members) > 1 else members
        for recipient, donor in zip(members, donors):
            mapping[recipient["pair_id"]] = donor
            changed += donor["pair_id"] != recipient["pair_id"]
    return mapping, changed / max(len(rows), 1)


def main() -> None:
    args = parse_args()
    decision = json.loads((args.out_dir / "decision/exp40a_pairwise_qualification_decision.json").read_text(encoding="utf-8"))
    if not decision.get("recommend_groupcv_training"):
        raise RuntimeError("Pair dataset construction is blocked by Exp40A qualification NO-GO")
    train = read_jsonl(args.train_jsonl)
    resolved = read_jsonl(args.out_dir / "private/resolved_pairs/exp40a_resolved_exp39a_pairs.jsonl")
    accepted = read_jsonl(args.out_dir / "private/accepted_pairs/exp40a_accepted_pairs.jsonl")
    if len(train) != 2654 or not accepted:
        raise ValueError("Exp40A requires 2654 original rows and a non-empty accepted pair pool")
    source_by_id = {sample_id(row): row for row in train}
    resolved_by_id = {row["pair_id"]: row for row in resolved}
    pair_count = len(accepted)
    if pair_count < 150:
        raise RuntimeError("Qualification decision and accepted pair count disagree")

    v1 = build_human_pairs(train, pair_count, args.seed)
    unverified = sorted(resolved, key=lambda row: stable_hash({"seed": args.seed, "pair": row["pair_id"]}))[:pair_count]
    v2 = [resolved_pair_to_payload(row, source_by_id[row["sample_id"]], "exp40a_v2", "unverified_counterfactual") for row in unverified]
    v3 = [resolved_pair_to_payload(row, source_by_id[row["sample_id"]], "exp40a_v3", "verified_edupair_cf") for row in accepted]

    donor_map, change_rate = deranged_donors(accepted, args.seed)
    v4 = []
    for recipient in accepted:
        donor = donor_map[recipient["pair_id"]]
        source = source_by_id[recipient["sample_id"]]
        worse = model_row(
            source,
            str(donor["counterfactual_answer"]),
            "exp40a_v4_cf_" + stable_hash({"recipient": recipient["pair_id"], "donor": donor["pair_id"]})[:20],
        )
        v4.append(
            pair_payload(
                "exp40a_v4_" + stable_hash({"recipient": recipient["pair_id"], "donor": donor["pair_id"]})[:24],
                str(source["question_key"]), source, worse, "shuffled_pair_alignment",
                recipient_pair_id=recipient["pair_id"], donor_pair_id=donor["pair_id"],
                shuffled_pair_changed=recipient["pair_id"] != donor["pair_id"],
                language=recipient.get("language"), metric_group=recipient.get("metric_group"),
            )
        )
    if change_rate < 0.80:
        raise RuntimeError(f"Shuffled actual-pair-change rate {change_rate:.4f} is below 0.80")

    datasets = {
        "v1_human_real_pairs": v1,
        "v2_unverified_counterfactual_pairs": v2,
        "v3_edupair_cf": v3,
        "v4_shuffled_pair_alignment": v4,
    }
    if {len(rows) for rows in datasets.values()} != {pair_count}:
        raise RuntimeError("All Exp40A pair variants must use identical pair counts")
    data_dir = args.out_dir / "private/data"
    hashes = {}
    summary = [{"variant": "v0h_human_soft", "pair_count": 0, "unique_question_keys": 0, "pair_type": "none"}]
    for variant in PAIR_VARIANTS:
        path = data_dir / f"exp40a_{variant}.jsonl"
        write_jsonl(path, datasets[variant])
        summary.append(
            {
                "variant": variant,
                "pair_count": len(datasets[variant]),
                "unique_question_keys": len({row["question_key"] for row in datasets[variant]}),
                "pair_type": datasets[variant][0]["pair_type"],
            }
        )
        hashes[variant] = {"path": str(path), "rows": len(datasets[variant]), "sha256": sha256_file(path)}
    write_csv(args.out_dir / "tables/exp40a_pair_variant_summary.csv", summary)
    write_csv(
        args.out_dir / "tables/exp40a_shuffled_pair_audit.csv",
        [
            {
                "rows": len(v4), "actual_pair_change_rate": change_rate,
                "changed_rows": sum(row["shuffled_pair_changed"] for row in v4),
                "language_metric_strata": len({(row.get("language"), row.get("metric_group")) for row in v4}),
                "pair_count_matches_v3": len(v4) == len(v3),
            }
        ],
    )
    lock = {
        "experiment": "Exp40A EduPair-CF GroupCV",
        "variants": ["v0h_human_soft", *PAIR_VARIANTS],
        "original_train_rows": 2654,
        "pair_count_rule": "V1/V2/V3/V4 each match the accepted V3 pair count after qualification GO",
        "loss": "human_empirical_soft_CE + 0.25 * RankNet_pair_loss",
        "lambda_pair": 0.25,
        "epochs": 10,
        "learning_rate": 2e-5,
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "warmup_ratio": 0.05,
        "micro_batch": 4,
        "gradient_accumulation": 32,
        "effective_batch": 128,
        "max_length": 2048,
        "seed": 42,
        "sample_weighting": False,
        "class_weighting": False,
        "custom_risk_or_ordinal_loss": False,
        "auxiliary_head": False,
        "fixed_final_epoch": 10,
        "heldout_checkpoint_selection": False,
        "dev_access_count": 0,
        "test_access_count": 0,
    }
    lock_path = args.out_dir / "configs/exp40a_training_matrix_lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text(encoding="utf-8")) != lock:
        raise RuntimeError(f"Frozen Exp40A training matrix changed: {lock_path}")
    write_json(lock_path, lock)
    write_json(
        args.out_dir / "hashes/exp40a_training_config_hashes.json",
        {"pair_datasets": hashes, "accepted_pair_count": pair_count, "training_lock": lock},
    )
    print(json.dumps({"status": "BUILT", "pair_count": pair_count, "shuffled_change_rate": change_rate}, sort_keys=True))


if __name__ == "__main__":
    main()
