"""Prepare QD-PR2 anchored pairwise ordinal scaffold artifacts."""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp09_pairwise_ordinal import (
    DEFAULT_GAP_WEIGHT,
    DEFAULT_LOW_HIGH_MARGIN,
    DEFAULT_LOW_HIGH_WEIGHT,
    DEFAULT_MARGIN_SCALE,
    DEFAULT_PAIR_SAMPLING_SEED,
    PAIR_TYPE_PROPORTIONS,
    QDPR2_DATASET_DIR,
    QDPR2_DEFAULT_DEV_PAIR_COUNT,
    QDPR2_DEFAULT_MAX_PAIRS_PER_LOW_RECORD,
    QDPR2_DEFAULT_MAX_PAIRS_PER_RECORD,
    QDPR2_DEFAULT_TRAIN_PAIR_COUNT,
    QDPR2_PAIRS_DIR,
    QDPR2_REPORTS_DIR,
    QDPR2_RUN_ID,
    QDPR2_TABLES_DIR,
    ensure_exp09_dirs,
)
from thesis_exp.src.edujudge.exp09_pairwise_ordinal.data import read_split, record_key
from thesis_exp.src.edujudge.exp09_pairwise_ordinal.losses import pair_margin, pair_weight
from thesis_exp.src.edujudge.exp09_pairwise_ordinal.pair_builder import pair_type, write_pair_shards
from thesis_exp.src.edujudge.utils.io import read_jsonl, relpath, write_csv, write_text


HIGH_COMPARABILITY_PRIORITIES = ["same_question", "same_metric_language"]
FALLBACK_PRIORITIES = ["same_metric", "any_valid"]
PAIR_INVENTORY_FIELDS = [
    "split",
    "pair_count",
    "target_count",
    "sampling_seed",
    "max_pairs_per_record",
    "max_pairs_per_low_record",
    "write_shards",
    "pair_shard_count",
]
PAIR_DISTRIBUTION_FIELDS = [
    "split",
    "pair_type",
    "target_count",
    "actual_count",
    "high_comparability_count",
    "fallback_same_metric_count",
    "fallback_any_valid_count",
    "shortage",
]
PAIR_COMPARABILITY_FIELDS = [
    "split",
    "pair_type",
    "pair_count",
    "same_question_count",
    "same_question_rate",
    "same_metric_count",
    "same_metric_rate",
    "same_language_count",
    "same_language_rate",
    "same_metric_language_count",
    "same_metric_language_rate",
    "fallback_count",
    "fallback_rate",
]
PAIR_REUSE_FIELDS = [
    "split",
    "records_used",
    "max_reuse",
    "mean_reuse",
    "low_records_used",
    "max_reuse_low_record",
    "mean_reuse_low_record",
    "low_record_reuse_limit",
]


@dataclass(frozen=True)
class QDPR2PairBuildConfig:
    train_pairs: int = QDPR2_DEFAULT_TRAIN_PAIR_COUNT
    dev_pairs: int = QDPR2_DEFAULT_DEV_PAIR_COUNT
    seed: int = DEFAULT_PAIR_SAMPLING_SEED
    max_pairs_per_record: int = QDPR2_DEFAULT_MAX_PAIRS_PER_RECORD
    max_pairs_per_low_record: int = QDPR2_DEFAULT_MAX_PAIRS_PER_LOW_RECORD
    margin_scale: float = DEFAULT_MARGIN_SCALE
    low_high_margin: float = DEFAULT_LOW_HIGH_MARGIN
    low_high_weight: float = DEFAULT_LOW_HIGH_WEIGHT
    gap_weight: float = DEFAULT_GAP_WEIGHT
    pair_type_proportions: dict[str, float] | None = None

    def proportions(self) -> dict[str, float]:
        return dict(self.pair_type_proportions or PAIR_TYPE_PROPORTIONS)


def priority(row_a: dict[str, Any], row_b: dict[str, Any]) -> str:
    same_question = question_identity(row_a) == question_identity(row_b)
    same_metric = row_a.get("metric_canonical") == row_b.get("metric_canonical")
    same_language = row_a.get("language") == row_b.get("language")
    if same_question:
        return "same_question"
    if same_metric and same_language:
        return "same_metric_language"
    if same_metric:
        return "same_metric"
    return "any_valid"


def question_identity(row: dict[str, Any]) -> str:
    return str(row.get("question_key") or row.get("question_id") or row.get("question") or "")


def target_counts(total: int, proportions: dict[str, float]) -> dict[str, int]:
    counts = {name: int(round(total * float(proportions[name]))) for name in proportions}
    delta = total - sum(counts.values())
    order = list(proportions)
    idx = 0
    while delta != 0:
        key = order[idx % len(order)]
        counts[key] += 1 if delta > 0 else -1
        delta += -1 if delta > 0 else 1
        idx += 1
    return counts


def candidate_pairs(rows: list[dict[str, Any]]) -> dict[str, dict[str, list[tuple[int, int]]]]:
    priorities = HIGH_COMPARABILITY_PRIORITIES + FALLBACK_PRIORITIES
    buckets: dict[str, dict[str, list[tuple[int, int]]]] = {
        pair_name: {name: [] for name in priorities} for pair_name in PAIR_TYPE_PROPORTIONS
    }
    for idx_a, row_a in enumerate(rows):
        label_a = int(row_a["label_5"])
        for idx_b in range(idx_a + 1, len(rows)):
            row_b = rows[idx_b]
            label_b = int(row_b["label_5"])
            if label_a == label_b:
                continue
            if label_a > label_b:
                win_idx, lose_idx = idx_a, idx_b
                win_label, lose_label = label_a, label_b
            else:
                win_idx, lose_idx = idx_b, idx_a
                win_label, lose_label = label_b, label_a
            buckets[pair_type(win_label, lose_label)][priority(rows[win_idx], rows[lose_idx])].append((win_idx, lose_idx))
    return buckets


def make_pair_row(
    split: str,
    pair_idx: int,
    win: dict[str, Any],
    lose: dict[str, Any],
    config: QDPR2PairBuildConfig,
) -> dict[str, Any]:
    win_label = int(win["label_5"])
    lose_label = int(lose["label_5"])
    gap = win_label - lose_label
    low_high = int(lose_label <= 2 and win_label >= 4)
    same_question = question_identity(win) == question_identity(lose)
    same_metric = win.get("metric_canonical") == lose.get("metric_canonical")
    same_language = win.get("language") == lose.get("language")
    margin = pair_margin([gap], [low_high], config.margin_scale, config.low_high_margin)[0]
    weight = pair_weight([gap], [low_high], config.low_high_weight, config.gap_weight)[0]
    return {
        "pair_id": f"qdpr2_{split}_pair_{pair_idx:06d}",
        "win_record_id": record_key(win),
        "lose_record_id": record_key(lose),
        "win_text": win["text"],
        "lose_text": lose["text"],
        "win_label_5": win_label,
        "lose_label_5": lose_label,
        "label_gap": gap,
        "pair_type": pair_type(win_label, lose_label),
        "priority": priority(win, lose),
        "metric_canonical": win.get("metric_canonical") if same_metric else f"{win.get('metric_canonical')} || {lose.get('metric_canonical')}",
        "language": win.get("language") if same_language else f"{win.get('language')} || {lose.get('language')}",
        "same_question_key": same_question,
        "same_metric": same_metric,
        "same_language": same_language,
        "risk_pair_low_high": bool(low_high),
        "pair_weight": float(margin * 0.0 + weight),
        "pair_margin": float(margin),
        "source": "question_seed42_human_only",
    }


def build_pairs_for_split(
    rows: list[dict[str, Any]],
    split: str,
    target_count: int,
    config: QDPR2PairBuildConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(config.seed + (0 if split == "train" else 100_000))
    candidates = candidate_pairs(rows)
    targets = target_counts(target_count, config.proportions())
    reuse: Counter[str] = Counter()
    selected: set[tuple[int, int]] = set()
    pairs: list[dict[str, Any]] = []
    distribution: list[dict[str, Any]] = []

    def can_use(win_idx: int, lose_idx: int) -> bool:
        win = rows[win_idx]
        lose = rows[lose_idx]
        lose_label = int(lose["label_5"])
        win_id = record_key(win)
        lose_id = record_key(lose)
        if reuse[win_id] >= config.max_pairs_per_record:
            return False
        limit = config.max_pairs_per_low_record if lose_label <= 2 else config.max_pairs_per_record
        return reuse[lose_id] < limit

    def add_pair(win_idx: int, lose_idx: int) -> None:
        selected.add((win_idx, lose_idx))
        reuse[record_key(rows[win_idx])] += 1
        reuse[record_key(rows[lose_idx])] += 1
        pairs.append(make_pair_row(split, len(pairs), rows[win_idx], rows[lose_idx], config))

    for bucket, bucket_target in targets.items():
        before = len(pairs)
        counts_by_priority: Counter[str] = Counter()
        for priority_name in HIGH_COMPARABILITY_PRIORITIES + FALLBACK_PRIORITIES:
            pool = [item for item in candidates[bucket][priority_name] if item not in selected]
            rng.shuffle(pool)
            for win_idx, lose_idx in pool:
                if len(pairs) - before >= bucket_target:
                    break
                if priority_name in FALLBACK_PRIORITIES and len(pairs) - before >= bucket_target:
                    break
                if can_use(win_idx, lose_idx):
                    add_pair(win_idx, lose_idx)
                    counts_by_priority[priority_name] += 1
            if len(pairs) - before >= bucket_target:
                break
        actual = len(pairs) - before
        distribution.append(
            {
                "split": split,
                "pair_type": bucket,
                "target_count": bucket_target,
                "actual_count": actual,
                "high_comparability_count": counts_by_priority["same_question"] + counts_by_priority["same_metric_language"],
                "fallback_same_metric_count": counts_by_priority["same_metric"],
                "fallback_any_valid_count": counts_by_priority["any_valid"],
                "shortage": max(0, bucket_target - actual),
            }
        )
    return pairs[:target_count], distribution


def pair_comparability_rows(pairs: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for type_name in PAIR_TYPE_PROPORTIONS:
        subset = [pair for pair in pairs if str(pair.get("pair_type")) == type_name]
        total = len(subset)
        same_question = sum(1 for pair in subset if bool(pair.get("same_question_key")))
        same_metric = sum(1 for pair in subset if bool(pair.get("same_metric")))
        same_language = sum(1 for pair in subset if bool(pair.get("same_language")))
        same_metric_language = sum(1 for pair in subset if bool(pair.get("same_metric")) and bool(pair.get("same_language")))
        fallback = sum(1 for pair in subset if str(pair.get("priority")) in FALLBACK_PRIORITIES)
        rows.append(
            {
                "split": split,
                "pair_type": type_name,
                "pair_count": total,
                "same_question_count": same_question,
                "same_question_rate": same_question / total if total else 0.0,
                "same_metric_count": same_metric,
                "same_metric_rate": same_metric / total if total else 0.0,
                "same_language_count": same_language,
                "same_language_rate": same_language / total if total else 0.0,
                "same_metric_language_count": same_metric_language,
                "same_metric_language_rate": same_metric_language / total if total else 0.0,
                "fallback_count": fallback,
                "fallback_rate": fallback / total if total else 0.0,
            }
        )
    return rows


def reuse_rows(pairs: list[dict[str, Any]], split: str, config: QDPR2PairBuildConfig) -> list[dict[str, Any]]:
    reuse: Counter[str] = Counter()
    low_reuse: Counter[str] = Counter()
    for pair in pairs:
        reuse[str(pair["win_record_id"])] += 1
        reuse[str(pair["lose_record_id"])] += 1
        if int(pair["lose_label_5"]) <= 2:
            low_reuse[str(pair["lose_record_id"])] += 1
    values = list(reuse.values())
    low_values = list(low_reuse.values())
    return [
        {
            "split": split,
            "records_used": len(values),
            "max_reuse": max(values) if values else 0,
            "mean_reuse": sum(values) / len(values) if values else 0.0,
            "low_records_used": len(low_values),
            "max_reuse_low_record": max(low_values) if low_values else 0,
            "mean_reuse_low_record": sum(low_values) / len(low_values) if low_values else 0.0,
            "low_record_reuse_limit": config.max_pairs_per_low_record,
        }
    ]


def write_qdpr2_pair_shards(split: str, pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return write_pair_shards(split, pairs, pairs_dir=QDPR2_PAIRS_DIR, shard_rows=2_500)


def read_qdpr2_pair_shards(split: str) -> list[dict[str, Any]]:
    paths = sorted(QDPR2_PAIRS_DIR.glob(f"{split}_pairs_shard_*.jsonl"))
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return rows


def write_reports(
    inventory_rows: list[dict[str, Any]],
    distribution_rows: list[dict[str, Any]],
    comparability: list[dict[str, Any]],
    reuse: list[dict[str, Any]],
) -> None:
    train_inventory = next(row for row in inventory_rows if row["split"] == "train")
    dev_inventory = next(row for row in inventory_rows if row["split"] == "dev")
    train_reuse = next(row for row in reuse if row["split"] == "train")
    lines = [
        "# QD-PR2 Anchored Pairwise Ordinal Fine-tuning Plan",
        "",
        f"Run ID: `{QDPR2_RUN_ID}`",
        "Formal training status: `not_started`.",
        "Training executed by scaffold stage: `no`.",
        "API called: `no`.",
        "Synthetic generated: `no`.",
        "",
        "## Motivation",
        "",
        "QD-PR1 is a negative result: it learned dev pairwise separation but damaged pointwise ordinal "
        "calibration and raised monotonic violation. QD-PR2 therefore uses QD-B1 as an anchor and performs "
        "small-weight pairwise fine-tuning instead of from-scratch training.",
        "",
        "## Loss",
        "",
        "`L_total = L_point + lambda_pair * L_pair + lambda_anchor * L_anchor + lambda_mono * L_mono`",
        "",
        "- `L_point`: QD-B1-style weighted ordinal BCE.",
        "- `L_pair`: `w_ij * softplus(m_ij - (r(win)-r(lose)))`, with `r(x)=1+sum_t sigmoid(z_t)`.",
        "- `L_anchor`: BCE with QD-B1 reference probabilities from the train pair examples.",
        "- `L_mono`: `mean_t max(0, p_{t+1}-p_t)`.",
        "",
        "Defaults: `lambda_pair=0.05`, `lambda_anchor=0.5`, `lambda_mono=0.1`, `epochs=3`, "
        "`learning_rate=1e-5`, `effective_batch_size=128`, `max_length=2048`.",
        "",
        "## Pair Dataset",
        "",
        f"- Train pairs: `{train_inventory['pair_count']}`",
        f"- Dev diagnostic pairs: `{dev_inventory['pair_count']}`",
        f"- Max low-record reuse: `{train_reuse['max_reuse_low_record']}`",
        f"- Mean low-record reuse: `{train_reuse['mean_reuse_low_record']}`",
        "",
        "Pairs are selected from question_seed42 human-only rows. The priority order is "
        "`same_question > same_metric_language`; `same_metric` and `any_valid` are fallback only and are "
        "reported when used.",
        "",
        "| split | pair_type | pairs | same_question | same_metric_language | fallback |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in comparability:
        lines.append(
            f"| {row['split']} | {row['pair_type']} | {row['pair_count']} | "
            f"{float(row['same_question_rate']):.4f} | {float(row['same_metric_language_rate']):.4f} | "
            f"{float(row['fallback_rate']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Training Gate",
            "",
            "QD-PR2 must load the QD-B1 checkpoint. If the checkpoint is missing, scripts must report "
            "`BLOCKED_MISSING_QDB1_CHECKPOINT` and stop before smoke/formal training.",
            "",
            "## QD-PR2 Controlled Changes",
            "",
            "- Initialize from QD-B1 checkpoint.",
            "- Keep `lambda_pair` small: first run `0.05`; optional sweep `{0.05, 0.1}`.",
            "- Keep anchor loss on train examples only.",
            "- Add monotonic regularization to protect cumulative probability order.",
            "- Use high-comparability pairs only unless fallback shortage is reported.",
            "- Fine-tune 2-3 epochs, never from scratch.",
        ]
    )
    report = "\n".join(lines)
    write_text(QDPR2_REPORTS_DIR / "qdpr2_method_plan.md", report)
    write_text(QDPR2_REPORTS_DIR / "review_package.md", report)
    write_text(QDPR2_REPORTS_DIR / "notion_qdpr2_summary.md", report)


def prepare_qdpr2_setup(config: QDPR2PairBuildConfig | None = None, write_shards: bool = False) -> dict[str, Any]:
    ensure_exp09_dirs()
    config = config or QDPR2PairBuildConfig()
    train_rows = read_split(QDPR2_DATASET_DIR, "train")
    dev_rows = read_split(QDPR2_DATASET_DIR, "dev")
    test_rows = read_split(QDPR2_DATASET_DIR, "test")
    train_pairs, train_distribution = build_pairs_for_split(train_rows, "train", config.train_pairs, config)
    dev_pairs, dev_distribution = build_pairs_for_split(dev_rows, "dev", config.dev_pairs, config)
    manifest_rows = []
    if write_shards:
        manifest_rows = write_qdpr2_pair_shards("train", train_pairs) + write_qdpr2_pair_shards("dev", dev_pairs)
        write_csv(QDPR2_TABLES_DIR / "qdpr2_pair_shard_manifest.csv", manifest_rows)
    inventory_rows = [
        {
            "split": "train",
            "pair_count": len(train_pairs),
            "target_count": config.train_pairs,
            "sampling_seed": config.seed,
            "max_pairs_per_record": config.max_pairs_per_record,
            "max_pairs_per_low_record": config.max_pairs_per_low_record,
            "write_shards": write_shards,
            "pair_shard_count": sum(1 for row in manifest_rows if row["split"] == "train"),
        },
        {
            "split": "dev",
            "pair_count": len(dev_pairs),
            "target_count": config.dev_pairs,
            "sampling_seed": config.seed,
            "max_pairs_per_record": config.max_pairs_per_record,
            "max_pairs_per_low_record": config.max_pairs_per_low_record,
            "write_shards": write_shards,
            "pair_shard_count": sum(1 for row in manifest_rows if row["split"] == "dev"),
        },
        {
            "split": "test",
            "pair_count": 0,
            "target_count": 0,
            "sampling_seed": config.seed,
            "max_pairs_per_record": config.max_pairs_per_record,
            "max_pairs_per_low_record": config.max_pairs_per_low_record,
            "write_shards": False,
            "pair_shard_count": 0,
        },
    ]
    comparability = pair_comparability_rows(train_pairs, "train") + pair_comparability_rows(dev_pairs, "dev")
    reuse = reuse_rows(train_pairs, "train", config) + reuse_rows(dev_pairs, "dev", config)
    write_csv(QDPR2_TABLES_DIR / "qdpr2_pair_inventory.csv", inventory_rows, fieldnames=PAIR_INVENTORY_FIELDS)
    write_csv(QDPR2_TABLES_DIR / "qdpr2_pair_distribution.csv", train_distribution + dev_distribution, fieldnames=PAIR_DISTRIBUTION_FIELDS)
    write_csv(QDPR2_TABLES_DIR / "qdpr2_pair_comparability_distribution.csv", comparability, fieldnames=PAIR_COMPARABILITY_FIELDS)
    write_csv(QDPR2_TABLES_DIR / "qdpr2_pair_reuse_stats.csv", reuse, fieldnames=PAIR_REUSE_FIELDS)
    write_reports(inventory_rows, train_distribution + dev_distribution, comparability, reuse)
    return {
        "train_pairs": train_pairs,
        "dev_pairs": dev_pairs,
        "test_rows": test_rows,
        "inventory_rows": inventory_rows,
        "distribution_rows": train_distribution + dev_distribution,
        "comparability_rows": comparability,
        "reuse_rows": reuse,
        "write_shards": write_shards,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare QD-PR2 anchored pairwise scaffold.")
    parser.add_argument("--write_shards", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = prepare_qdpr2_setup(write_shards=args.write_shards)
    print(
        "QD-PR2 setup prepared: "
        f"train_pairs={len(result['train_pairs'])} dev_pairs={len(result['dev_pairs'])} "
        f"write_shards={result['write_shards']}"
    )
    print(f"Reports: {relpath(QDPR2_REPORTS_DIR)}")


if __name__ == "__main__":
    main()
