"""Build Exp9 ordinal preference pair datasets."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp09_pairwise_ordinal import (
    DEFAULT_DEV_PAIR_COUNT,
    DEFAULT_GAP_WEIGHT,
    DEFAULT_LOW_HIGH_MARGIN,
    DEFAULT_LOW_HIGH_WEIGHT,
    DEFAULT_MARGIN_SCALE,
    DEFAULT_MAX_PAIRS_PER_LOW_RECORD,
    DEFAULT_MAX_PAIRS_PER_RECORD,
    DEFAULT_PAIR_SAMPLING_SEED,
    DEFAULT_TRAIN_PAIR_COUNT,
    EXP09_PAIRS_DIR,
    EXP09_REPORTS_DIR,
    EXP09_TABLES_DIR,
    PAIR_PRIORITIES,
    PAIR_TYPE_PROPORTIONS,
    ensure_exp09_dirs,
)
from thesis_exp.src.edujudge.exp09_pairwise_ordinal.data import record_key
from thesis_exp.src.edujudge.exp09_pairwise_ordinal.losses import pair_margin, pair_weight
from thesis_exp.src.edujudge.utils.io import read_jsonl, relpath, write_csv, write_jsonl, write_text


@dataclass(frozen=True)
class PairBuildConfig:
    train_pairs: int = DEFAULT_TRAIN_PAIR_COUNT
    dev_pairs: int = DEFAULT_DEV_PAIR_COUNT
    seed: int = DEFAULT_PAIR_SAMPLING_SEED
    max_pairs_per_record: int = DEFAULT_MAX_PAIRS_PER_RECORD
    max_pairs_per_low_record: int = DEFAULT_MAX_PAIRS_PER_LOW_RECORD
    margin_scale: float = DEFAULT_MARGIN_SCALE
    low_high_margin: float = DEFAULT_LOW_HIGH_MARGIN
    low_high_weight: float = DEFAULT_LOW_HIGH_WEIGHT
    gap_weight: float = DEFAULT_GAP_WEIGHT
    pair_type_proportions: dict[str, float] | None = None

    def proportions(self) -> dict[str, float]:
        return dict(self.pair_type_proportions or PAIR_TYPE_PROPORTIONS)


def pair_type(win_label: int, lose_label: int) -> str:
    gap = int(win_label) - int(lose_label)
    if lose_label <= 2 and win_label >= 4:
        return "low_high"
    if lose_label <= 2 and win_label == 3:
        return "low_mid"
    if gap == 1:
        return "adjacent"
    return "random_ordinal"


def _priority(row_a: dict[str, Any], row_b: dict[str, Any]) -> str:
    same_question = bool(row_a.get("question_key")) and row_a.get("question_key") == row_b.get("question_key")
    same_metric = row_a.get("metric_canonical") == row_b.get("metric_canonical")
    same_language = row_a.get("language") == row_b.get("language")
    if same_question:
        return "same_question"
    if same_metric and same_language:
        return "same_metric_language"
    if same_metric:
        return "same_metric"
    return "any_valid"


def _pair_row(
    split: str,
    pair_idx: int,
    win: dict[str, Any],
    lose: dict[str, Any],
    config: PairBuildConfig,
) -> dict[str, Any]:
    win_label = int(win["label_5"])
    lose_label = int(lose["label_5"])
    gap = win_label - lose_label
    low_high = int(lose_label <= 2 and win_label >= 4)
    margin = float(pair_margin([gap], [low_high], config.margin_scale, config.low_high_margin)[0])
    weight = float(pair_weight([gap], [low_high], config.low_high_weight, config.gap_weight)[0])
    same_question = bool(win.get("question_key")) and win.get("question_key") == lose.get("question_key")
    same_metric = win.get("metric_canonical") == lose.get("metric_canonical")
    same_language = win.get("language") == lose.get("language")
    return {
        "pair_id": f"{split}_pair_{pair_idx:06d}",
        "win_record_id": record_key(win),
        "lose_record_id": record_key(lose),
        "win_text": win["text"],
        "lose_text": lose["text"],
        "win_label_5": win_label,
        "lose_label_5": lose_label,
        "label_gap": gap,
        "pair_type": pair_type(win_label, lose_label),
        "priority": _priority(win, lose),
        "metric_canonical": win.get("metric_canonical") if same_metric else f"{win.get('metric_canonical')} || {lose.get('metric_canonical')}",
        "language": win.get("language") if same_language else f"{win.get('language')} || {lose.get('language')}",
        "same_question_key": same_question,
        "same_metric": same_metric,
        "same_language": same_language,
        "risk_pair_low_high": bool(low_high),
        "pair_weight": weight,
        "pair_margin": margin,
    }


def _candidate_pairs(rows: list[dict[str, Any]]) -> dict[str, dict[str, list[tuple[int, int]]]]:
    buckets: dict[str, dict[str, list[tuple[int, int]]]] = {
        pair_type_name: {priority: [] for priority in PAIR_PRIORITIES} for pair_type_name in PAIR_TYPE_PROPORTIONS
    }
    n = len(rows)
    for i in range(n):
        label_i = int(rows[i]["label_5"])
        for j in range(i + 1, n):
            label_j = int(rows[j]["label_5"])
            if label_i == label_j:
                continue
            if label_i > label_j:
                win_idx, lose_idx = i, j
                win_label, lose_label = label_i, label_j
            else:
                win_idx, lose_idx = j, i
                win_label, lose_label = label_j, label_i
            bucket = pair_type(win_label, lose_label)
            priority = _priority(rows[win_idx], rows[lose_idx])
            buckets[bucket][priority].append((win_idx, lose_idx))
    return buckets


def _target_counts(total: int, proportions: dict[str, float]) -> dict[str, int]:
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


def build_pairs_for_split(
    rows: list[dict[str, Any]],
    split: str,
    target_count: int,
    config: PairBuildConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(config.seed + (0 if split == "train" else 100_000))
    candidates = _candidate_pairs(rows)
    target_counts = _target_counts(target_count, config.proportions())
    reuse: Counter[str] = Counter()
    selected_keys: set[tuple[int, int]] = set()
    pairs: list[dict[str, Any]] = []
    shortage_rows: list[dict[str, Any]] = []

    def can_use(win_idx: int, lose_idx: int) -> bool:
        win = rows[win_idx]
        lose = rows[lose_idx]
        win_id = record_key(win)
        lose_id = record_key(lose)
        lose_label = int(lose["label_5"])
        if reuse[win_id] >= config.max_pairs_per_record:
            return False
        lose_limit = config.max_pairs_per_low_record if lose_label <= 2 else config.max_pairs_per_record
        if reuse[lose_id] >= lose_limit:
            return False
        return True

    def add_pair(win_idx: int, lose_idx: int) -> None:
        selected_keys.add((win_idx, lose_idx))
        reuse[record_key(rows[win_idx])] += 1
        reuse[record_key(rows[lose_idx])] += 1
        pairs.append(_pair_row(split, len(pairs), rows[win_idx], rows[lose_idx], config))

    for bucket, bucket_target in target_counts.items():
        before = len(pairs)
        for priority in PAIR_PRIORITIES:
            pool = [item for item in candidates[bucket][priority] if item not in selected_keys]
            rng.shuffle(pool)
            for win_idx, lose_idx in pool:
                if len(pairs) - before >= bucket_target:
                    break
                if can_use(win_idx, lose_idx):
                    add_pair(win_idx, lose_idx)
            if len(pairs) - before >= bucket_target:
                break
        actual = len(pairs) - before
        shortage_rows.append(
            {
                "split": split,
                "pair_type": bucket,
                "target_count": bucket_target,
                "actual_count": actual,
                "shortage": max(0, bucket_target - actual),
            }
        )

    if len(pairs) < target_count:
        fallback = []
        for bucket in PAIR_TYPE_PROPORTIONS:
            for priority in PAIR_PRIORITIES:
                fallback.extend(item for item in candidates[bucket][priority] if item not in selected_keys)
        rng.shuffle(fallback)
        for win_idx, lose_idx in fallback:
            if len(pairs) >= target_count:
                break
            if can_use(win_idx, lose_idx):
                add_pair(win_idx, lose_idx)

    return pairs[:target_count], shortage_rows


def distribution_rows(pairs: list[dict[str, Any]], split: str, field: str) -> list[dict[str, Any]]:
    counts = Counter(str(pair.get(field)) for pair in pairs)
    total = len(pairs)
    return [
        {"split": split, field: value, "count": count, "rate": count / total if total else 0.0}
        for value, count in sorted(counts.items())
    ]


def reuse_rows(pairs: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
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
        }
    ]


def pointwise_label_distribution(rows_by_split: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split, rows in rows_by_split.items():
        counts = Counter(int(row["label_5"]) for row in rows)
        total = len(rows)
        for label in [1, 2, 3, 4, 5]:
            out.append({"split": split, "label_5": label, "count": counts[label], "rate": counts[label] / total if total else 0.0})
    return out


def pair_shard_paths(split: str, pairs_dir: Path = EXP09_PAIRS_DIR) -> list[Path]:
    return sorted(pairs_dir.glob(f"{split}_pairs_shard_*.jsonl"))


def write_pair_shards(
    split: str,
    pairs: list[dict[str, Any]],
    pairs_dir: Path = EXP09_PAIRS_DIR,
    shard_rows: int = 2_500,
) -> list[dict[str, Any]]:
    pairs_dir.mkdir(parents=True, exist_ok=True)
    for old_path in pair_shard_paths(split, pairs_dir):
        old_path.unlink()
    legacy_path = pairs_dir / f"{split}_pairs.jsonl"
    if legacy_path.exists():
        legacy_path.unlink()
    manifest: list[dict[str, Any]] = []
    for shard_idx, start in enumerate(range(0, len(pairs), shard_rows)):
        shard = pairs[start : start + shard_rows]
        shard_path = pairs_dir / f"{split}_pairs_shard_{shard_idx:03d}.jsonl"
        write_jsonl(shard_path, shard)
        manifest.append(
            {
                "split": split,
                "shard_index": shard_idx,
                "path": relpath(shard_path),
                "row_count": len(shard),
                "byte_size": shard_path.stat().st_size,
            }
        )
    return manifest


def read_pair_shards(split: str, pairs_dir: Path = EXP09_PAIRS_DIR) -> list[dict[str, Any]]:
    paths = pair_shard_paths(split, pairs_dir)
    legacy = pairs_dir / f"{split}_pairs.jsonl"
    if not paths and legacy.exists():
        paths = [legacy]
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return rows


def write_pair_outputs(
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    config: PairBuildConfig | None = None,
) -> dict[str, Any]:
    ensure_exp09_dirs()
    config = config or PairBuildConfig()
    train_pairs, train_shortage = build_pairs_for_split(train_rows, "train", config.train_pairs, config)
    dev_pairs, dev_shortage = build_pairs_for_split(dev_rows, "dev", config.dev_pairs, config)
    manifest_rows = write_pair_shards("train", train_pairs) + write_pair_shards("dev", dev_pairs)
    write_csv(EXP09_TABLES_DIR / "pair_shard_manifest.csv", manifest_rows)

    inventory_rows = [
        {
            "split": "train",
            "pair_count": len(train_pairs),
            "target_count": config.train_pairs,
            "sampling_seed": config.seed,
            "max_pairs_per_record": config.max_pairs_per_record,
            "max_pairs_per_low_record": config.max_pairs_per_low_record,
        },
        {
            "split": "dev",
            "pair_count": len(dev_pairs),
            "target_count": config.dev_pairs,
            "sampling_seed": config.seed,
            "max_pairs_per_record": config.max_pairs_per_record,
            "max_pairs_per_low_record": config.max_pairs_per_low_record,
        },
    ]
    write_csv(EXP09_TABLES_DIR / "pair_inventory.csv", inventory_rows)
    write_csv(EXP09_TABLES_DIR / "pair_distribution.csv", train_shortage + dev_shortage)
    write_csv(EXP09_TABLES_DIR / "pair_type_distribution.csv", distribution_rows(train_pairs, "train", "pair_type") + distribution_rows(dev_pairs, "dev", "pair_type"))
    write_csv(EXP09_TABLES_DIR / "pair_label_gap_distribution.csv", distribution_rows(train_pairs, "train", "label_gap") + distribution_rows(dev_pairs, "dev", "label_gap"))
    write_csv(EXP09_TABLES_DIR / "pair_priority_distribution.csv", distribution_rows(train_pairs, "train", "priority") + distribution_rows(dev_pairs, "dev", "priority"))
    write_csv(EXP09_TABLES_DIR / "pair_reuse_stats.csv", reuse_rows(train_pairs, "train") + reuse_rows(dev_pairs, "dev"))
    write_csv(EXP09_TABLES_DIR / "pointwise_label_distribution.csv", pointwise_label_distribution({"train": train_rows, "dev": dev_rows, "test": test_rows}))

    train_type_counts = Counter(pair["pair_type"] for pair in train_pairs)
    dev_type_counts = Counter(pair["pair_type"] for pair in dev_pairs)
    reuse = reuse_rows(train_pairs, "train")[0]
    lines = [
        "# Exp9 Pair Dataset Report",
        "",
        "Status: `ready_for_smoke`",
        "",
        f"- Train pairs: `{len(train_pairs)}`",
        f"- Dev diagnostic pairs: `{len(dev_pairs)}`",
        f"- low_high train pairs: `{train_type_counts.get('low_high', 0)}`",
        f"- Pair shards: `{len(manifest_rows)}`",
        f"- Max low-record reuse: `{reuse['max_reuse_low_record']}`",
        f"- Mean low-record reuse: `{reuse['mean_reuse_low_record']}`",
        "",
        "## Train Pair Type Distribution",
        "",
        "| pair_type | count | rate |",
        "| --- | ---: | ---: |",
    ]
    for name in PAIR_TYPE_PROPORTIONS:
        count = train_type_counts.get(name, 0)
        lines.append(f"| {name} | {count} | {count / len(train_pairs) if train_pairs else 0:.4f} |")
    lines.extend(["", "## Dev Pair Type Distribution", "", "| pair_type | count | rate |", "| --- | ---: | ---: |"])
    for name in PAIR_TYPE_PROPORTIONS:
        count = dev_type_counts.get(name, 0)
        lines.append(f"| {name} | {count} | {count / len(dev_pairs) if dev_pairs else 0:.4f} |")
    lines.extend(
        [
            "",
            "No synthetic rows are used. Dev pairs are diagnostic only and are not used to tune test thresholds.",
        ]
    )
    report = "\n".join(lines)
    write_text(EXP09_REPORTS_DIR / "pair_dataset_report.md", report)
    write_text(EXP09_REPORTS_DIR / "review_package.md", report)
    write_text(EXP09_REPORTS_DIR / "notion_exp09_pairwise_summary.md", report)
    return {
        "train_pairs": len(train_pairs),
        "dev_pairs": len(dev_pairs),
        "low_high_train_pairs": train_type_counts.get("low_high", 0),
        "train_type_counts": dict(train_type_counts),
        "reuse": reuse,
    }
