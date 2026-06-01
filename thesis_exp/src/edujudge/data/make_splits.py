"""Create deterministic Exp 0 train/dev/test splits."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.data.build_dataset import PROCESSED_PATH
from thesis_exp.src.edujudge.utils.io import SPLITS_DIR, TABLES_DIR, ensure_exp_dirs, read_jsonl, write_csv, write_jsonl
from thesis_exp.src.edujudge.utils.text_norm import stringify


SEED = 42
SPLIT_NAMES = ["train", "dev", "test"]


def _target_counts(total: int, split_name: str) -> dict[str, int]:
    if split_name == "paper_like_triple_seed42" and abs(total - 5536) <= 100:
        train_pool = round(total * 3318 / 5536)
        test = total - train_pool
        train = round(train_pool * 0.8)
        dev = train_pool - train
        return {"train": train, "dev": dev, "test": test}
    train = round(total * 0.6)
    dev = round(total * 0.2)
    test = total - train - dev
    return {"train": train, "dev": dev, "test": test}


def _group_rows(rows: list[dict[str, Any]], group_key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[stringify(row.get(group_key, ""))].append(row)
    return dict(groups)


def _dominant(values: list[Any]) -> str:
    counts = Counter(stringify(value or "unknown") for value in values)
    return counts.most_common(1)[0][0] if counts else "unknown"


def _group_infos(groups: dict[str, list[dict[str, Any]]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    infos = []
    for group_id, group_rows in groups.items():
        labels = [row.get("label_5") for row in group_rows]
        metrics = [row.get("metric_canonical") for row in group_rows]
        scenarios = [row.get("scenario_canonical") for row in group_rows]
        infos.append(
            {
                "group_id": group_id,
                "rows": group_rows,
                "size": len(group_rows),
                "stratum": f"{_dominant(labels)}|{_dominant(metrics)}|{_dominant(scenarios)}",
                "rand": rng.random(),
            }
        )
    infos.sort(key=lambda item: (item["stratum"], item["rand"]))
    return infos


def _assign_groups(rows: list[dict[str, Any]], group_key: str, split_name: str, seed: int = SEED) -> dict[str, list[dict[str, Any]]]:
    groups = _group_rows(rows, group_key)
    targets = _target_counts(len(rows), split_name)
    assignments: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    counts = {name: 0 for name in SPLIT_NAMES}

    for info in _group_infos(groups, seed):
        size = int(info["size"])

        def score(split: str) -> tuple[float, int, str]:
            target = max(1, targets[split])
            overflow = max(0, counts[split] + size - targets[split])
            fill_ratio = counts[split] / target
            return (overflow, fill_ratio, split)

        chosen = min(SPLIT_NAMES, key=score)
        assignments[chosen].extend(info["rows"])
        counts[chosen] += size

    for split_rows in assignments.values():
        split_rows.sort(key=lambda row: int(row.get("source_row_index", 0)))
    return assignments


def _distribution_rows(split_name: str, assignments: dict[str, list[dict[str, Any]]], key: str, output_key: str) -> list[dict[str, Any]]:
    rows = []
    for split, split_rows in assignments.items():
        counts = Counter(stringify(row.get(key, "unknown") or "unknown") for row in split_rows)
        total = sum(counts.values())
        for value, count in counts.most_common():
            rows.append(
                {
                    "split_name": split_name,
                    "split": split,
                    output_key: value,
                    "count": count,
                    "pct_within_split": round(count / max(1, total), 6),
                }
            )
    return rows


def _split_stats(split_name: str, assignments: dict[str, list[dict[str, Any]]], group_key: str) -> list[dict[str, Any]]:
    rows = []
    total = sum(len(values) for values in assignments.values())
    targets = _target_counts(total, split_name)
    for split, split_rows in assignments.items():
        rows.append(
            {
                "split_name": split_name,
                "split": split,
                "group_key": group_key,
                "target_rows": targets[split],
                "num_rows": len(split_rows),
                "pct_rows": round(len(split_rows) / max(1, total), 6),
                "unique_triple_key": len({row.get("triple_key") for row in split_rows}),
                "unique_question_key": len({row.get("question_key") for row in split_rows}),
                "unique_answer_key": len({row.get("answer_key") for row in split_rows}),
                "label_distribution": dict(Counter(stringify(row.get("label_5")) for row in split_rows)),
            }
        )
    return rows


def create_splits(rows: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    split_configs = {
        "paper_like_triple_seed42": "triple_key",
        "question_seed42": "question_key",
    }
    all_assignments: dict[str, dict[str, list[dict[str, Any]]]] = {}
    stats_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []

    for split_name, group_key in split_configs.items():
        assignments = _assign_groups(rows, group_key, split_name)
        all_assignments[split_name] = assignments
        split_dir = SPLITS_DIR / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for split, split_rows in assignments.items():
            write_jsonl(split_dir / f"{split}.jsonl", split_rows)
        stats_rows.extend(_split_stats(split_name, assignments, group_key))
        label_rows.extend(_distribution_rows(split_name, assignments, "label_5", "label_5"))
        metric_rows.extend(_distribution_rows(split_name, assignments, "metric_canonical", "metric_canonical"))
        scenario_rows.extend(_distribution_rows(split_name, assignments, "scenario_canonical", "scenario_canonical"))

    write_csv(
        TABLES_DIR / "split_stats.csv",
        stats_rows,
        [
            "split_name",
            "split",
            "group_key",
            "target_rows",
            "num_rows",
            "pct_rows",
            "unique_triple_key",
            "unique_question_key",
            "unique_answer_key",
            "label_distribution",
        ],
    )
    write_csv(TABLES_DIR / "split_label_distribution.csv", label_rows, ["split_name", "split", "label_5", "count", "pct_within_split"])
    write_csv(
        TABLES_DIR / "split_metric_distribution.csv",
        metric_rows,
        ["split_name", "split", "metric_canonical", "count", "pct_within_split"],
    )
    write_csv(
        TABLES_DIR / "split_scenario_distribution.csv",
        scenario_rows,
        ["split_name", "split", "scenario_canonical", "count", "pct_within_split"],
    )
    return all_assignments


def main() -> None:
    ensure_exp_dirs()
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found. Run build_dataset first: {PROCESSED_PATH}")
    rows = read_jsonl(PROCESSED_PATH)
    assignments = create_splits(rows)
    summary = {
        split_name: {split: len(split_rows) for split, split_rows in split_values.items()}
        for split_name, split_values in assignments.items()
    }
    print(f"Wrote splits under {SPLITS_DIR}: {summary}")


if __name__ == "__main__":
    main()
