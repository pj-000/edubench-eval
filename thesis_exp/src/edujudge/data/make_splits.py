"""Create deterministic Exp 0 train/dev/test splits."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.data.build_dataset import PROCESSED_PATH
from thesis_exp.src.edujudge.data.reference_contract import PDF_HELDOUT_TEST_ROWS, PDF_TRAIN_POOL_ROWS
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


def _assign_original_paper_like(rows: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], str]:
    flags = [row.get("original_is_test_set") for row in rows]
    if all(flag is not None for flag in flags):
        groups = _group_rows(rows, "triple_key")
        test_keys: set[str] = set()
        pool_keys: set[str] = set()
        for key, group in groups.items():
            flags_for_group = {row.get("original_is_test_set") for row in group}
            if True in flags_for_group:
                test_keys.add(key)
            else:
                pool_keys.add(key)

        def materialize(keys: set[str]) -> list[dict[str, Any]]:
            return [row for key in keys for row in groups[key]]

        # If original held-out flags split duplicate triples, keep those triples
        # together, then swap whole singleton groups to restore the PDF row count.
        test = materialize(test_keys)
        pool = materialize(pool_keys)
        if len(test) > PDF_HELDOUT_TEST_ROWS:
            need = len(test) - PDF_HELDOUT_TEST_ROWS
            candidates = sorted(
                [key for key in test_keys if len(groups[key]) <= need],
                key=lambda key: (len(groups[key]), min(int(row.get("source_row_index", 0)) for row in groups[key])),
            )
            moved = 0
            for key in candidates:
                size = len(groups[key])
                if moved + size > need:
                    continue
                test_keys.remove(key)
                pool_keys.add(key)
                moved += size
                if moved == need:
                    break
        elif len(test) < PDF_HELDOUT_TEST_ROWS:
            need = PDF_HELDOUT_TEST_ROWS - len(test)
            candidates = sorted(
                [key for key in pool_keys if len(groups[key]) <= need],
                key=lambda key: (len(groups[key]), min(int(row.get("source_row_index", 0)) for row in groups[key])),
            )
            moved = 0
            for key in candidates:
                size = len(groups[key])
                if moved + size > need:
                    continue
                pool_keys.remove(key)
                test_keys.add(key)
                moved += size
                if moved == need:
                    break

        train_pool = materialize(pool_keys)
        test = materialize(test_keys)
        if len(train_pool) == PDF_TRAIN_POOL_ROWS and len(test) == PDF_HELDOUT_TEST_ROWS:
            pool_groups = _group_rows(train_pool, "triple_key")
            targets = {"train": round(PDF_TRAIN_POOL_ROWS * 0.8), "dev": PDF_TRAIN_POOL_ROWS - round(PDF_TRAIN_POOL_ROWS * 0.8)}
            pool_assignments = {"train": [], "dev": []}
            counts = {"train": 0, "dev": 0}
            for info in _group_infos(pool_groups, SEED):
                size = int(info["size"])
                chosen = min(
                    ["train", "dev"],
                    key=lambda split: (max(0, counts[split] + size - targets[split]), counts[split] / max(1, targets[split]), split),
                )
                pool_assignments[chosen].extend(info["rows"])
                counts[chosen] += size
            train = pool_assignments["train"]
            dev = pool_assignments["dev"]
            train.sort(key=lambda row: int(row.get("source_row_index", 0)))
            dev.sort(key=lambda row: int(row.get("source_row_index", 0)))
            test.sort(key=lambda row: int(row.get("source_row_index", 0)))
            return {"train": train, "dev": dev, "test": test}, "original_heldout_flag_repaired_for_triple_isolation"
    return _assign_groups(rows, "triple_key", "paper_like_triple_seed42"), "deterministic_reconstructed_triple_seed42"


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


def _split_stats(split_name: str, assignments: dict[str, list[dict[str, Any]]], group_key: str, split_source: str) -> list[dict[str, Any]]:
    rows = []
    total = sum(len(values) for values in assignments.values())
    targets = _target_counts(total, split_name)
    for split, split_rows in assignments.items():
        rows.append(
            {
                "split_name": split_name,
                "split": split,
                "group_key": group_key,
                "split_source": split_source,
                "target_rows": targets[split],
                "num_rows": len(split_rows),
                "train_pool_rows": len(assignments.get("train", [])) + len(assignments.get("dev", [])) if split == "test" else "",
                "pct_rows": round(len(split_rows) / max(1, total), 6),
                "unique_triple_key": len({row.get("triple_key") for row in split_rows}),
                "unique_question_key": len({row.get("question_key") for row in split_rows}),
                "unique_answer_key": len({row.get("answer_key") for row in split_rows}),
                "metric_count": len({row.get("metric_canonical") for row in split_rows}),
                "scenario_count": len({row.get("scenario_canonical") for row in split_rows}),
                "subject_count": len({row.get("subject_canonical") for row in split_rows}),
                "education_level_count": len({row.get("education_level_canonical") for row in split_rows}),
                "language_count": len({row.get("language") for row in split_rows}),
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
    subject_rows: list[dict[str, Any]] = []
    coverage_metric_rows: list[dict[str, Any]] = []
    coverage_scenario_rows: list[dict[str, Any]] = []
    coverage_subject_rows: list[dict[str, Any]] = []
    coverage_label_rows: list[dict[str, Any]] = []
    split_sources: dict[str, str] = {}

    for split_name, group_key in split_configs.items():
        if split_name == "paper_like_triple_seed42":
            assignments, split_source = _assign_original_paper_like(rows)
        else:
            assignments = _assign_groups(rows, group_key, split_name)
            split_source = "deterministic_reconstructed_question_seed42"
        split_sources[split_name] = split_source
        all_assignments[split_name] = assignments
        split_dir = SPLITS_DIR / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for split, split_rows in assignments.items():
            write_jsonl(split_dir / f"{split}.jsonl", split_rows)
        stats_rows.extend(_split_stats(split_name, assignments, group_key, split_source))
        label_rows.extend(_distribution_rows(split_name, assignments, "label_5", "label_5"))
        metric_rows.extend(_distribution_rows(split_name, assignments, "metric_canonical", "metric_canonical"))
        scenario_rows.extend(_distribution_rows(split_name, assignments, "scenario_canonical", "scenario_canonical"))
        subject_rows.extend(_distribution_rows(split_name, assignments, "subject_canonical", "subject_canonical"))
        coverage_metric_rows.extend(_distribution_rows(split_name, assignments, "metric_canonical", "metric_canonical"))
        coverage_scenario_rows.extend(_distribution_rows(split_name, assignments, "scenario_canonical", "scenario_canonical"))
        coverage_subject_rows.extend(_distribution_rows(split_name, assignments, "subject_canonical", "subject_canonical"))
        coverage_label_rows.extend(_distribution_rows(split_name, assignments, "label_5", "label_5"))

    write_csv(
        TABLES_DIR / "split_stats.csv",
        stats_rows,
        [
            "split_name",
            "split",
            "group_key",
            "split_source",
            "target_rows",
            "num_rows",
            "train_pool_rows",
            "pct_rows",
            "unique_triple_key",
            "unique_question_key",
            "unique_answer_key",
            "metric_count",
            "scenario_count",
            "subject_count",
            "education_level_count",
            "language_count",
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
    write_csv(
        TABLES_DIR / "split_subject_distribution.csv",
        subject_rows,
        ["split_name", "split", "subject_canonical", "count", "pct_within_split"],
    )
    write_csv(TABLES_DIR / "split_coverage_by_metric.csv", coverage_metric_rows, ["split_name", "split", "metric_canonical", "count", "pct_within_split"])
    write_csv(TABLES_DIR / "split_coverage_by_scenario.csv", coverage_scenario_rows, ["split_name", "split", "scenario_canonical", "count", "pct_within_split"])
    write_csv(TABLES_DIR / "split_coverage_by_subject.csv", coverage_subject_rows, ["split_name", "split", "subject_canonical", "count", "pct_within_split"])
    write_csv(TABLES_DIR / "split_coverage_by_label.csv", coverage_label_rows, ["split_name", "split", "label_5", "count", "pct_within_split"])
    _write_split_reference_check(all_assignments, split_sources)
    return all_assignments


def _write_split_reference_check(assignments_by_name: dict[str, dict[str, list[dict[str, Any]]]], split_sources: dict[str, str]) -> None:
    rows = []
    paper = assignments_by_name.get("paper_like_triple_seed42", {})
    test = paper.get("test", [])
    train_pool_rows = len(paper.get("train", [])) + len(paper.get("dev", []))
    rows.append({"field": "split_name", "value": "paper_like_triple_seed42"})
    rows.append({"field": "split_source", "value": split_sources.get("paper_like_triple_seed42", "")})
    rows.append({"field": "train rows", "value": len(paper.get("train", []))})
    rows.append({"field": "dev rows", "value": len(paper.get("dev", []))})
    rows.append({"field": "train_pool rows", "value": train_pool_rows})
    rows.append({"field": "test rows", "value": len(test)})
    rows.append({"field": "test scenario count", "value": len({row.get("scenario_canonical") for row in test})})
    rows.append({"field": "test metric count", "value": len({row.get("metric_canonical") for row in test})})
    rows.append({"field": "test subject count", "value": len({row.get("subject_canonical") for row in test})})
    rows.append({"field": "test education level count", "value": len({row.get("education_level_canonical") for row in test})})
    rows.append({"field": "test language count", "value": len({row.get("language") for row in test})})
    scenario_count = len({row.get("scenario_canonical") for row in test})
    note = "This split uses the local original held-out flag from report/results_merge_enriched.jsonl." if split_sources.get("paper_like_triple_seed42", "").startswith("original_") else "This is a reconstructed paper-like split matching row counts and triple isolation, not necessarily the exact original PDF split."
    if scenario_count != 8:
        note += " Test scenario count differs from the PDF reference of 8 scenarios."
    lines = [
        "# Split Reference Check",
        "",
        note,
        "",
        "| field | value |",
        "| --- | --- |",
        *[f"| {row['field']} | {row['value']} |" for row in rows],
        "",
        "PDF reference: train_pool = 3318, held-out test = 2218, split unit = question-answer-metric triple. The held-out test reported in the PDF spans 8 scenarios and complete coverage of subjects, education levels, and dimensions.",
    ]
    from thesis_exp.src.edujudge.utils.io import OUTPUT_DIR, write_text

    write_text(OUTPUT_DIR / "split_reference_check.md", "\n".join(lines))


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
