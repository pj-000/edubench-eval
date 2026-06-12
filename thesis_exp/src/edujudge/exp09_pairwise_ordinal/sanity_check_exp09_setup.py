"""Setup sanity checks for Exp9 pairwise ordinal training."""

from __future__ import annotations

import math
import os
import py_compile
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.src.edujudge.exp09_pairwise_ordinal import (
    DEFAULT_DEV_PAIR_COUNT,
    DEFAULT_GAP_WEIGHT,
    DEFAULT_LAMBDA_PAIR,
    DEFAULT_LOW_HIGH_MARGIN,
    DEFAULT_LOW_HIGH_WEIGHT,
    DEFAULT_MARGIN_SCALE,
    DEFAULT_MAX_PAIRS_PER_LOW_RECORD,
    DEFAULT_MAX_PAIRS_PER_RECORD,
    DEFAULT_TRAIN_PAIR_COUNT,
    EXP09_CONFIG_DIR,
    EXP09_OUTPUT_DIR,
    EXP09_PAIRS_DIR,
    EXP09_RUN_ID,
    EXP09_SRC_DIR,
    EXP09_TABLES_DIR,
    PAIR_TYPE_PROPORTIONS,
    ensure_exp09_dirs,
)
from thesis_exp.src.edujudge.exp09_pairwise_ordinal.collect_exp09_results import collect
from thesis_exp.src.edujudge.exp09_pairwise_ordinal.data import (
    class_weight_vector,
    dataset_sanity_rows,
    exp0_to_exp8_tracked_output_changes,
    load_splits,
    read_split,
    split_record_ids,
    tracked_weight_files,
    write_pointwise_class_weights,
)
from thesis_exp.src.edujudge.exp09_pairwise_ordinal.losses import (
    pair_margin,
    pair_weight,
    scalar_score_from_logits,
    total_pairwise_training_loss,
)
from thesis_exp.src.edujudge.exp09_pairwise_ordinal.pair_builder import (
    PairBuildConfig,
    read_pair_shards,
    reuse_rows,
    write_pair_outputs,
)
from thesis_exp.src.edujudge.utils.io import read_csv, relpath, write_csv, write_text


SCRIPT_PATHS = [
    Path("thesis_exp/scripts/run_exp09_qdpr1_smoke.sh"),
    Path("thesis_exp/scripts/run_exp09_qdpr1_train.sh"),
    Path("thesis_exp/scripts/sync_exp09_qdpr1_to_server.sh"),
]


def add(rows: list[dict[str, Any]], check_name: str, passed: bool, details: Any = "") -> None:
    rows.append({"check_name": check_name, "status": "PASS" if passed else "FAIL", "details": details})


def check_py_compile(rows: list[dict[str, Any]]) -> None:
    for path in sorted(EXP09_SRC_DIR.glob("*.py")):
        try:
            py_compile.compile(str(path), doraise=True)
            add(rows, f"py_compile {relpath(path)}", True)
        except py_compile.PyCompileError as exc:
            add(rows, f"py_compile {relpath(path)}", False, str(exc))


def check_scripts(rows: list[dict[str, Any]]) -> None:
    for path in SCRIPT_PATHS:
        exists = path.exists()
        add(rows, f"script exists {relpath(path)}", exists)
        if not exists:
            continue
        result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        add(rows, f"bash -n {relpath(path)}", result.returncode == 0, (result.stderr or result.stdout).strip())


def check_config(rows: list[dict[str, Any]]) -> None:
    formal = EXP09_CONFIG_DIR / "exp09_qdpr1_pairwise_human_only.yaml"
    smoke = EXP09_CONFIG_DIR / "exp09_qdpr1_pairwise_smoke.yaml"
    for path in [formal, smoke]:
        add(rows, f"config exists {relpath(path)}", path.exists())
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        add(rows, f"config declares run id {path.name}", EXP09_RUN_ID in text)
        add(rows, f"config disables synthetic {path.name}", "use_synthetic: false" in text)
        add(rows, f"config disables coral {path.name}", "use_coral: false" in text)
        add(rows, f"config disables edurisk {path.name}", "use_edurisk: false" in text)
    if formal.exists():
        text = formal.read_text(encoding="utf-8")
        add(rows, "formal train pair count is 20000", "pair_dataset_size_train: 20000" in text)
        add(rows, "formal dev pair count is 5000", "pair_dataset_size_dev: 5000" in text)
        add(rows, "formal lambda_pair is 0.3", "lambda_pair: 0.3" in text)


def check_pair_rows(rows: list[dict[str, Any]], train_pairs: list[dict[str, Any]], dev_pairs: list[dict[str, Any]]) -> None:
    add(rows, "train pair count = 20000", len(train_pairs) == DEFAULT_TRAIN_PAIR_COUNT, len(train_pairs))
    add(rows, "dev diagnostic pair count = 5000", len(dev_pairs) == DEFAULT_DEV_PAIR_COUNT, len(dev_pairs))
    for split, pairs in [("train", train_pairs), ("dev", dev_pairs)]:
        add(rows, f"{split} pair win_label > lose_label", all(int(pair["win_label_5"]) > int(pair["lose_label_5"]) for pair in pairs))
        add(rows, f"{split} pair margins positive", all(float(pair["pair_margin"]) > 0 for pair in pairs))
        add(rows, f"{split} pair weights finite", all(math.isfinite(float(pair["pair_weight"])) for pair in pairs))
        add(rows, f"{split} low_high pair count > 0", sum(1 for pair in pairs if pair["pair_type"] == "low_high") > 0)
        type_counts = Counter(str(pair["pair_type"]) for pair in pairs)
        total = len(pairs)
        for pair_type_name, target_rate in PAIR_TYPE_PROPORTIONS.items():
            rate = type_counts[pair_type_name] / total if total else 0.0
            add(
                rows,
                f"{split} {pair_type_name} proportion close or reported",
                abs(rate - target_rate) <= 0.05,
                f"rate={rate:.4f}; target={target_rate:.4f}",
            )


def check_split_leakage(rows: list[dict[str, Any]], train_pairs: list[dict[str, Any]]) -> None:
    dev_test_ids = split_record_ids(read_split(split="dev", data_dir=Path("thesis_exp/outputs/exp06_synthetic_low_score/training_datasets/QD-S0_human_only"))) | split_record_ids(
        read_split(split="test", data_dir=Path("thesis_exp/outputs/exp06_synthetic_low_score/training_datasets/QD-S0_human_only"))
    )
    train_pair_ids = {str(pair["win_record_id"]) for pair in train_pairs} | {str(pair["lose_record_id"]) for pair in train_pairs}
    overlap = sorted(train_pair_ids & dev_test_ids)
    add(rows, "no dev/test records in train pairs", not overlap, len(overlap))


def check_reuse(rows: list[dict[str, Any]], train_pairs: list[dict[str, Any]], dev_pairs: list[dict[str, Any]]) -> dict[str, Any]:
    train_reuse = reuse_rows(train_pairs, "train")[0]
    dev_reuse = reuse_rows(dev_pairs, "dev")[0]
    add(rows, "train max reuse within cap", int(train_reuse["max_reuse"]) <= DEFAULT_MAX_PAIRS_PER_LOW_RECORD, train_reuse)
    add(rows, "train low-record max reuse within cap", int(train_reuse["max_reuse_low_record"]) <= DEFAULT_MAX_PAIRS_PER_LOW_RECORD, train_reuse)
    add(rows, "dev max reuse within cap", int(dev_reuse["max_reuse"]) <= DEFAULT_MAX_PAIRS_PER_LOW_RECORD, dev_reuse)
    return train_reuse


def check_pointwise_weights(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    train_rows = read_split(Path("thesis_exp/outputs/exp06_synthetic_low_score/training_datasets/QD-S0_human_only"), "train")
    weight_rows = write_pointwise_class_weights(train_rows)
    vector = class_weight_vector(weight_rows)
    add(rows, "pointwise class weights have six-index vector", len(vector) == 6, vector)
    add(rows, "pointwise class weights finite", all(math.isfinite(value) for value in vector[1:]), vector)
    add(rows, "pointwise class weights clipped to [0.5, 3.0]", min(vector[1:]) >= 0.5 and max(vector[1:]) <= 3.0, vector)
    return weight_rows


def check_toy_loss(rows: list[dict[str, Any]], class_weights: list[float]) -> dict[str, float]:
    win_logits = np.array([[2.5, 1.0, 0.2, -1.0], [2.0, 0.9, -0.5, -1.2], [3.0, 2.0, 1.0, 0.1]], dtype=np.float64)
    lose_logits = np.array([[0.1, -0.6, -1.5, -2.2], [1.0, 0.0, -1.2, -2.0], [1.6, 0.7, -0.4, -1.6]], dtype=np.float64)
    win_labels = np.array([5, 4, 5], dtype=np.int64)
    lose_labels = np.array([1, 2, 4], dtype=np.int64)
    label_gap = win_labels - lose_labels
    low_high = ((lose_labels <= 2) & (win_labels >= 4)).astype(np.float64)
    margins = pair_margin(label_gap, low_high, DEFAULT_MARGIN_SCALE, DEFAULT_LOW_HIGH_MARGIN)
    weights = pair_weight(label_gap, low_high, DEFAULT_LOW_HIGH_WEIGHT, DEFAULT_GAP_WEIGHT)
    loss, debug = total_pairwise_training_loss(
        win_logits,
        lose_logits,
        win_labels,
        lose_labels,
        label_gap,
        low_high,
        class_weights,
        lambda_pair=DEFAULT_LAMBDA_PAIR,
    )
    debug = {key: round(float(value), 12) for key, value in debug.items()}
    win_scores = scalar_score_from_logits(win_logits)
    lose_scores = scalar_score_from_logits(lose_logits)
    add(rows, "toy pairwise loss finite", bool(np.isfinite(loss)), loss)
    add(rows, "toy pair margins positive", bool(np.all(margins > 0)), margins.tolist())
    add(rows, "toy pair weights finite", bool(np.isfinite(weights).all()), weights.tolist())
    add(rows, "toy scalar scores in [1,5]", bool(np.all((win_scores >= 1) & (win_scores <= 5)) and np.all((lose_scores >= 1) & (lose_scores <= 5))), {"win": win_scores.tolist(), "lose": lose_scores.tolist()})
    for key in [
        "L_total",
        "L_point",
        "L_pair",
        "weighted_L_pair",
        "mean_pair_weight",
        "mean_pair_margin",
        "mean_score_gap",
        "low_high_pair_loss",
        "adjacent_pair_loss",
        "mean_point_base_loss",
        "mean_point_sample_weight",
    ]:
        add(rows, f"toy debug has {key}", key in debug and np.isfinite(float(debug[key])), debug.get(key))
    write_csv(EXP09_TABLES_DIR / "toy_pairwise_loss_scale.csv", [debug])
    return debug


def check_repository_safety(rows: list[dict[str, Any]]) -> None:
    weights = tracked_weight_files()
    add(rows, "no checkpoint/weights tracked", not weights, ", ".join(weights))
    changed = exp0_to_exp8_tracked_output_changes()
    allow_existing = os.environ.get("EXP09_EXISTING_EXP0_EXP8_DIRTY_OK", "0") == "1"
    add(
        rows,
        "no tracked Exp0-Exp8 output modifications",
        not changed or allow_existing,
        "" if changed and allow_existing else ", ".join(changed),
    )


def write_setup_report(rows: list[dict[str, Any]], pair_summary: dict[str, Any], toy_debug: dict[str, float]) -> None:
    write_csv(EXP09_TABLES_DIR / "sanity_check_exp09_setup.csv", rows)
    failures = [row for row in rows if row["status"] != "PASS"]
    lines = [
        "# Exp9 Setup Sanity Check",
        "",
        f"Status: `{'PASS' if not failures else 'FAIL'}`",
        "",
        "Training executed: `no`.",
        "API called: `no`.",
        "Synthetic generated: `no`.",
        "",
        "## Pair Summary",
        "",
        f"- Train pairs: `{pair_summary.get('train_pairs')}`",
        f"- Dev diagnostic pairs: `{pair_summary.get('dev_pairs')}`",
        f"- Train low_high pairs: `{pair_summary.get('low_high_train_pairs')}`",
        f"- Toy L_total: `{toy_debug.get('L_total')}`",
        f"- Toy L_pair: `{toy_debug.get('L_pair')}`",
        "",
        "## Checks",
        "",
        "| check | status | details |",
        "| --- | --- | --- |",
    ]
    lines.extend(f"| {row['check_name']} | {row['status']} | {row.get('details', '')} |" for row in rows)
    write_text(EXP09_OUTPUT_DIR / "sanity_check_exp09_setup.md", "\n".join(lines))
    if failures:
        raise SystemExit(f"Exp9 setup sanity failed. See {relpath(EXP09_OUTPUT_DIR / 'sanity_check_exp09_setup.md')}")


def main() -> None:
    ensure_exp09_dirs()
    rows: list[dict[str, Any]] = []
    rows.extend(dataset_sanity_rows())
    check_config(rows)
    check_py_compile(rows)
    check_scripts(rows)
    splits = load_splits()
    pair_summary = write_pair_outputs(splits["train"], splits["dev"], splits["test"], PairBuildConfig())
    train_pairs = read_pair_shards("train", EXP09_PAIRS_DIR)
    dev_pairs = read_pair_shards("dev", EXP09_PAIRS_DIR)
    check_pair_rows(rows, train_pairs, dev_pairs)
    check_split_leakage(rows, train_pairs)
    check_reuse(rows, train_pairs, dev_pairs)
    weight_rows = check_pointwise_weights(rows)
    toy_debug = check_toy_loss(rows, class_weight_vector(weight_rows))
    pair_inventory = read_csv(EXP09_TABLES_DIR / "pair_inventory.csv")
    add(rows, "pair inventory has train/dev rows", {row["split"] for row in pair_inventory} == {"train", "dev"}, pair_inventory)
    collect()
    check_repository_safety(rows)
    write_setup_report(rows, pair_summary, toy_debug)
    print("Exp9 setup sanity PASS")


if __name__ == "__main__":
    main()
