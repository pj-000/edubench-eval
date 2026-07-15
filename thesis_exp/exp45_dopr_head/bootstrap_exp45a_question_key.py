"""Paired question-key bootstrap for fixed Exp45A comparisons."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from thesis_exp.exp45_dopr_head.common import (
    EXP44_RUN_ROOT,
    FOLDS,
    HEAD_VARIANTS,
    ROOT,
    embedding_path,
    head_prediction_path,
    read_jsonl,
    write_csv,
)


BOOT_METRICS = ("MAE", "QWK", "Exact_Match", "Kendall_tau", "low_to_high_rate", "high_to_low_rate", "label2_recall", "label5_recall")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--replicates", type=int, default=5000)
    return parser.parse_args()


def load_variant(root: Path, variant: str) -> list[dict]:
    if variant == "H0_E4_natural_head":
        return [row for fold in FOLDS for row in read_jsonl(embedding_path(root, fold, "outer_heldout"))]
    return [row for fold in FOLDS for row in read_jsonl(head_prediction_path(root, variant, fold))]


def compact_metrics(gold: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    table = np.bincount((gold - 1) * 5 + (pred - 1), minlength=25).reshape(5, 5).astype(float)
    weights = np.fromfunction(lambda i, j: ((i - j) ** 2) / 16.0, (5, 5), dtype=float)
    expected = np.outer(table.sum(1), table.sum(0)) / max(table.sum(), 1.0)
    denominator = float((weights * expected).sum())
    qwk = float("nan") if denominator == 0 else 1.0 - float((weights * table).sum()) / denominator
    concordant = discordant = 0.0
    for i in range(5):
        for j in range(5):
            concordant += table[i, j] * table[i + 1 :, j + 1 :].sum()
            discordant += table[i, j] * table[i + 1 :, :j].sum()
    tied_gold = sum(n * (n - 1) / 2 for n in table.sum(1))
    tied_pred = sum(n * (n - 1) / 2 for n in table.sum(0))
    tied_both = sum(n * (n - 1) / 2 for n in table.ravel())
    tau_denom = np.sqrt((concordant + discordant + tied_gold - tied_both) * (concordant + discordant + tied_pred - tied_both))
    low, high, label2, label5 = gold <= 2, gold >= 4, gold == 2, gold == 5
    return {
        "MAE": float(np.mean(np.abs(pred - gold))),
        "QWK": qwk,
        "Exact_Match": float(np.mean(pred == gold)),
        "Kendall_tau": float((concordant - discordant) / tau_denom) if tau_denom else float("nan"),
        "low_to_high_rate": float(np.mean(pred[low] >= 4)),
        "high_to_low_rate": float(np.mean(pred[high] <= 2)),
        "label2_recall": float(np.mean(pred[label2] == 2)),
        "label5_recall": float(np.mean(pred[label5] == 5)),
    }


def align(left: list[dict], right: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    right_by_id = {row["sample_id"]: row for row in right}
    if set(right_by_id) != {row["sample_id"] for row in left}:
        raise RuntimeError("Exp45A bootstrap sample IDs do not align")
    if any(
        str(right_by_id[row["sample_id"]]["question_key"]) != str(row["question_key"])
        or int(right_by_id[row["sample_id"]].get("fold", -1)) != int(row.get("fold", -1))
        for row in left
    ):
        raise RuntimeError("Exp45A bootstrap question keys or folds do not align")
    gold = np.asarray([int(row["gold_label_5"]) for row in left], dtype=np.int64)
    left_pred = np.asarray([int(row["pred_label_5"]) for row in left], dtype=np.int64)
    right_pred = np.asarray([int(right_by_id[row["sample_id"]]["pred_label_5"]) for row in left], dtype=np.int64)
    qkeys = np.asarray([str(row["question_key"]) for row in left], dtype=object)
    return gold, left_pred, right_pred, qkeys


def comparison_rows(name: str, left: list[dict], right: list[dict], replicates: int) -> list[dict]:
    gold, left_pred, right_pred, qkeys = align(left, right)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(qkeys):
        groups[key].append(index)
    keys = sorted(groups)
    rng = np.random.default_rng(45_042)
    point_left, point_right = compact_metrics(gold, left_pred), compact_metrics(gold, right_pred)
    samples = {metric: np.empty(replicates, dtype=np.float64) for metric in BOOT_METRICS}
    for replicate in range(replicates):
        selected = rng.choice(keys, size=len(keys), replace=True)
        indices = np.concatenate([np.asarray(groups[key], dtype=np.int64) for key in selected])
        left_metrics = compact_metrics(gold[indices], left_pred[indices])
        right_metrics = compact_metrics(gold[indices], right_pred[indices])
        for metric in BOOT_METRICS:
            samples[metric][replicate] = left_metrics[metric] - right_metrics[metric]
    return [
        {
            "comparison": name,
            "left": "H4_DOPR",
            "right": name.removeprefix("H4_DOPR_vs_"),
            "metric": metric,
            "delta_point": point_left[metric] - point_right[metric],
            "ci_lower": float(np.nanquantile(samples[metric], 0.025)),
            "ci_upper": float(np.nanquantile(samples[metric], 0.975)),
            "replicates": replicates,
            "question_key_groups": len(keys),
            "row_level_available": True,
        }
        for metric in BOOT_METRICS
    ]


def main() -> None:
    args = parse_args()
    if args.replicates != 5000:
        raise ValueError("Exp45A formal bootstrap is locked to 5000 replicates")
    cache = {variant: load_variant(args.out_dir, variant) for variant in HEAD_VARIANTS}
    output = []
    for right in HEAD_VARIANTS[:-1]:
        output.extend(comparison_rows(f"H4_DOPR_vs_{right}", cache["H4_DOPR"], cache[right], args.replicates))
    c1_rows = []
    for fold in FOLDS:
        path = EXP44_RUN_ROOT / f"groupcv/C1_balanced_plain_contrastive/seed_42/fold_{fold}/heldout_predictions.jsonl"
        if not path.exists():
            c1_rows = []
            break
        c1_rows.extend(read_jsonl(path))
    if c1_rows:
        output.extend(comparison_rows("H4_DOPR_vs_Exp44_C1_balanced_plain_contrastive", cache["H4_DOPR"], c1_rows, args.replicates))
    else:
        output.append({"comparison": "H4_DOPR_vs_Exp44_C1_balanced_plain_contrastive", "left": "H4_DOPR", "right": "Exp44_C1_balanced_plain_contrastive", "metric": "ROW_LEVEL_UNAVAILABLE", "delta_point": "", "ci_lower": "", "ci_upper": "", "replicates": 0, "question_key_groups": "", "row_level_available": False})
    write_csv(args.out_dir / "tables/exp45a_question_key_bootstrap_ci.csv", output)
    print({"status": "BOOTSTRAP_COMPLETE", "rows": len(output), "c1_row_level_available": bool(c1_rows), "replicates": args.replicates})


if __name__ == "__main__":
    main()
