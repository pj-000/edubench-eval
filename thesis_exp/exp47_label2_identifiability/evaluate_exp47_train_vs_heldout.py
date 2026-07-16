"""Compare existing outer-train and heldout predictions without retraining."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from thesis_exp.exp47_label2_identifiability.common import (
    MODEL_06B,
    MODEL_4B,
    ROOT,
    ensure_dirs,
    load_06b_oof_predictions,
    load_4b_predictions,
    metric_summary,
    sanitize_rows,
    write_csv,
)


SUBTYPES = ("all_hard_label2", "stable_label2", "strict_label2", "ambiguous_label2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    return parser.parse_args()


def select_subtype(rows: list[dict], subtype: str) -> list[dict]:
    label2 = [row for row in rows if int(row["gold_label_5"]) == 2]
    if subtype == "all_hard_label2":
        return label2
    field = {
        "stable_label2": "stable_label2",
        "strict_label2": "strict_label2",
        "ambiguous_label2": "ambiguous_label2",
    }[subtype]
    return [row for row in label2 if row[field]]


def overall_metric_row(model: str, role: str, fold: int | str, rows: list[dict]) -> dict:
    return {
        "model": model,
        "role": role,
        "fold": fold,
        "availability": "AVAILABLE",
        **metric_summary(rows),
    }


def subtype_metric_row(model: str, role: str, subtype: str, rows: list[dict]) -> dict:
    selected = select_subtype(rows, subtype)
    predictions = Counter(int(row["pred_label_5"]) for row in selected)
    correct = predictions[2]
    return {
        "model": model,
        "role": role,
        "label2_subtype": subtype,
        "availability": "AVAILABLE" if selected else "EMPTY_SUBSET",
        "n": len(selected),
        "label2_correct": correct,
        "label2_recall": correct / len(selected) if selected else None,
        "mean_pred": float(np.mean([row["pred_label_5"] for row in selected])) if selected else None,
        **{f"pred_count_{label}": predictions[label] for label in range(1, 6)},
    }


def main() -> None:
    args = parse_args()
    ensure_dirs(args.out_dir)
    teacher = load_4b_predictions()
    baseline = load_06b_oof_predictions()
    metric_rows = []
    for model, rows, roles in (
        (MODEL_4B, teacher, ("outer_train", "heldout")),
        (MODEL_06B, baseline, ("heldout",)),
    ):
        for role in roles:
            role_rows = [row for row in rows if row["role"] == role]
            for fold in range(5):
                fold_rows = [row for row in role_rows if int(row["fold"]) == fold]
                metric_rows.append(overall_metric_row(model, role, fold, fold_rows))
            metric_rows.append(overall_metric_row(model, role, "pooled", role_rows))
    metric_rows.append(
        {
            "model": MODEL_06B,
            "role": "outer_train",
            "fold": "pooled",
            "availability": "MISSING_CHECKPOINT_NOT_RECOMPUTED",
            "n": 0,
        }
    )
    write_csv(args.out_dir / "tables/exp47_train_vs_heldout_metrics.csv", sanitize_rows(metric_rows))

    subtype_rows = []
    for model, rows, roles in (
        (MODEL_4B, teacher, ("outer_train", "heldout")),
        (MODEL_06B, baseline, ("heldout",)),
    ):
        for role in roles:
            role_rows = [row for row in rows if row["role"] == role]
            for subtype in SUBTYPES:
                subtype_rows.append(subtype_metric_row(model, role, subtype, role_rows))
    for subtype in SUBTYPES:
        subtype_rows.append(
            {
                "model": MODEL_06B,
                "role": "outer_train",
                "label2_subtype": subtype,
                "availability": "MISSING_CHECKPOINT_NOT_RECOMPUTED",
                "n": 0,
            }
        )
    write_csv(args.out_dir / "tables/exp47_label2_subtype_metrics.csv", sanitize_rows(subtype_rows))

    lookup = {(row["model"], row["role"], row["label2_subtype"]): row for row in subtype_rows}
    gap_rows = []
    for model in (MODEL_06B, MODEL_4B):
        for subtype in SUBTYPES:
            train = lookup.get((model, "outer_train", subtype), {})
            heldout = lookup.get((model, "heldout", subtype), {})
            available = train.get("availability") == "AVAILABLE" and heldout.get("availability") == "AVAILABLE"
            train_recall = train.get("label2_recall") if available else None
            heldout_recall = heldout.get("label2_recall")
            gap_rows.append(
                {
                    "model": model,
                    "label2_subtype": subtype,
                    "train_availability": train.get("availability", "MISSING"),
                    "heldout_availability": heldout.get("availability", "MISSING"),
                    "outer_train_n": train.get("n", 0),
                    "outer_heldout_n": heldout.get("n", 0),
                    "outer_train_recall": train_recall,
                    "outer_heldout_recall": heldout_recall,
                    "train_minus_heldout_recall": train_recall - heldout_recall if available and train_recall is not None and heldout_recall is not None else None,
                }
            )
    write_csv(args.out_dir / "tables/exp47_label2_generalization_gap.csv", sanitize_rows(gap_rows))
    print(json.dumps({"status": "TRAIN_HELDOUT_AUDITED", "teacher_prediction_rows": len(teacher), "baseline_oof_rows": len(baseline), "new_training": False, "test_access_count": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
