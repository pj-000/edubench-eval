"""Analyze class-2 probability ranks and competing classes from existing outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from thesis_exp.exp47_label2_identifiability.common import (
    ROOT,
    ensure_dirs,
    load_06b_oof_predictions,
    load_4b_predictions,
    quantile,
    sanitize_rows,
    write_csv,
)


SUBTYPES = ("all_hard_label2", "stable_label2", "strict_label2", "ambiguous_label2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    return parser.parse_args()


def select(rows: list[dict], subtype: str) -> list[dict]:
    rows = [row for row in rows if int(row["gold_label_5"]) == 2]
    if subtype == "all_hard_label2":
        return rows
    return [row for row in rows if row[{"stable_label2": "stable_label2", "strict_label2": "strict_label2", "ambiguous_label2": "ambiguous_label2"}[subtype]]]


def derived(row: dict) -> dict[str, float | int]:
    logits = [float(row[f"logit_{label}"]) for label in range(1, 6)]
    probs = [float(row[f"prob_{label}"]) for label in range(1, 6)]
    z2 = logits[1]
    class2_rank = 1 + sum(value > z2 for value in logits)
    competitors = [logits[0], logits[2], logits[3], logits[4]]
    return {
        "class2_rank": class2_rank,
        "p2": probs[1],
        "margin_vs_best_other": z2 - max(competitors),
        "margin_z2_z1": z2 - logits[0],
        "margin_z2_z3": z2 - logits[2],
        "margin_z2_z4": z2 - logits[3],
        "margin_z2_z5": z2 - logits[4],
        "predicted_class": int(row["pred_label_5"]),
        "expected_score": float(row["pred_score_expected"]),
    }


def main() -> None:
    args = parse_args()
    ensure_dirs(args.out_dir)
    predictions = load_4b_predictions() + load_06b_oof_predictions()
    rank_rows, margin_rows, competing_rows = [], [], []
    groups = sorted({(row["model"], row["role"]) for row in predictions})
    for model, role in groups:
        role_rows = [row for row in predictions if row["model"] == model and row["role"] == role]
        for subtype in SUBTYPES:
            selected = select(role_rows, subtype)
            values = [derived(row) for row in selected]
            if not values:
                continue
            ranks = [int(value["class2_rank"]) for value in values]
            p2 = [float(value["p2"]) for value in values]
            rank_rows.append(
                {
                    "model": model,
                    "role": role,
                    "label2_subtype": subtype,
                    "n": len(values),
                    "logit_source": selected[0]["logit_source"],
                    "class2_rank_mean": float(np.mean(ranks)),
                    "class2_rank_median": float(np.median(ranks)),
                    "class2_rank1_rate": float(np.mean(np.asarray(ranks) == 1)),
                    "class2_top2_rate": float(np.mean(np.asarray(ranks) <= 2)),
                    "p2_mean": float(np.mean(p2)),
                    "p2_median": float(np.median(p2)),
                    "p2_p90": quantile(p2, 0.90),
                }
            )
            margin_rows.append(
                {
                    "model": model,
                    "role": role,
                    "label2_subtype": subtype,
                    "n": len(values),
                    **{
                        f"{key}_{stat}": function([float(value[key]) for value in values])
                        for key in ("margin_vs_best_other", "margin_z2_z1", "margin_z2_z3", "margin_z2_z4", "margin_z2_z5", "expected_score")
                        for stat, function in (("mean", lambda data: float(np.mean(data))), ("median", lambda data: float(np.median(data))))
                    },
                }
            )
            counts = Counter(int(value["predicted_class"]) for value in values)
            for label in range(1, 6):
                competing_rows.append(
                    {
                        "model": model,
                        "role": role,
                        "label2_subtype": subtype,
                        "predicted_class": label,
                        "count": counts[label],
                        "share": counts[label] / len(values),
                    }
                )
    write_csv(args.out_dir / "tables/exp47_label2_logit_rank_summary.csv", sanitize_rows(rank_rows))
    write_csv(args.out_dir / "tables/exp47_label2_margin_summary.csv", sanitize_rows(margin_rows))
    write_csv(args.out_dir / "tables/exp47_label2_competing_class_distribution.csv", competing_rows)
    print(json.dumps({"status": "CLASS2_LOGITS_AUDITED", "aggregate_groups": len(rank_rows), "row_level_outputs_written": False, "test_access_count": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
