"""Analyze frozen Exp38A score-range qualification and enforce the GO gate."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from thesis_exp.exp38_hails_score.common import (
    ROOT,
    TRAIN_PATH,
    frozen_view_map,
    qwk,
    read_jsonl,
    resolve_final_reference,
    sample_id,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--train-jsonl", type=Path, default=TRAIN_PATH)
    parser.add_argument("--final-reference", type=Path)
    return parser.parse_args()


def range_iou(left: tuple[int, int], right: tuple[int, int]) -> float:
    intersection = max(0, min(left[1], right[1]) - max(left[0], right[0]) + 1)
    union = max(left[1], right[1]) - min(left[0], right[0]) + 1
    return intersection / union


def aggregate(rows: list[dict[str, Any]], dimension: str, value: str) -> dict[str, Any]:
    centers = [int(row["qwen_center"]) for row in rows]
    silver = [int(row["silver_center"]) for row in rows]
    widths = [int(row["qwen_max"]) - int(row["qwen_min"]) + 1 for row in rows]
    return {
        "dimension": dimension,
        "value": value,
        "n": len(rows),
        "center_MAE": float(np.mean(np.abs(np.asarray(centers) - np.asarray(silver)))),
        "center_QWK": qwk(silver, centers),
        "silver_point_coverage": float(np.mean([row["qwen_min"] <= row["silver_center"] <= row["qwen_max"] for row in rows])),
        "range_overlap": float(np.mean([max(row["qwen_min"], row["silver_min"]) <= min(row["qwen_max"], row["silver_max"]) for row in rows])),
        "range_IoU": float(np.mean([range_iou((row["qwen_min"], row["qwen_max"]), (row["silver_min"], row["silver_max"])) for row in rows])),
        "mean_width": float(np.mean(widths)),
        "median_width": float(np.median(widths)),
        "non_singleton_rate": float(np.mean(np.asarray(widths) > 1)),
        "schema_success": 1.0,
        "target_scope_success": float(np.mean([row["target_scope"] for row in rows])),
    }


def safety(rows: list[dict[str, Any]], dimension: str, value: str) -> dict[str, Any]:
    definitions = {
        "silver_low_entirely_high": [row for row in rows if row["silver_center"] <= 2],
        "silver_high_entirely_low": [row for row in rows if row["silver_center"] >= 4],
        "human_low_entirely_high": [row for row in rows if row["human_label"] <= 2],
        "human_high_entirely_low": [row for row in rows if row["human_label"] >= 4],
        "human4_qwen_exact5": [row for row in rows if row["human_label"] == 4],
    }
    result: dict[str, Any] = {"dimension": dimension, "value": value, "n": len(rows)}
    checks = {
        "silver_low_entirely_high": lambda row: row["qwen_min"] >= 4,
        "silver_high_entirely_low": lambda row: row["qwen_max"] <= 2,
        "human_low_entirely_high": lambda row: row["qwen_min"] >= 4,
        "human_high_entirely_low": lambda row: row["qwen_max"] <= 2,
        "human4_qwen_exact5": lambda row: row["qwen_min"] == row["qwen_max"] == 5,
    }
    for name, subset in definitions.items():
        count = sum(checks[name](row) for row in subset)
        result[f"{name}_n"] = len(subset)
        result[f"{name}_count"] = count
        result[f"{name}_rate"] = count / len(subset) if subset else float("nan")
    return result


def main() -> None:
    args = parse_args()
    validation = json.loads((args.out_dir / "decision/exp38a_qualification_range_validation.json").read_text(encoding="utf-8"))
    if not validation.get("valid"):
        raise RuntimeError("Qualification outputs did not pass strict validation")
    ranges = {sample_id(row): row for row in read_jsonl(args.out_dir / "parsed_ranges_private/exp38a_qwen_ranges_qualification.jsonl")}
    _, references = resolve_final_reference(explicit=args.final_reference)
    reference = {sample_id(row): row for row in references}
    train = {sample_id(row): row for row in read_jsonl(args.train_jsonl)}
    views = frozen_view_map()
    if set(ranges) != set(reference) or set(ranges) != set(views):
        raise ValueError("Qwen ranges, final reference, and frozen views must have identical IDs")
    aligned = []
    for sid in sorted(ranges):
        qwen, silver, source = ranges[sid], reference[sid], train[sid]
        silver_range = silver["score_range"]
        aligned.append({
            "sample_id": sid, "view": views[sid], "language": str(source.get("language") or "unknown"),
            "metric_group": str(source.get("metric_group") or "unknown"), "human_label": int(source["label_5"]),
            "silver_min": int(silver_range[0]), "silver_center": int(silver["most_plausible_score"]), "silver_max": int(silver_range[-1]),
            "qwen_min": int(qwen["minimum_plausible_score"]), "qwen_center": int(qwen["most_plausible_score"]),
            "qwen_max": int(qwen["maximum_plausible_score"]), "target_scope": qwen["target_scope_confirmed"] is True,
        })
    slices: list[tuple[str, str, list[dict[str, Any]]]] = [("overall", "all", aligned)]
    for dimension in ("view", "language", "metric_group"):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in aligned:
            grouped[str(row[dimension])].append(row)
        slices.extend((dimension, value, rows) for value, rows in sorted(grouped.items()))
    metric_rows = [aggregate(rows, dimension, value) for dimension, value, rows in slices]
    safety_rows = [safety(rows, dimension, value) for dimension, value, rows in slices]
    write_csv(args.out_dir / "tables/exp38a_range_qualification_metrics.csv", metric_rows)
    write_csv(args.out_dir / "tables/exp38a_range_direction_safety.csv", safety_rows)
    overall, direction = metric_rows[0], safety_rows[0]
    gates = {
        "schema_success": overall["schema_success"] == 1.0,
        "target_scope_success": overall["target_scope_success"] == 1.0,
        "center_mae": overall["center_MAE"] <= 0.55,
        "center_qwk": overall["center_QWK"] >= 0.55,
        "silver_point_coverage": overall["silver_point_coverage"] >= 0.85,
        "range_overlap": overall["range_overlap"] >= 0.80,
        "mean_width": overall["mean_width"] <= 2.0,
        "non_singleton_rate": 0.15 <= overall["non_singleton_rate"] <= 0.80,
        "silver_low_direction": direction["silver_low_entirely_high_rate"] <= 0.10,
        "silver_high_direction": direction["silver_high_entirely_low_rate"] <= 0.10,
        "human_low_direction": direction["human_low_entirely_high_rate"] <= 0.15,
        "human_high_direction": direction["human_high_entirely_low_rate"] <= 0.10,
    }
    go = all(gates.values())
    decision = {
        "status": "RANGE_PROTOCOL_QUALIFIED" if go else "RANGE_PROTOCOL_NOT_QUALIFIED",
        "gates": gates,
        "recommend_full_train_range_annotation": go,
        "recommend_groupcv_training": go,
        "prompt_reuse_on_same_196_forbidden": True,
        "dev_access_count": 0,
        "test_access_count": 0,
    }
    write_json(args.out_dir / "decision/exp38a_range_qualification_decision.json", decision)
    report = [
        "# Exp38A Qwen score-range qualification", "",
        f"- Status: **{decision['status']}**",
        f"- Rows: {overall['n']}",
        f"- Center MAE / QWK: {overall['center_MAE']:.4f} / {overall['center_QWK']:.4f}",
        f"- Silver point coverage / range overlap: {overall['silver_point_coverage']:.4f} / {overall['range_overlap']:.4f}",
        f"- Mean / median width: {overall['mean_width']:.4f} / {overall['median_width']:.4f}",
        f"- Non-singleton rate: {overall['non_singleton_rate']:.4f}",
        f"- Gate checks: `{json.dumps(gates, sort_keys=True)}`",
        f"- Full-train annotation recommended: `{str(go).lower()}`",
        "- This frozen 196 set must not be reused to tune the prompt after observing results.",
        "- No reason/failure supervision, student training, dev, or test access occurred.",
    ]
    report_path = args.out_dir / "reports/exp38a_range_qualification_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
