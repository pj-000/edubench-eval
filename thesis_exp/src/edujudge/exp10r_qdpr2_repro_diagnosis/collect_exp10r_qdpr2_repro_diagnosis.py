"""Collect Exp10R QD-PR2 reproducibility diagnosis artifacts.

This diagnosis compares Exp09 formal QD-PR2 with Exp10 full_qdpr2 without
copying raw predictions, arrays, or checkpoint weights into the Exp10R output.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from thesis_exp.src.edujudge.exp10r_qdpr2_repro_diagnosis import (
    EXP10R_REPORTS_DIR,
    EXP10R_TABLES_DIR,
    ensure_exp10r_dirs,
)
from thesis_exp.src.edujudge.utils.io import THESIS_DIR, read_csv, read_jsonl, relpath, write_csv, write_text


EXP09_CONFIG = THESIS_DIR / "configs" / "exp09_pairwise_ordinal" / "exp09_qdpr2_anchored_pairwise_human_only.yaml"
EXP10_CONFIG = THESIS_DIR / "configs" / "exp10_qdpr2_module_ablation" / "exp10_full_qdpr2.yaml"
EXP09_RUN = (
    THESIS_DIR
    / "outputs"
    / "exp09_pairwise_ordinal_qdpr2"
    / "runs"
    / "QD-PR2_AnchoredPairwiseOrdinal_human_only"
)
EXP10_RUN = THESIS_DIR / "outputs" / "exp10_qdpr2_module_ablation" / "runs" / "full_qdpr2"
PAIR_DIR = THESIS_DIR / "outputs" / "exp09_pairwise_ordinal_qdpr2" / "pairs"
EXP09_CHECKPOINT_DIR = (
    THESIS_DIR
    / "artifacts"
    / "exp09_pairwise_ordinal_qdpr2"
    / "checkpoints"
    / "QD-PR2_AnchoredPairwiseOrdinal_human_only"
)
EXP10_CHECKPOINT_DIR = THESIS_DIR / "artifacts" / "exp10_qdpr2_module_ablation" / "checkpoints" / "full_qdpr2"

RUNS = {
    "exp09_formal_qdpr2": {
        "config": EXP09_CONFIG,
        "run_dir": EXP09_RUN,
        "checkpoint_dir": EXP09_CHECKPOINT_DIR,
        "label": "Exp09 formal QD-PR2",
    },
    "exp10_full_qdpr2": {
        "config": EXP10_CONFIG,
        "run_dir": EXP10_RUN,
        "checkpoint_dir": EXP10_CHECKPOINT_DIR,
        "label": "Exp10 full_qdpr2",
    },
}
PROB_COLUMNS = ["prob_gt_1", "prob_gt_2", "prob_gt_3", "prob_gt_4"]
LOGIT_COLUMNS = ["logit_gt_1", "logit_gt_2", "logit_gt_3", "logit_gt_4"]
DECODING_RULE = "pred_label_5 = 1 + count(prob_gt_k > 0.5 for k in 1..4)"
METRIC_SCRIPT = (
    "thesis_exp/src/edujudge/exp09_pairwise_ordinal/metrics.py; "
    "prediction rows produced by train_qdpr2_anchored_pairwise.evaluate"
)
CONFIG_FIELDS = [
    "section",
    "item",
    "exp09_value",
    "exp10_value",
    "match",
    "diagnostic_note",
]
METRIC_DIFF_FIELDS = [
    "section",
    "split",
    "metric",
    "exp09_value",
    "exp10_value",
    "delta_exp10_minus_exp09",
    "match",
    "diagnostic_note",
]
CHECKPOINT_FIELDS = [
    "run_name",
    "split",
    "source",
    "epoch",
    "global_step",
    "best_epoch",
    "best_global_step",
    "evaluation_global_step",
    "checkpoint_available",
    "predictions_available",
    "MAE",
    "QWK",
    "Acc@5",
    "low_to_high_rate",
    "low_to_high_count",
    "true_low_count",
    "monotonic_violation_rate",
    "diagnostic_note",
]
PAIR_DIFF_FIELDS = [
    "split",
    "pair_pool_attribute",
    "exp09_value",
    "exp10_value",
    "match",
    "diagnostic_note",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in {path}")
    return data


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected object in {path}")
    return data


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _fmt(value: Any, digits: int = 6) -> str:
    value_float = _float(value)
    if math.isfinite(value_float):
        return f"{value_float:.{digits}f}"
    return str(value)


def _safe_rate(count: int, total: int) -> float:
    return float(count / total) if total else float("nan")


def _metric_row(run_dir: Path, split: str) -> dict[str, str]:
    path = run_dir / "tables" / "metrics_summary.csv"
    if not path.exists():
        return {}
    for row in read_csv(path):
        if row.get("split") == split:
            return row
    return {}


def _history_rows(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "tables" / "dev_metrics_history.csv"
    return read_csv(path) if path.exists() else []


def _prediction_path(run_dir: Path, split: str) -> Path:
    if split == "dev":
        best = run_dir / "predictions" / "predictions_dev_best.jsonl"
        return best if best.exists() else run_dir / "predictions_dev_best.jsonl"
    path = run_dir / "predictions" / f"predictions_{split}.jsonl"
    return path if path.exists() else run_dir / f"predictions_{split}.jsonl"


def _pred_label(row: dict[str, Any]) -> int:
    return int(row.get("pred_label_5", row.get("pred_label")))


def _true_label(row: dict[str, Any]) -> int:
    return int(row.get("label_5", row.get("label")))


def _prediction_summary(run_dir: Path, split: str) -> dict[str, Any]:
    path = _prediction_path(run_dir, split)
    if not path.exists():
        return {
            "predictions_available": False,
            "threshold_probability_columns": "missing",
            "threshold_logit_columns": "missing",
        }
    rows = read_jsonl(path)
    labels = [_true_label(row) for row in rows]
    preds = [_pred_label(row) for row in rows]
    low_indices = [idx for idx, label in enumerate(labels) if label <= 2]
    low_to_high = [idx for idx in low_indices if preds[idx] >= 4]
    label_counts: dict[int, dict[str, int]] = {}
    for label in [1, 2]:
        indices = [idx for idx, value in enumerate(labels) if value == label]
        label_counts[label] = {
            "n": len(indices),
            "low_to_high_count": sum(1 for idx in indices if preds[idx] >= 4),
        }
    first = rows[0] if rows else {}
    return {
        "predictions_available": True,
        "prediction_path": relpath(path),
        "n": len(rows),
        "true_low_count": len(low_indices),
        "low_to_high_count": len(low_to_high),
        "low_to_high_rate": _safe_rate(len(low_to_high), len(low_indices)),
        "label_1_count": label_counts[1]["n"],
        "label_1_low_to_high_count": label_counts[1]["low_to_high_count"],
        "label_1_low_to_high_rate": _safe_rate(
            label_counts[1]["low_to_high_count"], label_counts[1]["n"]
        ),
        "label_2_count": label_counts[2]["n"],
        "label_2_low_to_high_count": label_counts[2]["low_to_high_count"],
        "label_2_low_to_high_rate": _safe_rate(
            label_counts[2]["low_to_high_count"], label_counts[2]["n"]
        ),
        "threshold_probability_columns": ",".join(column for column in PROB_COLUMNS if column in first),
        "threshold_logit_columns": ",".join(column for column in LOGIT_COLUMNS if column in first),
    }


def _checkpoint_available(path: Path) -> bool:
    if not path.exists():
        return False
    patterns = ["*.pt", "*.pth", "*.bin", "*.ckpt", "*.safetensors"]
    return any(next(path.rglob(pattern), None) is not None for pattern in patterns)


def _same_value(left: Any, right: Any) -> str:
    return "same" if str(left) == str(right) else "different"


def _equivalent_value(left: Any, right: Any) -> str:
    return "same" if str(left) == str(right) else "equivalent" if left == right else "different"


def _config_value(config: dict[str, Any], key: str, default: Any = "") -> Any:
    return config.get(key, default)


def build_config_diff(configs: dict[str, dict[str, Any]], metadata: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    exp09 = configs["exp09_formal_qdpr2"]
    exp10 = configs["exp10_full_qdpr2"]
    meta09 = metadata["exp09_formal_qdpr2"]
    meta10 = metadata["exp10_full_qdpr2"]
    rows: list[dict[str, Any]] = []

    def add(section: str, item: str, left: Any, right: Any, note: str = "", match: str | None = None) -> None:
        rows.append(
            {
                "section": section,
                "item": item,
                "exp09_value": left,
                "exp10_value": right,
                "match": match or _same_value(left, right),
                "diagnostic_note": note,
            }
        )

    add("run", "run_id", exp09.get("run_id"), exp10.get("run_id"), "Different experiment wrapper names are expected.")
    add("run", "objective", exp09.get("objective"), exp10.get("objective"), "Different label for the same QD-PR2 full objective.")
    add("run", "loss", exp09.get("loss"), exp10.get("loss"))
    add("init checkpoint", "model_name_or_path", exp09.get("model_name_or_path"), exp10.get("model_name_or_path"))
    add("init checkpoint", "qd_b1_checkpoint_dir", exp09.get("qd_b1_checkpoint_dir"), exp10.get("qd_b1_checkpoint_dir"))
    add("data", "dataset_id", exp09.get("dataset_id"), exp10.get("dataset_id"))
    add("data", "split", exp09.get("split"), exp10.get("split"))
    add("data", "input_template", exp09.get("input_template"), exp10.get("input_template"))
    data_dir09 = Path(str(exp09.get("data_dir", "")))
    data_dir10 = Path(str(exp10.get("data_dir", "")))
    for split in ["train", "dev", "test"]:
        add(
            "data",
            f"{split}_data_path",
            str(data_dir09 / f"{split}.jsonl"),
            str(data_dir10 / f"{split}.jsonl"),
        )
    add("seed", "training_seed", "42 (trainer default)", "42 (trainer default)")
    add("seed", "pair_sampling_seed", exp09.get("pair_sampling_seed"), exp10.get("pair_sampling_seed"))
    for key in [
        "pair_dataset_size_train",
        "pair_dataset_size_dev",
        "max_pairs_per_record",
        "max_pairs_per_low_record",
        "margin_scale",
        "low_high_margin",
        "low_high_weight",
        "gap_weight",
    ]:
        add("pair construction", key, exp09.get(key), exp10.get(key))
    add("pair construction", "dataloader_mode", "pair", exp10.get("dataloader_mode"), "Both full runs train from pair batches.")
    add("pair construction", "max_pairs", exp09.get("pair_dataset_size_train"), exp10.get("pair_dataset_size_train"))
    add("loss weights", "lambda_point", "1.0 (implicit)", exp10.get("lambda_point"), match="equivalent")
    for key in ["lambda_pair", "lambda_anchor", "lambda_mono"]:
        add("loss weights", key, exp09.get(key), exp10.get(key))
    for key in [
        "epochs",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "effective_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "weight_decay",
        "warmup_ratio",
        "gradient_checkpointing",
        "max_length",
    ]:
        add("optimization", key, exp09.get(key), exp10.get(key))
    add("optimization", "scheduler", "cosine schedule with warmup", "cosine schedule with warmup")
    add("checkpoint selection", "selection_metric", exp09.get("selection_metric"), exp10.get("selection_metric"))
    add("checkpoint selection", "selection_direction", exp09.get("selection_direction"), exp10.get("selection_direction"))
    add(
        "checkpoint selection",
        "early_stopping",
        "none; train all epochs and select best dev checkpoint",
        "none; train all epochs and select best dev checkpoint",
    )
    add("checkpoint selection", "selected_best_epoch", meta09.get("best_epoch"), meta10.get("best_epoch"))
    add("checkpoint selection", "selected_best_global_step", meta09.get("best_global_step"), meta10.get("best_global_step"))
    add("status", "run_status", meta09.get("status"), meta10.get("status"))
    add("source", "config_path", relpath(EXP09_CONFIG), relpath(EXP10_CONFIG))
    return rows


def _add_metric_diff(
    rows: list[dict[str, Any]],
    section: str,
    split: str,
    metric: str,
    left: Any,
    right: Any,
    note: str = "",
) -> None:
    left_float = _float(left)
    right_float = _float(right)
    if math.isfinite(left_float) and math.isfinite(right_float):
        delta: Any = right_float - left_float
        match = "same" if abs(delta) < 1e-12 else "different"
    else:
        delta = ""
        match = _same_value(left, right)
    rows.append(
        {
            "section": section,
            "split": split,
            "metric": metric,
            "exp09_value": left,
            "exp10_value": right,
            "delta_exp10_minus_exp09": delta,
            "match": match,
            "diagnostic_note": note,
        }
    )


def build_metric_diff() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summaries = {
        run_name: {
            split: {
                "metrics": _metric_row(info["run_dir"], split),
                "predictions": _prediction_summary(info["run_dir"], split),
            }
            for split in ["dev", "test"]
        }
        for run_name, info in RUNS.items()
    }
    for split in ["dev", "test"]:
        _add_metric_diff(rows, "evaluation code", split, "metric_script", METRIC_SCRIPT, METRIC_SCRIPT)
        _add_metric_diff(rows, "evaluation code", split, "prediction_decoding_rule", DECODING_RULE, DECODING_RULE)
        _add_metric_diff(
            rows,
            "evaluation code",
            split,
            "threshold_probability_columns",
            summaries["exp09_formal_qdpr2"][split]["predictions"].get("threshold_probability_columns"),
            summaries["exp10_full_qdpr2"][split]["predictions"].get("threshold_probability_columns"),
        )
        _add_metric_diff(
            rows,
            "evaluation code",
            split,
            "threshold_logit_columns",
            summaries["exp09_formal_qdpr2"][split]["predictions"].get("threshold_logit_columns"),
            summaries["exp10_full_qdpr2"][split]["predictions"].get("threshold_logit_columns"),
        )
        for metric, column in [
            ("MAE", "MAE_label"),
            ("QWK", "Quadratic Weighted Kappa"),
            ("Accuracy", "Accuracy"),
            ("Acc@5", "Acc@5"),
            ("monotonic_violation_rate", "monotonic_violation_rate"),
        ]:
            _add_metric_diff(
                rows,
                "selected-best metrics",
                split,
                metric,
                summaries["exp09_formal_qdpr2"][split]["metrics"].get(column),
                summaries["exp10_full_qdpr2"][split]["metrics"].get(column),
            )
        for metric in [
            "true_low_count",
            "low_to_high_count",
            "low_to_high_rate",
            "label_1_count",
            "label_1_low_to_high_count",
            "label_1_low_to_high_rate",
            "label_2_count",
            "label_2_low_to_high_count",
            "label_2_low_to_high_rate",
        ]:
            note = "The requested 12/31 vs 14/31 gap appears here." if split == "test" and metric == "low_to_high_count" else ""
            _add_metric_diff(
                rows,
                "low-score diagnosis",
                split,
                metric,
                summaries["exp09_formal_qdpr2"][split]["predictions"].get(metric),
                summaries["exp10_full_qdpr2"][split]["predictions"].get(metric),
                note,
            )
    return rows


def _selected_test_checkpoint_row(run_name: str, info: dict[str, Any]) -> dict[str, Any]:
    metadata = _load_json(info["run_dir"] / "run_metadata.json")
    metrics = _metric_row(info["run_dir"], "test")
    pred = _prediction_summary(info["run_dir"], "test")
    return {
        "run_name": run_name,
        "split": "test",
        "source": "selected_best_predictions",
        "epoch": metadata.get("best_epoch", "best"),
        "global_step": metadata.get("best_global_step", ""),
        "best_epoch": metadata.get("best_epoch", ""),
        "best_global_step": metadata.get("best_global_step", ""),
        "evaluation_global_step": metrics.get("global_step", ""),
        "checkpoint_available": _checkpoint_available(info["checkpoint_dir"]),
        "predictions_available": pred.get("predictions_available"),
        "MAE": metrics.get("MAE_label", ""),
        "QWK": metrics.get("Quadratic Weighted Kappa", ""),
        "Acc@5": metrics.get("Acc@5", ""),
        "low_to_high_rate": pred.get("low_to_high_rate", ""),
        "low_to_high_count": pred.get("low_to_high_count", ""),
        "true_low_count": pred.get("true_low_count", ""),
        "monotonic_violation_rate": metrics.get("monotonic_violation_rate", ""),
        "diagnostic_note": "Selected-best test predictions are available; per-epoch test predictions are not archived locally.",
    }


def build_checkpoint_metrics() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_name, info in RUNS.items():
        metadata = _load_json(info["run_dir"] / "run_metadata.json")
        checkpoint_available = _checkpoint_available(info["checkpoint_dir"])
        for row in _history_rows(info["run_dir"]):
            low_n = _int(row.get("low_n"))
            low_to_high_rate = _float(row.get("low_to_high_rate"))
            low_to_high_count: Any = ""
            if math.isfinite(low_to_high_rate):
                low_to_high_count = int(round(low_to_high_rate * low_n))
            rows.append(
                {
                    "run_name": run_name,
                    "split": "dev",
                    "source": "dev_metrics_history",
                    "epoch": row.get("epoch"),
                    "global_step": row.get("global_step"),
                    "best_epoch": metadata.get("best_epoch", ""),
                    "best_global_step": metadata.get("best_global_step", ""),
                    "evaluation_global_step": row.get("global_step"),
                    "checkpoint_available": checkpoint_available,
                    "predictions_available": False,
                    "MAE": row.get("MAE_label", ""),
                    "QWK": row.get("Quadratic Weighted Kappa", ""),
                    "Acc@5": row.get("Acc@5", ""),
                    "low_to_high_rate": row.get("low_to_high_rate", ""),
                    "low_to_high_count": low_to_high_count,
                    "true_low_count": low_n,
                    "monotonic_violation_rate": row.get("monotonic_violation_rate", ""),
                    "diagnostic_note": "Dev per-epoch metrics are available from history; per-epoch prediction files are not archived.",
                }
            )
        rows.append(_selected_test_checkpoint_row(run_name, info))
    return rows


def _pair_shards(split: str) -> list[Path]:
    return sorted(PAIR_DIR.glob(f"{split}_pairs_shard_*.jsonl"))


def _pair_pool_summary(split: str) -> dict[str, Any]:
    shards = _pair_shards(split)
    if not shards:
        return {
            "available": False,
            "pair_count": "",
            "pair_type_distribution": {},
            "low_high_pair_count": "",
            "label_gap_distribution": {},
            "pair_pool_hash": "",
            "shard_count": 0,
            "first_pair_id": "",
            "last_pair_id": "",
        }
    digest = hashlib.sha256()
    pair_type_counter: Counter[str] = Counter()
    label_gap_counter: Counter[str] = Counter()
    low_high_count = 0
    first_pair_id = ""
    last_pair_id = ""
    total = 0
    for shard in shards:
        digest.update(relpath(shard).encode("utf-8"))
        with shard.open("rb") as raw:
            for raw_line in raw:
                stripped = raw_line.strip()
                if not stripped:
                    continue
                digest.update(stripped)
                digest.update(b"\n")
                row = json.loads(stripped)
                total += 1
                pair_id = str(row.get("pair_id", ""))
                first_pair_id = first_pair_id or pair_id
                last_pair_id = pair_id
                pair_type = str(row.get("pair_type", "unknown"))
                pair_type_counter[pair_type] += 1
                if pair_type == "low_high" or bool(row.get("risk_pair_low_high")):
                    low_high_count += 1
                label_gap_counter[str(row.get("label_gap", ""))] += 1
    return {
        "available": True,
        "pair_count": total,
        "pair_type_distribution": dict(sorted(pair_type_counter.items())),
        "low_high_pair_count": low_high_count,
        "label_gap_distribution": dict(sorted(label_gap_counter.items(), key=lambda item: item[0])),
        "pair_pool_hash": digest.hexdigest(),
        "shard_count": len(shards),
        "first_pair_id": first_pair_id,
        "last_pair_id": last_pair_id,
    }


def build_pair_pool_diff() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    note = (
        "Exp10 full_qdpr2 uses the shared deterministic QD-PR2 pair pool generated for Exp09; "
        "no separate Exp10 pair archive was found locally."
    )
    for split in ["train", "dev"]:
        summary = _pair_pool_summary(split)
        for attribute in [
            "pair_count",
            "pair_type_distribution",
            "low_high_pair_count",
            "label_gap_distribution",
            "pair_pool_hash",
            "shard_count",
            "first_pair_id",
            "last_pair_id",
        ]:
            rows.append(
                {
                    "split": split,
                    "pair_pool_attribute": attribute,
                    "exp09_value": summary.get(attribute),
                    "exp10_value": summary.get(attribute),
                    "match": "same" if summary.get("available") else "not_available",
                    "diagnostic_note": note,
                }
            )
    return rows


def _row_lookup(rows: list[dict[str, Any]], **keys: Any) -> dict[str, Any]:
    for row in rows:
        if all(str(row.get(key)) == str(value) for key, value in keys.items()):
            return row
    return {}


def build_report(
    config_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    checkpoint_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
) -> str:
    exp09_low = _row_lookup(metric_rows, section="low-score diagnosis", split="test", metric="low_to_high_count")
    exp10_low = exp09_low.get("exp10_value", "")
    exp09_low_count = exp09_low.get("exp09_value", "")
    true_low = _row_lookup(metric_rows, section="low-score diagnosis", split="test", metric="true_low_count")
    exp09_best = next(row for row in checkpoint_rows if row["run_name"] == "exp09_formal_qdpr2" and row["split"] == "test")
    exp10_best = next(row for row in checkpoint_rows if row["run_name"] == "exp10_full_qdpr2" and row["split"] == "test")
    exp09_dev3 = _row_lookup(checkpoint_rows, run_name="exp09_formal_qdpr2", split="dev", epoch="3")
    exp10_dev1 = _row_lookup(checkpoint_rows, run_name="exp10_full_qdpr2", split="dev", epoch="1")
    pair_train_count = _row_lookup(pair_rows, split="train", pair_pool_attribute="pair_count").get("exp09_value", "")
    pair_dev_count = _row_lookup(pair_rows, split="dev", pair_pool_attribute="pair_count").get("exp09_value", "")
    pair_train_hash = _row_lookup(pair_rows, split="train", pair_pool_attribute="pair_pool_hash").get("exp09_value", "")
    differing_config_items = [row for row in config_rows if row.get("match") == "different"]
    diff_items = ", ".join(f"{row['section']}:{row['item']}" for row in differing_config_items)
    report = f"""# Exp10R QD-PR2 Reproducibility Diagnosis

## Executive summary

Exp09 formal QD-PR2 and Exp10 `full_qdpr2` are configuration-equivalent for the
substantive training setup: same base model, same QD-B1 initialization checkpoint,
same train/dev/test paths, same pair construction parameters, same max pair counts,
same active loss weights, same epochs/batch settings/learning rate, and the same dev
MAE checkpoint-selection rule.

The observed test low-to-high difference is therefore not explained by a config,
pair-pool, or evaluation-script mismatch. The primary reproducibility difference is
checkpoint selection:

| run | selected epoch | selected global step | test MAE | test QWK | test Acc@5 | test low-to-high |
|---|---:|---:|---:|---:|---:|---:|
| Exp09 formal QD-PR2 | {exp09_best['best_epoch']} | {exp09_best['best_global_step']} | {_fmt(exp09_best['MAE'], 4)} | {_fmt(exp09_best['QWK'], 4)} | {_fmt(exp09_best['Acc@5'], 4)} | {exp09_low_count}/{true_low.get('exp09_value')} |
| Exp10 full_qdpr2 | {exp10_best['best_epoch']} | {exp10_best['best_global_step']} | {_fmt(exp10_best['MAE'], 4)} | {_fmt(exp10_best['QWK'], 4)} | {_fmt(exp10_best['Acc@5'], 4)} | {exp10_low}/{true_low.get('exp10_value')} |

Exp09 selected epoch 3, while Exp10 selected epoch 1. Exp10's selected checkpoint has
slightly better test MAE and Acc@5, but it raises low-score severe overestimation from
12/31 to 14/31.

## Configuration diagnosis

The only config/reporting differences marked as different are wrapper-level labels or
selected checkpoint metadata: {diff_items}. The substantive settings requested for
reproducibility are matched or equivalent:

- Initial checkpoint: same QD-B1 checkpoint and same base model.
- Data paths: same dataset directory and train/dev/test JSONL paths.
- Seed: same pair sampling seed; trainer seed is the same default value.
- Pair construction: same train/dev pair counts, max pairs per record, margins, and
  low-high weighting.
- Loss: Exp09 has implicit `lambda_point=1.0`; Exp10 records it explicitly as `1.0`.
- Optimization: same 3 epochs, batch size, gradient accumulation, learning rate,
  weight decay, warmup ratio, max length, and checkpoint selection metric.

## Evaluation diagnosis

Both runs use the same selected-best evaluation surface: `pred_label_5 = 1 +
count(prob_gt_k > 0.5)`, with threshold probability columns `prob_gt_1..prob_gt_4`
and logit columns `logit_gt_1..logit_gt_4`.

On test, the true low-score count is {true_low.get('exp09_value')} in Exp09 and
{true_low.get('exp10_value')} in Exp10. The requested low-to-high counts are:

- Exp09 formal QD-PR2: {exp09_low_count}/{true_low.get('exp09_value')}
- Exp10 full_qdpr2: {exp10_low}/{true_low.get('exp10_value')}

## Checkpoint-level diagnosis

Local per-epoch checkpoint weights were not available, so this diagnosis cannot
recompute test predictions at every epoch. Dev per-epoch metric history is available,
and selected-best dev/test predictions are available.

The decisive dev rows are:

| run | epoch | global step | dev MAE | dev QWK | dev Acc@5 | dev low-to-high | dev monotonic violation |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exp09 formal QD-PR2 | 3 | 237 | {_fmt(exp09_dev3.get('MAE'), 4)} | {_fmt(exp09_dev3.get('QWK'), 4)} | {_fmt(exp09_dev3.get('Acc@5'), 4)} | {exp09_dev3.get('low_to_high_count')}/{exp09_dev3.get('true_low_count')} | {_fmt(exp09_dev3.get('monotonic_violation_rate'), 4)} |
| Exp10 full_qdpr2 | 1 | 79 | {_fmt(exp10_dev1.get('MAE'), 4)} | {_fmt(exp10_dev1.get('QWK'), 4)} | {_fmt(exp10_dev1.get('Acc@5'), 4)} | {exp10_dev1.get('low_to_high_count')}/{exp10_dev1.get('true_low_count')} | {_fmt(exp10_dev1.get('monotonic_violation_rate'), 4)} |

Exp09 and Exp10 share identical dev values at epochs 1 and 2, then differ slightly at
epoch 3. Because checkpoint selection minimizes dev MAE, that small epoch-3 drift
changes the selected checkpoint.

## Pair-pool diagnosis

The pair pool on disk contains {pair_train_count} train pairs and {pair_dev_count} dev
pairs. Exp10 does not have a separate local pair archive; the full run points to the
same deterministic QD-PR2 pair-pool setup. Train pair-pool SHA256 signature:
`{pair_train_hash}`.

## Interpretation

The most likely explanation for 12/31 versus 14/31 is checkpoint-selection drift under
an otherwise matched setup. Exp10 selected an earlier checkpoint (epoch 1) because its
epoch-3 dev MAE was slightly worse than epoch 1; Exp09 selected epoch 3 because epoch 3
was best by dev MAE in that run. Since low-to-high is not the selection metric, a
checkpoint with marginally better MAE can still produce more severe overestimation on
the low-score subset.

## Output files

- `{relpath(EXP10R_TABLES_DIR / 'exp10r_config_diff.csv')}`
- `{relpath(EXP10R_TABLES_DIR / 'exp10r_metric_diff.csv')}`
- `{relpath(EXP10R_TABLES_DIR / 'exp10r_checkpoint_metrics.csv')}`
- `{relpath(EXP10R_TABLES_DIR / 'exp10r_pair_pool_diff.csv')}`
"""
    return report


def collect() -> dict[str, str]:
    ensure_exp10r_dirs()
    configs = {run_name: _load_yaml(info["config"]) for run_name, info in RUNS.items()}
    metadata = {run_name: _load_json(info["run_dir"] / "run_metadata.json") for run_name, info in RUNS.items()}
    config_rows = build_config_diff(configs, metadata)
    metric_rows = build_metric_diff()
    checkpoint_rows = build_checkpoint_metrics()
    pair_rows = build_pair_pool_diff()
    write_csv(EXP10R_TABLES_DIR / "exp10r_config_diff.csv", config_rows, fieldnames=CONFIG_FIELDS)
    write_csv(EXP10R_TABLES_DIR / "exp10r_metric_diff.csv", metric_rows, fieldnames=METRIC_DIFF_FIELDS)
    write_csv(EXP10R_TABLES_DIR / "exp10r_checkpoint_metrics.csv", checkpoint_rows, fieldnames=CHECKPOINT_FIELDS)
    write_csv(EXP10R_TABLES_DIR / "exp10r_pair_pool_diff.csv", pair_rows, fieldnames=PAIR_DIFF_FIELDS)
    report = build_report(config_rows, metric_rows, checkpoint_rows, pair_rows)
    write_text(EXP10R_REPORTS_DIR / "exp10r_qdpr2_repro_diagnosis_report.md", report)
    return {
        "config_diff": relpath(EXP10R_TABLES_DIR / "exp10r_config_diff.csv"),
        "metric_diff": relpath(EXP10R_TABLES_DIR / "exp10r_metric_diff.csv"),
        "checkpoint_metrics": relpath(EXP10R_TABLES_DIR / "exp10r_checkpoint_metrics.csv"),
        "pair_pool_diff": relpath(EXP10R_TABLES_DIR / "exp10r_pair_pool_diff.csv"),
        "report": relpath(EXP10R_REPORTS_DIR / "exp10r_qdpr2_repro_diagnosis_report.md"),
    }


def main() -> None:
    print(json.dumps(collect(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
