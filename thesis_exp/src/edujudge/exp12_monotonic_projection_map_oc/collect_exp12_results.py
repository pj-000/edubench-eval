"""Collect Exp12 monotonic projection / MAP-OC lightweight results."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp12_monotonic_projection_map_oc import (
    DEFAULT_SELECTION_DELTA,
    DEFAULT_SELECTION_RULE,
    EXP12_LOCAL_RUNS_DIR,
    EXP12_OUTPUT_DIR,
    EXP12_REPORTS_DIR,
    EXP12_TABLES_DIR,
    EXP12B_RUNS,
    ensure_exp12_dirs,
)
from thesis_exp.src.edujudge.utils.io import read_csv, relpath, write_csv, write_text


METRIC_FIELDS = [
    "source",
    "run_name",
    "seed",
    "selection_rule",
    "selected_epoch",
    "selected_global_step",
    "epoch",
    "global_step",
    "split",
    "diagnostic",
    "decode_mode",
    "selection_decode_mode",
    "uses_test_for_selection",
    "MAE",
    "QWK",
    "Accuracy",
    "Acc@5",
    "low_to_high",
    "low_to_high_count",
    "true_low_score_count",
    "label1_low_to_high",
    "label1_low_to_high_count",
    "label2_low_to_high",
    "label2_low_to_high_count",
    "monotonic_violation",
    "p1_lt_p2",
    "p2_lt_p3",
    "p3_lt_p4",
    "mean_projection_l2_delta",
    "mean_projection_linf_delta",
    "low_score_mean_projection_l2_delta",
]

EFFECT_FIELDS = [
    "source",
    "run_name",
    "seed",
    "epoch",
    "global_step",
    "split",
    "raw_low_to_high_count",
    "projected_low_to_high_count",
    "delta_low_to_high_count",
    "raw_low_to_high",
    "projected_low_to_high",
    "delta_low_to_high",
    "raw_MAE",
    "projected_MAE",
    "delta_MAE",
    "raw_QWK",
    "projected_QWK",
    "delta_QWK",
    "raw_Acc@5",
    "projected_Acc@5",
    "delta_Acc@5",
    "raw_monotonic_violation",
    "projected_monotonic_violation",
    "delta_monotonic_violation",
]

SELECTED_FIELDS = [
    "run_name",
    "seed",
    "selection_rule",
    "selection_decode_mode",
    "uses_test_for_selection",
    "selected_epoch",
    "selected_global_step",
    "decode_mode",
    "test_MAE",
    "test_QWK",
    "test_Accuracy",
    "test_Acc@5",
    "test_low_to_high",
    "test_low_to_high_count",
    "test_true_low_score_count",
    "test_label1_low_to_high",
    "test_label1_low_to_high_count",
    "test_label2_low_to_high",
    "test_label2_low_to_high_count",
    "test_monotonic_violation",
    "test_p1_lt_p2",
    "test_p2_lt_p3",
    "test_p3_lt_p4",
]


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


def _fmt(value: Any, digits: int = 4) -> str:
    value = _float(value)
    return f"{value:.{digits}f}" if math.isfinite(value) else "NA"


def _mean(values: Any) -> float:
    nums = [_float(value) for value in values]
    nums = [value for value in nums if math.isfinite(value)]
    return float(statistics.fmean(nums)) if nums else float("nan")


def _read_csv_if_exists(path: Path) -> list[dict[str, str]]:
    return read_csv(path) if path.exists() else []


def _load_training_config(run_dir: Path) -> dict[str, Any]:
    for path in [
        run_dir.parent / "checkpoints" / "best" / "training_config.json",
        run_dir / "training_config.json",
    ]:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _annotate_rows(rows: list[dict[str, Any]], source: str, run_name: str, run_dir: Path, seed: int) -> list[dict[str, Any]]:
    config = _load_training_config(run_dir)
    selection_decode_mode = str(config.get("selection_decode_mode", "raw" if source == "exp12a" else "projected"))
    out = []
    for row in rows:
        item = dict(row)
        item.setdefault("decode_mode", "raw")
        item.update(
            {
                "source": source,
                "run_name": run_name,
                "seed": seed,
                "selection_rule": config.get("selection_rule", DEFAULT_SELECTION_RULE),
                "selection_decode_mode": selection_decode_mode,
                "uses_test_for_selection": False,
            }
        )
        out.append(item)
    return out


def _exp12a_run_dirs(runs_root: Path) -> list[Path]:
    return sorted((runs_root / "exp12a_decode_projection").glob("seed_*/eval_epoch_*"))


def _exp12b_run_dirs(runs_root: Path) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for run_name in EXP12B_RUNS:
        for run_dir in sorted((runs_root / run_name).glob("seed_*/run")):
            out.append((run_name, run_dir))
    return out


def load_exp12_rows(runs_root: Path) -> dict[str, list[dict[str, Any]]]:
    exp12a_metrics: list[dict[str, Any]] = []
    exp12a_low: list[dict[str, Any]] = []
    exp12b_dev: list[dict[str, Any]] = []
    exp12b_test: list[dict[str, Any]] = []
    exp12b_soft_dev: list[dict[str, Any]] = []
    exp12b_soft_test: list[dict[str, Any]] = []
    exp12b_low: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    manifest_rows = _read_csv_if_exists(EXP12_OUTPUT_DIR / "tables" / "exp12a_eval_manifest.csv")
    manifest_by_seed_epoch = {
        (_int(row.get("seed")), _int(row.get("selected_epoch"))): row for row in manifest_rows if row.get("status") == "COMPLETED"
    }

    for run_dir in _exp12a_run_dirs(runs_root):
        seed = _int(run_dir.parent.name.rsplit("_", 1)[-1])
        epoch = _int(run_dir.name.rsplit("_", 1)[-1])
        manifest = manifest_by_seed_epoch.get((seed, epoch), {})
        tables = run_dir / "tables"
        metrics = _read_csv_if_exists(tables / "epoch_metrics_dev.csv") + _read_csv_if_exists(tables / "epoch_metrics_test_diagnostic.csv")
        for row in _annotate_rows(metrics, "exp12a", "decode_projection_only", run_dir, seed):
            row["selected_epoch"] = epoch
            row["selected_global_step"] = manifest.get("selected_global_step", row.get("selected_global_step", ""))
            exp12a_metrics.append(row)
        exp12a_low.extend(_annotate_rows(_read_csv_if_exists(tables / "low_score_by_epoch_dev.csv"), "exp12a", "decode_projection_only", run_dir, seed))
        exp12a_low.extend(
            _annotate_rows(_read_csv_if_exists(tables / "low_score_by_epoch_test_diagnostic.csv"), "exp12a", "decode_projection_only", run_dir, seed)
        )

    for run_name, run_dir in _exp12b_run_dirs(runs_root):
        seed = _int(run_dir.parent.name.rsplit("_", 1)[-1])
        tables = run_dir / "tables"
        config = _load_training_config(run_dir)
        exp12b_dev.extend(_annotate_rows(_read_csv_if_exists(tables / "epoch_metrics_dev.csv"), "exp12b", run_name, run_dir, seed))
        exp12b_test.extend(_annotate_rows(_read_csv_if_exists(tables / "epoch_metrics_test_diagnostic.csv"), "exp12b", run_name, run_dir, seed))
        exp12b_soft_dev.extend(_annotate_rows(_read_csv_if_exists(tables / "soft_risk_metrics_dev.csv"), "exp12b", run_name, run_dir, seed))
        exp12b_soft_test.extend(_annotate_rows(_read_csv_if_exists(tables / "soft_risk_metrics_test_diagnostic.csv"), "exp12b", run_name, run_dir, seed))
        exp12b_low.extend(_annotate_rows(_read_csv_if_exists(tables / "low_score_by_epoch_dev.csv"), "exp12b", run_name, run_dir, seed))
        exp12b_low.extend(_annotate_rows(_read_csv_if_exists(tables / "low_score_by_epoch_test_diagnostic.csv"), "exp12b", run_name, run_dir, seed))
        for key in [
            "run_id",
            "ablation_name",
            "seed",
            "num_train_epochs",
            "lambda_point",
            "lambda_pair",
            "lambda_anchor",
            "lambda_mono",
            "eta_proj",
            "projection_method",
            "projection_in_decode",
            "projection_in_pair_score",
            "projection_in_point_loss",
            "projection_in_anchor",
            "use_projected_anchor",
            "use_raw_projection_consistency",
            "selection_rule",
            "selection_delta",
            "selection_decode_mode",
            "learning_rate",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
        ]:
            config_rows.append({"run_name": run_name, "seed": seed, "item": key, "value": config.get(key, ""), "note": ""})
        ckpt_root = run_dir.parent / "checkpoints"
        for row in _read_csv_if_exists(tables / "epoch_metrics_dev.csv"):
            if row.get("decode_mode", "raw") != "raw":
                continue
            epoch = _int(row.get("epoch"))
            ckpt = ckpt_root / f"epoch_{epoch:02d}"
            files = list(ckpt.rglob("*")) if ckpt.exists() else []
            size_mb = sum(path.stat().st_size for path in files if path.is_file()) / (1024 * 1024) if files else 0.0
            inventory_rows.append(
                {
                    "run_name": run_name,
                    "seed": seed,
                    "epoch": epoch,
                    "global_step": row.get("global_step", ""),
                    "checkpoint_path_local": relpath(ckpt),
                    "checkpoint_exists": ckpt.exists(),
                    "checkpoint_size_mb": size_mb,
                    "git_tracked": False,
                    "note": "Local ignored checkpoint directory under thesis_exp/runs.",
                }
            )
    return {
        "exp12a_metrics": sorted(exp12a_metrics, key=lambda row: (_int(row.get("seed")), str(row.get("split")), str(row.get("decode_mode")))),
        "exp12a_low": exp12a_low,
        "exp12b_dev": sorted(exp12b_dev, key=lambda row: (row.get("run_name", ""), _int(row.get("seed")), _int(row.get("epoch")), str(row.get("decode_mode")))),
        "exp12b_test": sorted(exp12b_test, key=lambda row: (row.get("run_name", ""), _int(row.get("seed")), _int(row.get("epoch")), str(row.get("decode_mode")))),
        "exp12b_soft_dev": exp12b_soft_dev,
        "exp12b_soft_test": exp12b_soft_test,
        "exp12b_low": exp12b_low,
        "config": config_rows,
        "inventory": inventory_rows,
    }


def _by_key(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {(row.get("source"), row.get("run_name"), _int(row.get("seed")), _int(row.get("epoch")), row.get("split"), row.get("decode_mode")): row for row in rows}


def build_projection_effect_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = _by_key(rows)
    out = []
    for key, raw in by_key.items():
        source, run_name, seed, epoch, split, decode = key
        if decode != "raw":
            continue
        projected = by_key.get((source, run_name, seed, epoch, split, "projected"))
        if not projected:
            continue
        out.append(
            {
                "source": source,
                "run_name": run_name,
                "seed": seed,
                "epoch": epoch,
                "global_step": raw.get("global_step", ""),
                "split": split,
                "raw_low_to_high_count": raw.get("low_to_high_count", ""),
                "projected_low_to_high_count": projected.get("low_to_high_count", ""),
                "delta_low_to_high_count": _float(projected.get("low_to_high_count")) - _float(raw.get("low_to_high_count")),
                "raw_low_to_high": raw.get("low_to_high", ""),
                "projected_low_to_high": projected.get("low_to_high", ""),
                "delta_low_to_high": _float(projected.get("low_to_high")) - _float(raw.get("low_to_high")),
                "raw_MAE": raw.get("MAE", ""),
                "projected_MAE": projected.get("MAE", ""),
                "delta_MAE": _float(projected.get("MAE")) - _float(raw.get("MAE")),
                "raw_QWK": raw.get("QWK", ""),
                "projected_QWK": projected.get("QWK", ""),
                "delta_QWK": _float(projected.get("QWK")) - _float(raw.get("QWK")),
                "raw_Acc@5": raw.get("Acc@5", ""),
                "projected_Acc@5": projected.get("Acc@5", ""),
                "delta_Acc@5": _float(projected.get("Acc@5")) - _float(raw.get("Acc@5")),
                "raw_monotonic_violation": raw.get("monotonic_violation", ""),
                "projected_monotonic_violation": projected.get("monotonic_violation", ""),
                "delta_monotonic_violation": _float(projected.get("monotonic_violation")) - _float(raw.get("monotonic_violation")),
            }
        )
    return out


def _select_epoch(dev_rows: list[dict[str, Any]], soft_rows: list[dict[str, Any]], delta: float) -> dict[str, Any]:
    if not dev_rows:
        return {}
    soft_by_epoch = {_int(row.get("epoch")): row for row in soft_rows}
    merged = []
    for row in dev_rows:
        soft = soft_by_epoch.get(_int(row.get("epoch")), {})
        merged.append({**row, "p_gt_3_low_mean": soft.get("p_gt_3_low_mean", "")})
    min_mae = min(_float(row.get("MAE")) for row in merged)
    eligible = [row for row in merged if _float(row.get("MAE")) <= min_mae + delta]
    return sorted(eligible, key=lambda row: (_float(row.get("p_gt_3_low_mean")), _float(row.get("MAE")), _int(row.get("epoch"))))[0]


def build_exp12b_selection(rows: dict[str, list[dict[str, Any]]], delta: float) -> list[dict[str, Any]]:
    out = []
    test_by_key = {
        (row.get("run_name"), _int(row.get("seed")), _int(row.get("epoch")), row.get("decode_mode")): row for row in rows["exp12b_test"]
    }
    keys = sorted({(row.get("run_name"), _int(row.get("seed")), row.get("selection_decode_mode", "projected")) for row in rows["exp12b_dev"]})
    for run_name, seed, selection_decode_mode in keys:
        dev = [
            row
            for row in rows["exp12b_dev"]
            if row.get("run_name") == run_name and _int(row.get("seed")) == seed and row.get("decode_mode", "raw") == selection_decode_mode
        ]
        soft = [
            row
            for row in rows["exp12b_soft_dev"]
            if row.get("run_name") == run_name and _int(row.get("seed")) == seed and row.get("decode_mode", "raw") == selection_decode_mode
        ]
        selected = _select_epoch(dev, soft, delta)
        epoch = _int(selected.get("epoch"))
        for decode_mode in ["raw", "projected"]:
            test = test_by_key.get((run_name, seed, epoch, decode_mode), {})
            if not test:
                continue
            out.append(
                {
                    "run_name": run_name,
                    "seed": seed,
                    "selection_rule": DEFAULT_SELECTION_RULE,
                    "selection_decode_mode": selection_decode_mode,
                    "uses_test_for_selection": False,
                    "selected_epoch": epoch,
                    "selected_global_step": selected.get("global_step", ""),
                    "decode_mode": decode_mode,
                    "test_MAE": test.get("MAE", ""),
                    "test_QWK": test.get("QWK", ""),
                    "test_Accuracy": test.get("Accuracy", ""),
                    "test_Acc@5": test.get("Acc@5", ""),
                    "test_low_to_high": test.get("low_to_high", ""),
                    "test_low_to_high_count": test.get("low_to_high_count", ""),
                    "test_true_low_score_count": test.get("true_low_score_count", ""),
                    "test_label1_low_to_high": test.get("label1_low_to_high", ""),
                    "test_label1_low_to_high_count": test.get("label1_low_to_high_count", ""),
                    "test_label2_low_to_high": test.get("label2_low_to_high", ""),
                    "test_label2_low_to_high_count": test.get("label2_low_to_high_count", ""),
                    "test_monotonic_violation": test.get("monotonic_violation", ""),
                    "test_p1_lt_p2": test.get("p1_lt_p2", ""),
                    "test_p2_lt_p3": test.get("p2_lt_p3", ""),
                    "test_p3_lt_p4": test.get("p3_lt_p4", ""),
                }
            )
    return out


def build_ablation_summary(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for run_name in sorted({row.get("run_name") for row in selected}):
        for decode_mode in ["raw", "projected"]:
            items = [row for row in selected if row.get("run_name") == run_name and row.get("decode_mode") == decode_mode]
            if not items:
                continue
            out.append(
                {
                    "run_name": run_name,
                    "decode_mode": decode_mode,
                    "num_seeds": len(items),
                    "mean_test_MAE": _mean(row.get("test_MAE") for row in items),
                    "mean_test_QWK": _mean(row.get("test_QWK") for row in items),
                    "mean_test_Acc@5": _mean(row.get("test_Acc@5") for row in items),
                    "mean_test_low_to_high": _mean(row.get("test_low_to_high") for row in items),
                    "mean_test_low_to_high_count": _mean(row.get("test_low_to_high_count") for row in items),
                    "mean_test_monotonic_violation": _mean(row.get("test_monotonic_violation") for row in items),
                }
            )
    return out


def build_low_to_high_by_label(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if row.get("split") != "test":
            continue
        out.append(
            {
                "source": row.get("source", ""),
                "run_name": row.get("run_name", ""),
                "seed": row.get("seed", ""),
                "epoch": row.get("epoch", ""),
                "decode_mode": row.get("decode_mode", ""),
                "label1_low_to_high": row.get("label1_low_to_high", ""),
                "label1_low_to_high_count": row.get("label1_low_to_high_count", ""),
                "label2_low_to_high": row.get("label2_low_to_high", ""),
                "label2_low_to_high_count": row.get("label2_low_to_high_count", ""),
                "true_low_score_count": row.get("true_low_score_count", ""),
            }
        )
    return out


def build_monotonic_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "source": row.get("source", ""),
            "run_name": row.get("run_name", ""),
            "seed": row.get("seed", ""),
            "epoch": row.get("epoch", ""),
            "split": row.get("split", ""),
            "decode_mode": row.get("decode_mode", ""),
            "monotonic_violation": row.get("monotonic_violation", ""),
            "p1_lt_p2": row.get("p1_lt_p2", ""),
            "p2_lt_p3": row.get("p2_lt_p3", ""),
            "p3_lt_p4": row.get("p3_lt_p4", ""),
        }
        for row in rows
    ]


def build_projection_delta_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    projected = [row for row in rows if row.get("decode_mode") == "projected"]
    out = []
    for key in sorted({(row.get("source"), row.get("run_name"), row.get("split")) for row in projected}):
        source, run_name, split = key
        items = [row for row in projected if (row.get("source"), row.get("run_name"), row.get("split")) == key]
        out.append(
            {
                "source": source,
                "run_name": run_name,
                "split": split,
                "num_rows": len(items),
                "mean_projection_l2_delta": _mean(row.get("mean_projection_l2_delta") for row in items),
                "mean_projection_linf_delta": _mean(row.get("mean_projection_linf_delta") for row in items),
                "low_score_mean_projection_l2_delta": _mean(row.get("low_score_mean_projection_l2_delta") for row in items),
            }
        )
    return out


def build_vs_baselines(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines = [
        {"model": "QD-B1", "MAE": 0.4279, "QWK": 0.6012, "low_to_high_count": 14, "low_to_high": 0.4516, "Acc@5": 0.7419, "monotonic_violation": 0.3119},
        {"model": "Exp09_QD-PR2", "MAE": 0.4192, "QWK": 0.6084, "low_to_high_count": 12, "low_to_high": 0.3871, "Acc@5": 0.7549, "monotonic_violation": 0.3527},
    ]
    out = []
    for baseline in baselines:
        out.append({"source": "reference", **baseline})
    for row in selected:
        if row.get("decode_mode") != "projected":
            continue
        out.append(
            {
                "source": "exp12b",
                "model": f"{row.get('run_name')}_{row.get('decode_mode')}_seed{row.get('seed')}",
                "MAE": row.get("test_MAE", ""),
                "QWK": row.get("test_QWK", ""),
                "low_to_high_count": row.get("test_low_to_high_count", ""),
                "low_to_high": row.get("test_low_to_high", ""),
                "Acc@5": row.get("test_Acc@5", ""),
                "monotonic_violation": row.get("test_monotonic_violation", ""),
            }
        )
    return out


def build_report(rows: dict[str, list[dict[str, Any]]], exp12a_effect: list[dict[str, Any]], selected: list[dict[str, Any]]) -> str:
    has_a = bool(rows["exp12a_metrics"])
    has_b = bool(rows["exp12b_dev"])
    lines = [
        "# Exp12 Monotonic Projection / MAP-OC",
        "",
        f"Exp12A status: `{'COMPLETED' if has_a else 'NO_COMPLETED_DECODE_RUNS'}`",
        f"Exp12B status: `{'COMPLETED' if has_b else 'NO_COMPLETED_TRAINING_RUNS'}`",
        "",
        "This report uses dev-only checkpoint selection. Test metrics are final evaluation or post-hoc diagnostic only and are not used for checkpoint selection, tuning, or training decisions.",
        "",
        "## Exp12A Decode-Only Projection",
        "",
    ]
    if not has_a:
        lines.extend(
            [
                "No completed Exp12A decode-only run was found under the local ignored runs directory.",
                "On the server, run `RUN_EXP12A_ONLY=1 ./thesis_exp/scripts/run_exp12_monotonic_projection_map_oc.sh` after Exp11 checkpoints are available.",
            ]
        )
    else:
        test_effect = [row for row in exp12a_effect if row.get("split") == "test"]
        best = test_effect[0] if test_effect else {}
        lines.extend(
            [
                f"- Projected monotonic violation: `{_fmt(best.get('projected_monotonic_violation'))}`.",
                f"- Raw to projected low-to-high count delta: `{_fmt(best.get('delta_low_to_high_count'), 0)}`.",
                f"- MAE delta: `{_fmt(best.get('delta_MAE'))}`; QWK delta: `{_fmt(best.get('delta_QWK'))}`; Acc@5 delta: `{_fmt(best.get('delta_Acc@5'))}`.",
                "- Low-score test subset is small, so count-level interpretation is necessary.",
            ]
        )
    lines.extend(["", "## Exp12B Train-Time MAP-OC", ""])
    if not has_b:
        lines.append("NO_COMPLETED_TRAINING_RUNS: Exp12B has not produced completed per-epoch metrics yet.")
    else:
        projected = [row for row in selected if row.get("decode_mode") == "projected"]
        lines.append("| run | seed | selected epoch | test low-to-high | count | MAE | QWK | monotonic |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in projected:
            lines.append(
                f"| {row.get('run_name')} | {row.get('seed')} | {row.get('selected_epoch')} | "
                f"{_fmt(row.get('test_low_to_high'))} | {row.get('test_low_to_high_count')} | "
                f"{_fmt(row.get('test_MAE'))} | {_fmt(row.get('test_QWK'))} | {_fmt(row.get('test_monotonic_violation'))} |"
            )
        lines.extend(
            [
                "",
                "If MAP-OC lowers monotonic violation but worsens low-to-high, this should be treated as a negative result. If it lowers low-to-high with MAE/QWK tradeoff, report both sides.",
            ]
        )
    lines.extend(
        [
            "",
            "## Method Notes",
            "",
            "- Monotonic projection uses exact PAVA onto `q1 >= q2 >= q3 >= q4`, clipped to `[0,1]`.",
            "- The method keeps ordinal threshold semantics and does not sort thresholds.",
            "- Pairwise learning here is supervised ordinal calibration, not a generative policy training method.",
            "- The fixed checkpoint selection policy is `mae_guard_p_gt_3_low_mean` with `delta=0.005` unless a config explicitly changes it.",
            "- All conclusions should center on low-to-high risk under the ordinal scoring constraint.",
        ]
    )
    return "\n".join(lines)


def collect(runs_root: Path, delta: float) -> dict[str, str]:
    ensure_exp12_dirs()
    rows = load_exp12_rows(runs_root)
    exp12a_effect = build_projection_effect_rows(rows["exp12a_metrics"])
    exp12b_effect = build_projection_effect_rows(rows["exp12b_dev"] + rows["exp12b_test"])
    exp12b_selected = build_exp12b_selection(rows, delta)
    exp12b_summary = build_ablation_summary(exp12b_selected)
    all_metric_rows = rows["exp12a_metrics"] + rows["exp12b_dev"] + rows["exp12b_test"]

    write_csv(EXP12_TABLES_DIR / "exp12a_decode_projection_metrics.csv", rows["exp12a_metrics"], fieldnames=METRIC_FIELDS)
    write_csv(EXP12_TABLES_DIR / "exp12a_raw_vs_projected_selected.csv", rows["exp12a_metrics"], fieldnames=METRIC_FIELDS)
    write_csv(EXP12_TABLES_DIR / "exp12a_projection_effect_by_seed_epoch.csv", exp12a_effect, fieldnames=EFFECT_FIELDS)
    write_csv(EXP12_TABLES_DIR / "exp12a_low_score_projection_distribution.csv", rows["exp12a_low"])
    write_csv(EXP12_TABLES_DIR / "exp12a_monotonic_by_threshold.csv", build_monotonic_rows(rows["exp12a_metrics"]))

    write_csv(EXP12_TABLES_DIR / "exp12b_train_metrics_dev.csv", rows["exp12b_dev"], fieldnames=METRIC_FIELDS)
    write_csv(EXP12_TABLES_DIR / "exp12b_train_metrics_test_diagnostic.csv", rows["exp12b_test"], fieldnames=METRIC_FIELDS)
    write_csv(EXP12_TABLES_DIR / "exp12b_selected_checkpoint_test_metrics.csv", exp12b_selected, fieldnames=SELECTED_FIELDS)
    write_csv(EXP12_TABLES_DIR / "exp12b_ablation_summary.csv", exp12b_summary)
    write_csv(EXP12_TABLES_DIR / "exp12b_raw_vs_projected_metrics.csv", exp12b_effect, fieldnames=EFFECT_FIELDS)
    write_csv(EXP12_TABLES_DIR / "exp12b_low_to_high_by_label.csv", build_low_to_high_by_label(rows["exp12b_test"]))
    write_csv(EXP12_TABLES_DIR / "exp12b_low_score_prediction_distribution.csv", rows["exp12b_low"])
    write_csv(EXP12_TABLES_DIR / "exp12b_monotonic_by_threshold.csv", build_monotonic_rows(rows["exp12b_dev"] + rows["exp12b_test"]))
    write_csv(EXP12_TABLES_DIR / "exp12b_projection_delta_summary.csv", build_projection_delta_summary(rows["exp12b_dev"] + rows["exp12b_test"]))
    write_csv(EXP12_TABLES_DIR / "exp12b_vs_qdb1_qdpr2_exp11.csv", build_vs_baselines(exp12b_selected))
    write_csv(EXP12_TABLES_DIR / "exp12_run_config_summary.csv", rows["config"])
    write_csv(EXP12_TABLES_DIR / "exp12_checkpoint_inventory.csv", rows["inventory"])
    if len({row.get("seed") for row in exp12b_selected}) > 1:
        write_csv(EXP12_TABLES_DIR / "exp12_multiseed_summary.csv", exp12b_selected, fieldnames=SELECTED_FIELDS)
        write_csv(EXP12_TABLES_DIR / "exp12_multiseed_rule_aggregate.csv", exp12b_summary)
    else:
        write_csv(EXP12_TABLES_DIR / "exp12_multiseed_summary.csv", [], fieldnames=SELECTED_FIELDS)
        write_csv(EXP12_TABLES_DIR / "exp12_multiseed_rule_aggregate.csv", [])

    report = build_report(rows, exp12a_effect, exp12b_selected)
    write_text(EXP12_REPORTS_DIR / "exp12_monotonic_projection_map_oc_report.md", report)
    write_text(
        EXP12_REPORTS_DIR / "exp12_monotonic_projection_map_oc_review_package.md",
        report
        + "\n\n## Review Checklist\n\n"
        + "- Verify `uses_test_for_selection` is false.\n"
        + "- Verify projected monotonic violation is zero or near zero when projected rows exist.\n"
        + "- Verify no checkpoints, weights, raw predictions, arrays, or logs are under tracked Exp12 outputs.\n",
    )
    return {
        "report": relpath(EXP12_REPORTS_DIR / "exp12_monotonic_projection_map_oc_report.md"),
        "tables": relpath(EXP12_TABLES_DIR),
        "metric_rows": str(len(all_metric_rows)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Exp12 monotonic projection / MAP-OC results.")
    parser.add_argument("--runs_root", type=Path, default=EXP12_LOCAL_RUNS_DIR)
    parser.add_argument("--delta", type=float, default=DEFAULT_SELECTION_DELTA)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(collect(args.runs_root, args.delta), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
