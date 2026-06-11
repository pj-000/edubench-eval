"""Data loading helpers for Exp7-C ordinal calibration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.src.edujudge.exp07_rank_consistent import (
    EXP07_CALIBRATION_CONFIGS_DIR,
    EXP07_CALIBRATION_TABLES_DIR,
    EXP07_OUTPUT_DIR,
    EXP07_RUN_ID,
    QD_B0_RUN_ID,
    QD_B1_RUN_ID,
    QD_BASELINE_RUNS_DIR,
    ensure_exp07_calibration_dirs,
    exp07_run_dir,
)
from thesis_exp.src.edujudge.utils.io import read_csv, read_jsonl, write_csv, write_json


PROB_COLUMNS = ["prob_gt_1", "prob_gt_2", "prob_gt_3", "prob_gt_4"]
LOGIT_COLUMNS = ["logit_gt_1", "logit_gt_2", "logit_gt_3", "logit_gt_4"]
META_COLUMNS = [
    "record_id",
    "label_5",
    "human_mean_5",
    "metric_canonical",
    "language",
    "scenario_canonical",
]
THRESHOLD_GRID = [round(0.30 + idx * 0.05, 2) for idx in range(12)]
TEMPERATURE_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]
LAMBDA_LOW_GRID = [0.5, 1.0, 2.0, 3.0]
ETA_HIGH_GRID = [0.5, 1.0]


@dataclass(frozen=True)
class CalibrationBaseConfig:
    base_model: str
    run_dir: Path


@dataclass
class SplitData:
    split: str
    records: list[dict[str, Any]]
    labels: np.ndarray
    human_mean: np.ndarray
    probs: np.ndarray | None
    logits: np.ndarray | None


@dataclass
class BaseCalibrationData:
    base_model: str
    dev: SplitData
    test: SplitData


BASE_CONFIGS = [
    CalibrationBaseConfig(QD_B0_RUN_ID, QD_BASELINE_RUNS_DIR / QD_B0_RUN_ID),
    CalibrationBaseConfig(QD_B1_RUN_ID, QD_BASELINE_RUNS_DIR / QD_B1_RUN_ID),
    CalibrationBaseConfig(EXP07_RUN_ID, exp07_run_dir()),
]


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def raw_prediction_from_probs(probs: np.ndarray, thresholds: list[float] | np.ndarray | None = None) -> np.ndarray:
    threshold_values = np.asarray(thresholds if thresholds is not None else [0.5, 0.5, 0.5, 0.5], dtype=float)
    return np.clip(1 + (probs > threshold_values.reshape(1, 4)).sum(axis=1), 1, 5).astype(int)


def expected_score_from_probs(probs: np.ndarray) -> np.ndarray:
    return np.clip(1.0 + probs.sum(axis=1), 1.0, 5.0)


def _array_path(config: CalibrationBaseConfig, name: str, split: str) -> Path:
    return config.run_dir / "arrays" / f"{name}_{split}.npy"


def _predictions_path(config: CalibrationBaseConfig, split: str) -> Path:
    return config.run_dir / "predictions" / f"predictions_{split}.jsonl"


def _inventory_source_rows() -> dict[str, dict[str, str]]:
    path = EXP07_OUTPUT_DIR / "tables" / "exp07_calibration_logit_inventory.csv"
    if not path.exists():
        return {}
    return {row.get("run_id", ""): row for row in read_csv(path)}


def _availability(local_path: Path, source_row: dict[str, str], source_key: str) -> str:
    if local_path.exists():
        return "yes_local"
    value = source_row.get(source_key, "")
    if value == "yes_server":
        return "server_ready_local_missing"
    return "no"


def _has_local_probs(config: CalibrationBaseConfig) -> bool:
    return _array_path(config, "probs", "dev").exists() and _array_path(config, "probs", "test").exists()


def _has_local_logits(config: CalibrationBaseConfig) -> bool:
    return _array_path(config, "logits", "dev").exists() and _array_path(config, "logits", "test").exists()


def _blocking_reason(row: dict[str, Any]) -> str:
    if row["can_threshold_calibrate"] == "yes":
        return "ready"
    values = [
        row["dev_probs_available"],
        row["test_probs_available"],
        row["dev_logits_available"],
        row["test_logits_available"],
    ]
    if any(value == "server_ready_local_missing" for value in values):
        return "server_ready_local_missing; sync artifacts locally or run calibration on the server"
    return "BLOCKED_MISSING_LOCAL_ARTIFACTS"


def calibration_base_inventory(source_rows: dict[str, dict[str, str]] | None = None) -> list[dict[str, Any]]:
    source_rows = source_rows if source_rows is not None else _inventory_source_rows()
    rows: list[dict[str, Any]] = []
    for config in BASE_CONFIGS:
        source = source_rows.get(config.base_model, {})
        row = {
            "base_model": config.base_model,
            "dev_probs_available": _availability(_array_path(config, "probs", "dev"), source, "dev_probs_available"),
            "test_probs_available": _availability(_array_path(config, "probs", "test"), source, "test_probs_available"),
            "dev_logits_available": _availability(_array_path(config, "logits", "dev"), source, "dev_logits_available"),
            "test_logits_available": _availability(_array_path(config, "logits", "test"), source, "test_logits_available"),
        }
        can_threshold = _has_local_probs(config)
        can_temperature = can_threshold and _has_local_logits(config)
        row["can_threshold_calibrate"] = "yes" if can_threshold else "no"
        row["can_temperature_calibrate"] = "yes" if can_temperature else "no"
        row["blocking_reason"] = _blocking_reason(row)
        rows.append(row)
    return rows


def write_base_inventory(source_rows: dict[str, dict[str, str]] | None = None) -> list[dict[str, Any]]:
    ensure_exp07_calibration_dirs()
    rows = calibration_base_inventory(source_rows)
    write_csv(EXP07_CALIBRATION_TABLES_DIR / "calibration_base_inventory.csv", rows)
    write_json(
        EXP07_CALIBRATION_CONFIGS_DIR / "exp07_c_calibration_config.json",
        {
            "temperature_grid": TEMPERATURE_GRID,
            "threshold_grid": THRESHOLD_GRID,
            "lambda_low_grid": LAMBDA_LOW_GRID,
            "eta_high_grid": ETA_HIGH_GRID,
            "selection": "dev-only; test set used only for final evaluation",
            "training": "not run",
            "api": "not called",
        },
    )
    return rows


def load_split(config: CalibrationBaseConfig, split: str) -> SplitData:
    predictions_path = _predictions_path(config, split)
    records = read_jsonl(predictions_path) if predictions_path.exists() else []
    probs_path = _array_path(config, "probs", split)
    logits_path = _array_path(config, "logits", split)
    labels_path = _array_path(config, "labels", split)
    record_ids_path = config.run_dir / "arrays" / f"record_ids_{split}.txt"
    probs = np.load(probs_path).astype(float) if probs_path.exists() else None
    logits = np.load(logits_path).astype(float) if logits_path.exists() else None
    labels = np.load(labels_path).astype(int) if labels_path.exists() else np.array([], dtype=int)
    if not records and labels.size and probs is not None:
        record_ids = [line.strip() for line in record_ids_path.read_text(encoding="utf-8").splitlines()] if record_ids_path.exists() else [f"{split}_{idx}" for idx in range(labels.size)]
        records = [
            {
                "record_id": record_ids[idx],
                "label_5": int(labels[idx]),
                "human_mean_5": float(labels[idx]),
                "metric_canonical": "",
                "language": "",
                "scenario_canonical": "",
            }
            for idx in range(labels.size)
        ]
    if not records or probs is None:
        raise FileNotFoundError(f"Missing local calibration inputs for {config.base_model} {split}")
    labels = np.array([int(row.get("label_5", labels[idx] if labels.size else 0)) for idx, row in enumerate(records)], dtype=int)
    human_mean = np.array([float(row.get("human_mean_5", labels[idx])) for idx, row in enumerate(records)], dtype=float)
    if probs.shape[0] != labels.shape[0]:
        raise ValueError(f"{config.base_model} {split} probs rows {probs.shape[0]} != labels {labels.shape[0]}")
    if logits is not None and logits.shape != probs.shape:
        raise ValueError(f"{config.base_model} {split} logits shape {logits.shape} != probs shape {probs.shape}")
    return SplitData(split=split, records=records, labels=labels, human_mean=human_mean, probs=probs, logits=logits)


def load_available_bases(inventory_rows: list[dict[str, Any]] | None = None) -> list[BaseCalibrationData]:
    inventory = inventory_rows if inventory_rows is not None else calibration_base_inventory()
    ready = {row["base_model"] for row in inventory if row.get("can_threshold_calibrate") == "yes"}
    out: list[BaseCalibrationData] = []
    for config in BASE_CONFIGS:
        if config.base_model not in ready:
            continue
        out.append(BaseCalibrationData(base_model=config.base_model, dev=load_split(config, "dev"), test=load_split(config, "test")))
    return out


def unified_prediction_rows(split_data: SplitData, pred_label: np.ndarray, method: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected = expected_score_from_probs(split_data.probs if split_data.probs is not None else np.zeros((len(pred_label), 4)))
    for idx, record in enumerate(split_data.records):
        row = {key: record.get(key, "") for key in META_COLUMNS}
        probs = split_data.probs[idx] if split_data.probs is not None else np.full(4, np.nan)
        logits = split_data.logits[idx] if split_data.logits is not None else np.full(4, np.nan)
        row.update({PROB_COLUMNS[col]: float(probs[col]) for col in range(4)})
        row.update({LOGIT_COLUMNS[col]: float(logits[col]) for col in range(4)})
        row["pred_label_raw"] = int(record.get("pred_label", raw_prediction_from_probs(probs.reshape(1, 4))[0]))
        row["pred_score_expected_raw"] = float(record.get("pred_score_expected", expected[idx]))
        row["pred_label_calibrated"] = int(pred_label[idx])
        row["calibration_method"] = method
        rows.append(row)
    return rows
