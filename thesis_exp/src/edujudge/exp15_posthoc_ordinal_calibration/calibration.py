"""Post-hoc calibration utilities for ordinal threshold outputs."""

from __future__ import annotations

import math
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.src.edujudge.exp02.train_ce_baseline import qwk
from thesis_exp.src.edujudge.exp12_monotonic_projection_map_oc.monotone_projection import (
    project_nonincreasing_probs,
)
from thesis_exp.src.edujudge.utils.io import REPO_ROOT, relpath


THRESHOLDS = np.array([1, 2, 3, 4], dtype=np.int64)


@dataclass(frozen=True)
class ArraySource:
    source_id: str
    source_family: str
    run_name: str
    seed: str
    arrays_dir: Path
    note: str = ""


@dataclass(frozen=True)
class OrdinalArrays:
    logits_dev: np.ndarray
    logits_test: np.ndarray
    labels_dev: np.ndarray
    labels_test: np.ndarray
    targets_dev: np.ndarray
    targets_test: np.ndarray
    record_ids_dev: list[str]
    record_ids_test: list[str]


@dataclass(frozen=True)
class CalibratorSpec:
    calibrator_id: str
    family: str
    description: str
    temperature: float | None = None
    bias: tuple[float, float, float, float] | None = None
    decode_thresholds: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 0.5)
    use_projection: bool = True
    use_isotonic: bool = False


@dataclass(frozen=True)
class IsotonicModel:
    x_sorted: np.ndarray
    y_fit_sorted: np.ndarray


def sigmoid(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    out = np.empty_like(arr, dtype=np.float64)
    pos = arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-arr[pos]))
    exp_x = np.exp(arr[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def _read_record_ids(path: Path, n: int) -> list[str]:
    if path.exists():
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [str(idx) for idx in range(n)]


def _load_array(arrays_dir: Path, name: str) -> np.ndarray:
    direct = arrays_dir / f"{name}.npy"
    if direct.exists():
        return np.load(direct, allow_pickle=False)
    bundle = arrays_dir / "dev_test_arrays.npz"
    if bundle.exists():
        return np.load(bundle, allow_pickle=True)[name]
    raise FileNotFoundError(f"Missing {name} under {arrays_dir}")


def _load_human_means(arrays_dir: Path, split: str, fallback: np.ndarray) -> np.ndarray:
    run_dir = arrays_dir.parent
    candidates = [
        run_dir / f"predictions_{split}.jsonl",
        run_dir / "predictions" / f"predictions_{split}.jsonl",
        run_dir / f"predictions_{split}_projected.jsonl",
        run_dir / "predictions" / f"predictions_{split}_projected.jsonl",
    ]
    for path in candidates:
        if not path.exists():
            continue
        values: list[float] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "human_mean_5" not in row:
                    values = []
                    break
                values.append(float(row["human_mean_5"]))
        if len(values) == len(fallback):
            return np.asarray(values, dtype=np.float64)
    return np.asarray(fallback, dtype=np.float64)


def load_ordinal_arrays(arrays_dir: Path) -> OrdinalArrays:
    logits_dev = np.asarray(_load_array(arrays_dir, "logits_dev"), dtype=np.float64)
    logits_test = np.asarray(_load_array(arrays_dir, "logits_test"), dtype=np.float64)
    labels_dev = np.asarray(_load_array(arrays_dir, "labels_dev"), dtype=np.int64)
    labels_test = np.asarray(_load_array(arrays_dir, "labels_test"), dtype=np.int64)
    try:
        target_labels_dev = np.asarray(_load_array(arrays_dir, "targets_dev"), dtype=np.float64)
        target_labels_test = np.asarray(_load_array(arrays_dir, "targets_test"), dtype=np.float64)
    except FileNotFoundError:
        target_labels_dev = labels_dev.astype(np.float64)
        target_labels_test = labels_test.astype(np.float64)
    targets_dev = _load_human_means(arrays_dir, "dev", target_labels_dev)
    targets_test = _load_human_means(arrays_dir, "test", target_labels_test)
    if logits_dev.ndim != 2 or logits_dev.shape[1] != 4:
        raise ValueError(f"Expected dev logits shape [n,4], got {logits_dev.shape}")
    if logits_test.ndim != 2 or logits_test.shape[1] != 4:
        raise ValueError(f"Expected test logits shape [n,4], got {logits_test.shape}")
    return OrdinalArrays(
        logits_dev=logits_dev,
        logits_test=logits_test,
        labels_dev=labels_dev,
        labels_test=labels_test,
        targets_dev=targets_dev,
        targets_test=targets_test,
        record_ids_dev=_read_record_ids(arrays_dir / "record_ids_dev.txt", len(labels_dev)),
        record_ids_test=_read_record_ids(arrays_dir / "record_ids_test.txt", len(labels_test)),
    )


def _source_from_run_dir(source_family: str, run_dir: Path) -> ArraySource:
    parts = run_dir.relative_to(REPO_ROOT).parts
    run_name = run_dir.name
    seed = ""
    if "seed_42" in parts or any(part.startswith("seed_") for part in parts):
        seed_parts = [part for part in parts if part.startswith("seed_")]
        seed = seed_parts[-1].replace("seed_", "") if seed_parts else ""
        if len(parts) >= 2:
            run_name = parts[-3] if parts[-1] == "run" else run_dir.parent.name
    source_id = f"{source_family}:{run_name}:seed{seed or 'na'}"
    return ArraySource(source_id=source_id, source_family=source_family, run_name=run_name, seed=seed, arrays_dir=run_dir / "arrays")


def _ready_arrays_dir(arrays_dir: Path) -> bool:
    bundle = arrays_dir / "dev_test_arrays.npz"
    if bundle.exists():
        return True
    required = ["logits_dev.npy", "logits_test.npy", "labels_dev.npy", "labels_test.npy"]
    return all((arrays_dir / name).exists() for name in required)


def discover_array_sources() -> list[ArraySource]:
    sources: list[ArraySource] = []
    qdpr2 = (
        REPO_ROOT
        / "thesis_exp"
        / "outputs"
        / "exp09_pairwise_ordinal_qdpr2"
        / "runs"
        / "QD-PR2_AnchoredPairwiseOrdinal_human_only"
        / "arrays"
    )
    sources.append(
        ArraySource(
            source_id="exp09_qdpr2:formal:seed42",
            source_family="exp09_qdpr2",
            run_name="QD-PR2_AnchoredPairwiseOrdinal_human_only",
            seed="42",
            arrays_dir=qdpr2,
            note="Formal QD-PR2 best checkpoint arrays.",
        )
    )
    for family, root in [
        ("exp12", REPO_ROOT / "thesis_exp" / "runs" / "exp12_monotonic_projection_map_oc"),
        ("exp13", REPO_ROOT / "thesis_exp" / "runs" / "exp13_risk_boundary_map_oc"),
        ("exp14", REPO_ROOT / "thesis_exp" / "runs" / "exp14_logit_margin_tail_risk_oc"),
    ]:
        if not root.exists():
            continue
        for run_dir in sorted(path.parent.parent for path in root.rglob("arrays/logits_dev.npy")):
            if "smoke_test" in run_dir.parts:
                continue
            if not _ready_arrays_dir(run_dir / "arrays"):
                continue
            sources.append(_source_from_run_dir(family, run_dir))
    deduped: list[ArraySource] = []
    seen: set[Path] = set()
    for source in sources:
        key = source.arrays_dir.resolve()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(source)
    return deduped


def source_inventory_rows(sources: list[ArraySource]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in sources:
        required = [source.arrays_dir / f"{name}.npy" for name in ["logits_dev", "logits_test", "labels_dev", "labels_test"]]
        bundle = source.arrays_dir / "dev_test_arrays.npz"
        ready = all(path.exists() for path in required) or bundle.exists()
        rows.append(
            {
                "source_id": source.source_id,
                "source_family": source.source_family,
                "run_name": source.run_name,
                "seed": source.seed,
                "arrays_dir": relpath(source.arrays_dir),
                "status": "READY" if ready else "MISSING_ARRAYS",
                "note": source.note,
            }
        )
    return rows


def raw_probs_from_spec(logits: np.ndarray, spec: CalibratorSpec) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float64)
    if spec.temperature is not None:
        arr = arr / float(spec.temperature)
    if spec.bias is not None:
        arr = arr + np.asarray(spec.bias, dtype=np.float64).reshape(1, 4)
    probs = sigmoid(arr)
    return project_nonincreasing_probs(probs) if spec.use_projection else probs


def fit_isotonic_1d(x: np.ndarray, y: np.ndarray) -> IsotonicModel:
    order = np.argsort(x, kind="mergesort")
    xs = np.asarray(x, dtype=np.float64)[order]
    ys = np.asarray(y, dtype=np.float64)[order]
    blocks: list[dict[str, Any]] = []
    for idx, value in enumerate(ys):
        blocks.append({"start": idx, "end": idx + 1, "sum": float(value), "count": 1})
        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]
            if left["sum"] / left["count"] <= right["sum"] / right["count"]:
                break
            blocks[-2:] = [
                {
                    "start": left["start"],
                    "end": right["end"],
                    "sum": left["sum"] + right["sum"],
                    "count": left["count"] + right["count"],
                }
            ]
    y_fit = np.empty_like(ys, dtype=np.float64)
    for block in blocks:
        y_fit[block["start"] : block["end"]] = block["sum"] / block["count"]
    return IsotonicModel(x_sorted=xs, y_fit_sorted=np.clip(y_fit, 0.0, 1.0))


def fit_isotonic_thresholds(dev_probs: np.ndarray, labels: np.ndarray) -> list[IsotonicModel]:
    targets = (labels.reshape(-1, 1) > THRESHOLDS.reshape(1, -1)).astype(np.float64)
    return [fit_isotonic_1d(dev_probs[:, idx], targets[:, idx]) for idx in range(4)]


def apply_isotonic_thresholds(probs: np.ndarray, models: list[IsotonicModel]) -> np.ndarray:
    out = np.empty_like(probs, dtype=np.float64)
    for idx, model in enumerate(models):
        out[:, idx] = np.interp(probs[:, idx], model.x_sorted, model.y_fit_sorted, left=model.y_fit_sorted[0], right=model.y_fit_sorted[-1])
    return np.clip(out, 0.0, 1.0)


def calibrated_probs(
    logits: np.ndarray,
    spec: CalibratorSpec,
    isotonic_models: list[IsotonicModel] | None = None,
) -> np.ndarray:
    probs = raw_probs_from_spec(logits, spec)
    if spec.use_isotonic:
        if isotonic_models is None:
            raise ValueError("isotonic calibrator requires fitted isotonic models")
        probs = apply_isotonic_thresholds(probs, isotonic_models)
        if spec.use_projection:
            probs = project_nonincreasing_probs(probs)
    return probs


def cumulative_decode(probs: np.ndarray, thresholds: tuple[float, float, float, float]) -> np.ndarray:
    decisions = probs > np.asarray(thresholds, dtype=np.float64).reshape(1, 4)
    decisions = np.minimum.accumulate(decisions.astype(np.int64), axis=1)
    return (1 + decisions.sum(axis=1)).astype(np.int64)


def threshold_brier(probs: np.ndarray, labels: np.ndarray) -> float:
    targets = (labels.reshape(-1, 1) > THRESHOLDS.reshape(1, -1)).astype(np.float64)
    return float(np.mean((np.asarray(probs, dtype=np.float64) - targets) ** 2))


def threshold_ece(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    targets = (labels.reshape(-1, 1) > THRESHOLDS.reshape(1, -1)).astype(np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = probs.size
    if total == 0:
        return float("nan")
    error = 0.0
    for idx in range(4):
        p = probs[:, idx]
        y = targets[:, idx]
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
            if not np.any(mask):
                continue
            error += float(mask.sum()) * abs(float(np.mean(p[mask])) - float(np.mean(y[mask])))
    return error / float(total)


def _rate(count: int, total: int) -> float:
    return float(count / total) if total else float("nan")


def _quantile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.quantile(values, q)) if values.size else float("nan")


def metric_row(
    source: ArraySource,
    spec: CalibratorSpec,
    split: str,
    probs: np.ndarray,
    labels: np.ndarray,
    targets: np.ndarray,
    ece_bins: int,
    selected: bool = False,
    eligible: bool = True,
) -> dict[str, Any]:
    preds = cumulative_decode(probs, spec.decode_thresholds)
    expected = 1.0 + np.sum(probs, axis=1)
    low = labels <= 2
    label1 = labels == 1
    label2 = labels == 2
    high = labels >= 4
    label4 = labels == 4
    label5 = labels == 5
    low_to_high_count = int(np.sum(low & (preds >= 4)))
    label1_l2h = int(np.sum(label1 & (preds >= 4)))
    label2_l2h = int(np.sum(label2 & (preds >= 4)))
    high_to_low = int(np.sum(high & (preds <= 2)))
    violations = np.any(probs[:, :-1] < probs[:, 1:], axis=1)
    pair_violations = probs[:, :-1] < probs[:, 1:]
    row = {
        "source_id": source.source_id,
        "source_family": source.source_family,
        "run_name": source.run_name,
        "seed": source.seed,
        "split": split,
        "calibrator_id": spec.calibrator_id,
        "calibrator_family": spec.family,
        "description": spec.description,
        "decode_thresholds": ";".join(f"{value:.3f}" for value in spec.decode_thresholds),
        "uses_test_for_selection": False,
        "selected": selected,
        "eligible": eligible,
        "n": int(labels.size),
        "MAE_label": float(np.mean(np.abs(preds.astype(np.float64) - targets))),
        "MAE_expected": float(np.mean(np.abs(expected - targets))),
        "QWK": qwk(labels.astype(int).tolist(), preds.astype(int).tolist()),
        "Accuracy": float(np.mean(labels == preds)),
        "Acc@5": _rate(int(np.sum(label5 & (preds == 5))), int(np.sum(label5))),
        "low_to_high": _rate(low_to_high_count, int(np.sum(low))),
        "low_to_high_count": low_to_high_count,
        "true_low_score_count": int(np.sum(low)),
        "label1_low_to_high": _rate(label1_l2h, int(np.sum(label1))),
        "label1_low_to_high_count": label1_l2h,
        "label1_count": int(np.sum(label1)),
        "label2_low_to_high": _rate(label2_l2h, int(np.sum(label2))),
        "label2_low_to_high_count": label2_l2h,
        "label2_count": int(np.sum(label2)),
        "high_to_low": _rate(high_to_low, int(np.sum(high))),
        "high_to_low_count": high_to_low,
        "true_high_score_count": int(np.sum(high)),
        "label4_recall": _rate(int(np.sum(label4 & (preds == 4))), int(np.sum(label4))),
        "label5_recall": _rate(int(np.sum(label5 & (preds == 5))), int(np.sum(label5))),
        "monotonic_violation": float(np.mean(violations)),
        "p1_lt_p2": float(np.mean(pair_violations[:, 0])),
        "p2_lt_p3": float(np.mean(pair_violations[:, 1])),
        "p3_lt_p4": float(np.mean(pair_violations[:, 2])),
        "threshold_brier": threshold_brier(probs, labels),
        "threshold_ece": threshold_ece(probs, labels, bins=ece_bins),
        "p_gt_3_low_mean": float(np.mean(probs[low, 2])) if np.any(low) else float("nan"),
        "p_gt_3_label1_mean": float(np.mean(probs[label1, 2])) if np.any(label1) else float("nan"),
        "p_gt_3_label2_mean": float(np.mean(probs[label2, 2])) if np.any(label2) else float("nan"),
        "p_gt_3_low_q90": _quantile(probs[low, 2], 0.90) if np.any(low) else float("nan"),
        "p_gt_3_low_q95": _quantile(probs[low, 2], 0.95) if np.any(low) else float("nan"),
        "low_count_with_p_gt_3_over_0p5": int(np.sum(low & (probs[:, 2] > 0.5))),
    }
    return row


def prediction_distribution_rows(
    source: ArraySource,
    spec: CalibratorSpec,
    split: str,
    probs: np.ndarray,
    labels: np.ndarray,
) -> list[dict[str, Any]]:
    preds = cumulative_decode(probs, spec.decode_thresholds)
    rows: list[dict[str, Any]] = []
    for pred_label in [1, 2, 3, 4, 5]:
        count = int(np.sum(preds == pred_label))
        rows.append(
            {
                "source_id": source.source_id,
                "split": split,
                "calibrator_id": spec.calibrator_id,
                "pred_label": pred_label,
                "count": count,
                "rate": _rate(count, int(labels.size)),
            }
        )
    for true_label in [1, 2, 3, 4, 5]:
        mask = labels == true_label
        for pred_label in [1, 2, 3, 4, 5]:
            count = int(np.sum(mask & (preds == pred_label)))
            rows.append(
                {
                    "source_id": source.source_id,
                    "split": split,
                    "calibrator_id": spec.calibrator_id,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "count": count,
                    "rate": _rate(count, int(np.sum(mask))),
                }
            )
    return rows


def candidate_specs() -> list[CalibratorSpec]:
    specs = [
        CalibratorSpec("raw_t0p50", "baseline", "Raw sigmoid probabilities with 0.5 threshold.", use_projection=False),
        CalibratorSpec("pava_t0p50", "projection", "PAVA-projected probabilities with 0.5 threshold."),
        CalibratorSpec("isotonic_pava_t0p50", "isotonic", "Threshold-wise isotonic calibration on dev, then PAVA.", use_isotonic=True),
    ]
    for temp in [0.5, 0.75, 1.25, 1.5, 2.0, 3.0, 4.0]:
        token = str(temp).replace(".", "p")
        specs.append(
            CalibratorSpec(
                f"temp{token}_pava_t0p50",
                "temperature",
                f"Global temperature T={temp} before sigmoid, then PAVA.",
                temperature=temp,
            )
        )
    for tau in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        token = f"{tau:.2f}".replace(".", "p")
        specs.append(
            CalibratorSpec(
                f"pava_all_tau{token}",
                "decode_threshold",
                f"PAVA probabilities with all ordinal decode thresholds at {tau:.2f}.",
                decode_thresholds=(tau, tau, tau, tau),
            )
        )
    for tau in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98]:
        token = f"{tau:.2f}".replace(".", "p")
        specs.append(
            CalibratorSpec(
                f"pava_q3_tau{token}",
                "q3_decode_threshold",
                f"PAVA probabilities with only q3 decode threshold at {tau:.2f}.",
                decode_thresholds=(0.5, 0.5, tau, 0.5),
            )
        )
    for bias in [-0.25, -0.5, -0.75, -1.0, -1.5, -2.0]:
        token = str(abs(bias)).replace(".", "p")
        specs.append(
            CalibratorSpec(
                f"q3_bias_neg{token}_pava",
                "q3_logit_bias",
                f"Add {bias:.2f} to q3 logit before sigmoid, then PAVA.",
                bias=(0.0, 0.0, bias, 0.0),
            )
        )
    return specs


def select_dev_row(rows: list[dict[str, Any]], delta: float) -> dict[str, Any] | None:
    if not rows:
        return None
    baseline = next((row for row in rows if row.get("calibrator_id") == "pava_t0p50"), rows[0])
    baseline_mae = float(baseline["MAE_label"])
    candidates = [row for row in rows if float(row["MAE_label"]) <= baseline_mae + delta]
    for row in rows:
        row["eligible"] = row in candidates
    selected = sorted(
        candidates,
        key=lambda row: (
            int(row["low_to_high_count"]),
            int(row["label2_low_to_high_count"]),
            int(row["high_to_low_count"]),
            float(row["threshold_brier"]),
            float(row["threshold_ece"]),
            float(row["MAE_label"]),
            0 if row.get("calibrator_id") == baseline.get("calibrator_id") else 1,
        ),
    )[0]
    for row in rows:
        row["selected"] = row is selected
        row["baseline_low_to_high_count"] = baseline.get("low_to_high_count")
        row["delta_low_to_high_count_vs_pava"] = int(row["low_to_high_count"]) - int(baseline["low_to_high_count"])
        row["baseline_MAE_label"] = baseline.get("MAE_label")
        row["delta_MAE_label_vs_pava"] = float(row["MAE_label"]) - float(baseline["MAE_label"])
    return selected
