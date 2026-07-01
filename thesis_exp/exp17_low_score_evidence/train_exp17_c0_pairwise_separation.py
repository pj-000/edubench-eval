"""Train Exp17-C0 pairwise-low quality separation under Boundary Linking."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from thesis_exp.src.edujudge.exp16_boundary_linking.data import (
    BoundaryLinkingCollator,
    BoundaryLinkingDataset,
    load_samples,
)
from thesis_exp.src.edujudge.exp16_boundary_linking.losses import ordinal_bce_loss
from thesis_exp.src.edujudge.exp16_boundary_linking.metrics import compute_metrics
from thesis_exp.src.edujudge.exp16_boundary_linking.model import BoundaryLinkingOrdinalModel
from thesis_exp.src.edujudge.exp16_boundary_linking.train_boundary_linking import (
    autocast_context,
    make_tokenizer,
    move_batch,
    set_seed,
)
from thesis_exp.src.edujudge.utils.io import read_jsonl, relpath, write_csv, write_json


DEFAULT_TRAIN_PATH = Path("thesis_exp/data/splits/question_seed42/train.jsonl")
DEFAULT_DEV_PATH = Path("thesis_exp/data/splits/question_seed42/dev.jsonl")
DEFAULT_INIT_CKPT = Path("thesis_exp/outputs/exp16_boundary_linking/runs/qmr/seed_42/checkpoint_best/state_dict.pt")
DEFAULT_A0_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp17_a0_train_hidden_failure_signals_seed42")
DEFAULT_D1_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/d1_hidden_failure_audit_seed42_dev")
DEFAULT_OUTPUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp17_c0_pairwise_separation_seed42")

C0_CONFIGS: dict[str, dict[str, Any]] = {
    "C0_0_ordinal_continue": {"gamma": 0.0, "margin": 0.2, "temperature": 0.2, "pair_source": "none"},
    "C0_1_all_pairs_gamma0p02_m0p2": {"gamma": 0.02, "margin": 0.2, "temperature": 0.2, "pair_source": "all_a0_pairs"},
    "C0_2_all_pairs_gamma0p05_m0p2": {"gamma": 0.05, "margin": 0.2, "temperature": 0.2, "pair_source": "all_a0_pairs"},
    "C0_3_all_pairs_gamma0p10_m0p2": {"gamma": 0.10, "margin": 0.2, "temperature": 0.2, "pair_source": "all_a0_pairs"},
    "C0_4_pairwise_low_only_gamma0p05_m0p2": {"gamma": 0.05, "margin": 0.2, "temperature": 0.2, "pair_source": "pairwise_low_only"},
    "C0_5_evidence_positive_plus_pairwise_low_gamma0p05_m0p2": {
        "gamma": 0.05,
        "margin": 0.2,
        "temperature": 0.2,
        "pair_source": "evidence_positive_plus_pairwise_low",
    },
    "C0_6_random_pair_control_gamma0p05_m0p2": {
        "gamma": 0.05,
        "margin": 0.2,
        "temperature": 0.2,
        "pair_source": "random_low_high_pairs",
    },
    "C0_7_same_subject_only_gamma0p05_m0p2": {"gamma": 0.05, "margin": 0.2, "temperature": 0.2, "pair_source": "same_subject_only"},
    "C0_8_high_weight_only_gamma0p05_m0p2": {"gamma": 0.05, "margin": 0.2, "temperature": 0.2, "pair_source": "high_weight_only_p75"},
    "C0_9_same_subject_high_weight_gamma0p05_m0p2": {
        "gamma": 0.05,
        "margin": 0.2,
        "temperature": 0.2,
        "pair_source": "same_subject_high_weight_p75",
    },
    "C0_10_exclude_format_auxiliary_gamma0p05_m0p2": {
        "gamma": 0.05,
        "margin": 0.2,
        "temperature": 0.2,
        "pair_source": "exclude_format_auxiliary",
    },
    "C0_11_exclude_answer_key_dependent_gamma0p05_m0p2": {
        "gamma": 0.05,
        "margin": 0.2,
        "temperature": 0.2,
        "pair_source": "exclude_answer_key_dependent",
    },
    "C0_12_random_matched_metric_rubric_gamma0p05_m0p2": {
        "gamma": 0.05,
        "margin": 0.2,
        "temperature": 0.2,
        "pair_source": "random_matched_metric_rubric",
    },
    "C0_13_random_matched_metric_rubric_subject_gamma0p05_m0p2": {
        "gamma": 0.05,
        "margin": 0.2,
        "temperature": 0.2,
        "pair_source": "random_matched_metric_rubric_subject",
    },
    "C0_14_same_question_group_upper_bound_gamma0p05_m0p2": {
        "gamma": 0.05,
        "margin": 0.2,
        "temperature": 0.2,
        "pair_source": "same_question_group_upper_bound",
        "allow_empty_pairs": True,
    },
}

DEV_FIELDS = [
    "config_name",
    "seed",
    "gamma",
    "margin",
    "temperature",
    "pair_source",
    "MAE",
    "QWK",
    "accuracy",
    "low_to_high_count",
    "low_to_high_rate",
    "label2_recall",
    "label2_pred_ge4_rate",
    "monotonic_violation_rate",
    "mean_s_label2",
    "mean_s_label4_5",
    "mean_g_i3_label2",
]

PAIR_FIELDS = [
    "config_name",
    "seed",
    "pair_source",
    "train_pair_count",
    "dev_d1_pair_count",
    "margin",
    "train_pair_gap_mean",
    "train_pair_gap_p10",
    "train_pair_gap_violation_rate_at_margin",
    "dev_d1_s_gap_control_minus_hidden_mean",
    "dev_d1_s_gap_control_minus_hidden_median",
    "dev_d1_s_gap_violation_rate",
    "dev_d1_g_i3_hidden_mean",
    "dev_d1_g_i3_control_mean",
    "d1_hidden_vs_control_s_auc",
    "d1_pairwise_low_vs_control_s_auc",
    "d1_evidence_positive_vs_control_s_auc",
    "pair_source_available_pair_count",
    "pair_weight_mean",
    "pair_weight_p50",
    "pair_weight_p75",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value: Any, default: float = 1.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except Exception:
        return default


def safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("sample_id") or row.get("record_id") or row.get("id") or "")


def row_label(row: dict[str, Any]) -> int:
    return int(float(row.get("label_5", row.get("label", 0))))


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def text_hash(value: Any) -> str:
    return hashlib.sha1(str(value or "").strip().encode("utf-8")).hexdigest()


def sample_subject(sample: dict[str, Any]) -> str:
    return str(sample.get("subject_canonical") or sample.get("subject") or "").strip()


def sample_group_key(sample: dict[str, Any]) -> str:
    return str(sample.get("question_group_id") or sample.get("question_key") or "").strip()


def sample_match_key(sample: dict[str, Any], include_subject: bool = False) -> tuple[str, ...]:
    key = (
        str(sample.get("metric") or "").strip(),
        str(sample.get("language") or "").strip(),
        text_hash(sample.get("rubric_text") or sample.get("rubric") or ""),
    )
    if include_subject:
        return (*key, sample_subject(sample))
    return key


def sample_len(sample: dict[str, Any]) -> int:
    return len(str(sample.get("answer") or "").split())


def rows_p75(rows: list[dict[str, str]]) -> float:
    values = [safe_float(row.get("pair_weight"), float("nan")) for row in rows]
    values = [value for value in values if not math.isnan(value)]
    return percentile(values, 75) if values else float("inf")


def make_pair_row(low_id: str, high_id: str, weight: float, use: str, candidate_type: str, mode: str) -> dict[str, Any]:
    return {
        "low_sample_id": low_id,
        "high_sample_id": high_id,
        "pair_weight": weight,
        "recommended_pair_use": use,
        "low_candidate_type": candidate_type,
        "low_failure_mode_auto": mode,
    }


def random_matched_pairs(train_samples: list[dict[str, Any]], target_n: int, seed: int, include_subject: bool) -> list[dict[str, Any]]:
    rng = random.Random(seed + (17 if include_subject else 11))
    lows = [sample for sample in train_samples if int(sample["label"]) <= 2]
    highs = [sample for sample in train_samples if int(sample["label"]) >= 4]
    high_by_key: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for sample in highs:
        high_by_key.setdefault(sample_match_key(sample, include_subject=include_subject), []).append(sample)
    candidates: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for low in lows:
        matched = high_by_key.get(sample_match_key(low, include_subject=include_subject), [])
        if matched:
            candidates.append((low, matched))
    if not candidates:
        return []
    selected: list[dict[str, Any]] = []
    for _ in range(max(0, target_n)):
        low, high_pool = rng.choice(candidates)
        high = min(rng.sample(high_pool, k=len(high_pool)), key=lambda item: abs(sample_len(item) - sample_len(low)))
        selected.append(make_pair_row(low["sample_id"], high["sample_id"], 1.0, "random_matched_control", "random_low", "random_control"))
    return selected


def same_question_group_pairs(train_samples: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    del seed
    groups: dict[tuple[str, str, str], dict[str, list[dict[str, Any]]]] = {}
    for sample in train_samples:
        group = sample_group_key(sample)
        if not group:
            continue
        rubric_text = sample.get("rubric_text") or sample.get("rubric") or sample.get("rubric_canonical") or ""
        key = (group, str(sample.get("metric") or ""), text_hash(rubric_text))
        bucket = groups.setdefault(key, {"low": [], "high": []})
        if int(sample["label"]) <= 2:
            bucket["low"].append(sample)
        elif int(sample["label"]) >= 4:
            bucket["high"].append(sample)
    pairs: list[dict[str, Any]] = []
    for bucket in groups.values():
        for low in bucket["low"]:
            high_pool = sorted(bucket["high"], key=lambda item: abs(sample_len(item) - sample_len(low)))[:5]
            for high in high_pool:
                pairs.append(make_pair_row(low["sample_id"], high["sample_id"], 1.0, "same_question_upper_bound", "same_question_low", "upper_bound"))
    return pairs


def load_init_checkpoint(model: BoundaryLinkingOrdinalModel, path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing Exp16A qmr init checkpoint: {path}")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:10]}")
    return f"loaded missing_keys={len(missing)}"


def prediction_rows(outputs: dict[str, torch.Tensor], labels: torch.Tensor, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    probs = outputs["probs"].detach().float().cpu().numpy()
    pred = outputs["pred_label"].detach().cpu().numpy()
    s = outputs["quality_score_s"].detach().float().cpu().numpy()
    tau = outputs["thresholds_tau"].detach().float().cpu().numpy()
    alpha = outputs["scale_alpha"].detach().float().cpu().numpy()
    gold = labels.detach().cpu().numpy()
    rows: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        row = {
            "sample_id": sample["sample_id"],
            "question_key": sample["question_key"],
            "boundary_key": sample["boundary_key"],
            "metric": sample["metric"],
            "gold_label": int(gold[idx]),
            "pred_label": int(pred[idx]),
            "probs": [float(value) for value in probs[idx].tolist()],
            "quality_score_s": float(s[idx]),
            "tau1": float(tau[idx, 0]),
            "tau2": float(tau[idx, 1]),
            "tau3": float(tau[idx, 2]),
            "tau4": float(tau[idx, 3]),
            "scale_alpha": float(alpha[idx]),
            "margin_tau2": float(s[idx] - tau[idx, 1]),
            "margin_tau3": float(s[idx] - tau[idx, 2]),
            "g_i3": float(alpha[idx] * (s[idx] - tau[idx, 2])),
        }
        row["is_low_to_high"] = bool(row["gold_label"] <= 2 and row["pred_label"] >= 4)
        rows.append(row)
    return rows


@torch.no_grad()
def evaluate_model(model: BoundaryLinkingOrdinalModel, loader: DataLoader, device: torch.device, args: Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    predictions: list[dict[str, Any]] = []
    for batch in loader:
        batch = move_batch(batch, device)
        with autocast_context(args, device):
            outputs = model(
                quality_input_ids=batch["quality_input_ids"],
                quality_attention_mask=batch["quality_attention_mask"],
                boundary_input_ids=batch["boundary_input_ids"],
                boundary_attention_mask=batch["boundary_attention_mask"],
            )
        predictions.extend(prediction_rows(outputs, batch["labels"], batch["samples"]))
    return compute_metrics(predictions, split="dev"), predictions


def dev_metric_row(config_name: str, seed: int, cfg: dict[str, Any], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = compute_metrics(predictions, split="dev")
    label2 = [row for row in predictions if int(row["gold_label"]) == 2]
    high = [row for row in predictions if int(row["gold_label"]) >= 4]
    low = [row for row in predictions if int(row["gold_label"]) <= 2]
    label2_pred_ge4 = sum(1 for row in label2 if int(row["pred_label"]) >= 4)
    return {
        "config_name": config_name,
        "seed": seed,
        "gamma": cfg["gamma"],
        "margin": cfg["margin"],
        "temperature": cfg["temperature"],
        "pair_source": cfg["pair_source"],
        "MAE": metrics.get("MAE"),
        "QWK": metrics.get("QWK"),
        "accuracy": metrics.get("Accuracy"),
        "low_to_high_count": metrics.get("low_to_high_count"),
        "low_to_high_rate": metrics.get("low_to_high_rate"),
        "label2_recall": sum(1 for row in label2 if int(row["pred_label"]) == 2) / len(label2) if label2 else float("nan"),
        "label2_pred_ge4_rate": label2_pred_ge4 / len(label2) if label2 else float("nan"),
        "monotonic_violation_rate": metrics.get("monotonic_violation_rate", 0.0),
        "mean_s_label2": safe_mean([float(row["quality_score_s"]) for row in label2]),
        "mean_s_label4_5": safe_mean([float(row["quality_score_s"]) for row in high]),
        "mean_g_i3_label2": safe_mean([float(row["g_i3"]) for row in label2 or low]),
    }


def load_a0_pairs(a0_dir: Path, pair_source: str, train_samples: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    by_id = {sample["sample_id"]: sample for sample in train_samples}
    path = a0_dir / "train_hidden_failure_pairs.csv"
    rows = read_csv_rows(path) if path.exists() else []
    valid_rows = [row for row in rows if row.get("low_sample_id", "") in by_id and row.get("high_sample_id", "") in by_id]
    p75 = rows_p75(valid_rows)
    if pair_source == "none":
        return []
    if pair_source == "all_a0_pairs":
        selected = valid_rows
    elif pair_source == "pairwise_low_only":
        selected = [row for row in valid_rows if row.get("recommended_pair_use") == "pairwise_low"]
    elif pair_source == "evidence_positive_plus_pairwise_low":
        selected = [
            row
            for row in valid_rows
            if row.get("recommended_pair_use") == "pairwise_low"
            or row.get("low_candidate_type") in {"weak_evidence_positive", "strong_evidence_positive"}
        ]
    elif pair_source == "same_subject_only":
        selected = [row for row in valid_rows if truthy(row.get("same_subject"))]
    elif pair_source == "high_weight_only_p75":
        selected = [row for row in valid_rows if safe_float(row.get("pair_weight"), 0.0) >= p75]
    elif pair_source == "same_subject_high_weight_p75":
        selected = [row for row in valid_rows if truthy(row.get("same_subject")) and safe_float(row.get("pair_weight"), 0.0) >= p75]
    elif pair_source == "exclude_format_auxiliary":
        selected = [
            row
            for row in valid_rows
            if row.get("low_candidate_type") != "format_auxiliary" and row.get("low_failure_mode_auto") != "format_violation"
        ]
    elif pair_source == "exclude_answer_key_dependent":
        selected = [
            row
            for row in valid_rows
            if row.get("low_candidate_type") != "answer_key_dependent"
            and row.get("low_failure_mode_auto") != "answer_key_or_reference_mismatch"
        ]
    elif pair_source == "weak_evidence_only":
        selected = [row for row in valid_rows if row.get("low_candidate_type") in {"weak_evidence_positive", "strong_evidence_positive"}]
    elif pair_source == "missing_key_point_only":
        selected = [row for row in valid_rows if "missing_key_point" in row.get("low_failure_mode_auto", "")]
    elif pair_source == "factual_or_rubric_mismatch_only":
        selected = [
            row
            for row in valid_rows
            if "factual_or_rubric_mismatch" in row.get("low_failure_mode_auto", "")
            or "answer_key_or_reference_mismatch" in row.get("low_failure_mode_auto", "")
        ]
    elif pair_source == "insufficient_evidence_only":
        selected = [row for row in valid_rows if "insufficient_evidence" in row.get("low_failure_mode_auto", "")]
    elif pair_source == "random_low_high_pairs":
        rng = random.Random(seed)
        low_ids = [sid for sid, sample in by_id.items() if int(sample["label"]) <= 2]
        high_ids = [sid for sid, sample in by_id.items() if int(sample["label"]) >= 4]
        target_n = len(valid_rows)
        selected = []
        if low_ids and high_ids:
            for _ in range(target_n):
                selected.append(make_pair_row(rng.choice(low_ids), rng.choice(high_ids), 1.0, "random_control", "random_low", "random_control"))
    elif pair_source == "random_matched_metric_rubric":
        selected = random_matched_pairs(train_samples, len(valid_rows), seed, include_subject=False)
    elif pair_source == "random_matched_metric_rubric_subject":
        selected = random_matched_pairs(train_samples, len([row for row in valid_rows if truthy(row.get("same_subject"))]), seed, include_subject=True)
    elif pair_source == "same_question_group_upper_bound":
        selected = same_question_group_pairs(train_samples, seed)
    else:
        raise ValueError(f"Unsupported pair_source: {pair_source}")

    pairs: list[dict[str, Any]] = []
    for row in selected:
        low_id = row.get("low_sample_id", "")
        high_id = row.get("high_sample_id", "")
        if low_id in by_id and high_id in by_id:
            pairs.append(
                {
                    "low": by_id[low_id],
                    "high": by_id[high_id],
                    "weight": max(0.05, safe_float(row.get("pair_weight"), 1.0)),
                    "recommended_pair_use": row.get("recommended_pair_use", ""),
                    "low_candidate_type": row.get("low_candidate_type", ""),
                    "low_failure_mode_auto": row.get("low_failure_mode_auto", ""),
                    "source_row": row,
                }
            )
    return pairs


class PairDataset(Dataset):
    def __init__(self, pairs: list[dict[str, Any]]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.pairs[idx]


class PairCollator:
    def __init__(self, base_collator: BoundaryLinkingCollator) -> None:
        self.base_collator = base_collator

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "low": self.base_collator([row["low"] for row in rows]),
            "high": self.base_collator([row["high"] for row in rows]),
            "pair_weights": torch.tensor([float(row["weight"]) for row in rows], dtype=torch.float32),
            "pairs": rows,
        }


def pair_sep_loss(model: BoundaryLinkingOrdinalModel, pair_batch: dict[str, Any], device: torch.device, args: Namespace, margin: float, temperature: float) -> torch.Tensor:
    low = move_batch(pair_batch["low"], device)
    high = move_batch(pair_batch["high"], device)
    weights = pair_batch["pair_weights"].to(device)
    with autocast_context(args, device):
        low_out = model(
            quality_input_ids=low["quality_input_ids"],
            quality_attention_mask=low["quality_attention_mask"],
            boundary_input_ids=low["boundary_input_ids"],
            boundary_attention_mask=low["boundary_attention_mask"],
        )
        high_out = model(
            quality_input_ids=high["quality_input_ids"],
            quality_attention_mask=high["quality_attention_mask"],
            boundary_input_ids=high["boundary_input_ids"],
            boundary_attention_mask=high["boundary_attention_mask"],
        )
        gap = high_out["quality_score_s"] - low_out["quality_score_s"]
        raw = F.softplus((float(margin) - gap) / max(float(temperature), 1e-6))
        return (raw * weights).sum() / weights.sum().clamp_min(1e-8)


def score_is_better(metrics: dict[str, Any], best: float | None, save_best_by: str) -> tuple[bool, float]:
    if save_best_by == "dev_qwk":
        score = float(metrics["QWK"])
        return best is None or score > best, score
    if save_best_by == "latest":
        return True, float(metrics.get("epoch", 0))
    score = float(metrics["MAE"])
    return best is None or score < best, score


def load_dev_d1_pairs(d1_dir: Path) -> list[dict[str, str]]:
    path = d1_dir / "d1_matched_case_control_review.csv"
    if not path.exists():
        return []
    annotation_paths = [
        d1_dir / "d1_hidden_failure_annotation_template_filled_human_rationale_recovered.csv",
        d1_dir / "d1_hidden_failure_annotation_template_filled_codex_independent.csv",
        d1_dir / "d1_hidden_failure_annotation_template_filled.csv",
    ]
    annotations: dict[str, dict[str, str]] = {}
    for annotation_path in annotation_paths:
        if annotation_path.exists():
            annotations = {row.get("sample_id", ""): row for row in read_csv_rows(annotation_path)}
            break
    rows = read_csv_rows(path)
    pairs: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        pair = (row.get("case_sample_id", ""), row.get("control_sample_id", ""))
        if all(pair) and pair not in seen:
            annotation = annotations.get(pair[0], {})
            pairs.append(
                {
                    "case": pair[0],
                    "control": pair[1],
                    "primary_failure_mode_manual": annotation.get("primary_failure_mode_manual", ""),
                    "recommended_training_use_manual": annotation.get("recommended_training_use_manual", ""),
                    "trainability_manual": annotation.get("trainability_manual", ""),
                }
            )
            seen.add(pair)
    return pairs


def separation_auc(case_scores: list[float], control_scores: list[float]) -> float:
    if not case_scores or not control_scores:
        return float("nan")
    wins = 0.0
    total = 0
    for case_score in case_scores:
        for control_score in control_scores:
            total += 1
            if control_score > case_score:
                wins += 1.0
            elif control_score == case_score:
                wins += 0.5
    return wins / total if total else float("nan")


def pair_weights(train_pairs: list[dict[str, Any]]) -> list[float]:
    return [float(pair.get("weight", 1.0)) for pair in train_pairs]


def pair_eval_row(
    config_name: str,
    seed: int,
    pair_source: str,
    margin: float,
    train_pairs: list[dict[str, Any]],
    train_pair_predictions: dict[str, float],
    dev_predictions: list[dict[str, Any]],
    d1_pairs: list[dict[str, str]],
) -> dict[str, Any]:
    train_gaps = [
        train_pair_predictions[pair["high"]["sample_id"]] - train_pair_predictions[pair["low"]["sample_id"]]
        for pair in train_pairs
        if pair["high"]["sample_id"] in train_pair_predictions and pair["low"]["sample_id"] in train_pair_predictions
    ]
    dev_s = {row["sample_id"]: float(row["quality_score_s"]) for row in dev_predictions}
    dev_g = {row["sample_id"]: float(row.get("g_i3", float("nan"))) for row in dev_predictions}
    usable_d1_pairs = [pair for pair in d1_pairs if pair["case"] in dev_s and pair["control"] in dev_s]
    dev_gaps = [dev_s[pair["control"]] - dev_s[pair["case"]] for pair in usable_d1_pairs]
    case_scores = [dev_s[pair["case"]] for pair in usable_d1_pairs]
    control_scores = [dev_s[pair["control"]] for pair in usable_d1_pairs]
    pairwise_low_pairs = [pair for pair in usable_d1_pairs if pair.get("recommended_training_use_manual") == "pairwise_low"]
    evidence_pairs = [
        pair
        for pair in usable_d1_pairs
        if pair.get("recommended_training_use_manual") in {"evidence_positive", "weak_evidence_positive"}
        or pair.get("trainability_manual") in {"strong_train_signal", "weak_train_signal"}
    ]
    weights = pair_weights(train_pairs)
    return {
        "config_name": config_name,
        "seed": seed,
        "pair_source": pair_source,
        "train_pair_count": len(train_pairs),
        "dev_d1_pair_count": len(dev_gaps),
        "margin": margin,
        "train_pair_gap_mean": safe_mean(train_gaps),
        "train_pair_gap_p10": percentile(train_gaps, 10),
        "train_pair_gap_violation_rate_at_margin": sum(1 for gap in train_gaps if gap < margin) / len(train_gaps) if train_gaps else float("nan"),
        "dev_d1_s_gap_control_minus_hidden_mean": safe_mean(dev_gaps),
        "dev_d1_s_gap_control_minus_hidden_median": percentile(dev_gaps, 50),
        "dev_d1_s_gap_violation_rate": sum(1 for gap in dev_gaps if gap <= 0.0) / len(dev_gaps) if dev_gaps else float("nan"),
        "dev_d1_g_i3_hidden_mean": safe_mean([dev_g[pair["case"]] for pair in usable_d1_pairs if pair["case"] in dev_g]),
        "dev_d1_g_i3_control_mean": safe_mean([dev_g[pair["control"]] for pair in usable_d1_pairs if pair["control"] in dev_g]),
        "d1_hidden_vs_control_s_auc": separation_auc(case_scores, control_scores),
        "d1_pairwise_low_vs_control_s_auc": separation_auc(
            [dev_s[pair["case"]] for pair in pairwise_low_pairs],
            [dev_s[pair["control"]] for pair in pairwise_low_pairs],
        ),
        "d1_evidence_positive_vs_control_s_auc": separation_auc(
            [dev_s[pair["case"]] for pair in evidence_pairs],
            [dev_s[pair["control"]] for pair in evidence_pairs],
        ),
        "pair_source_available_pair_count": len(train_pairs),
        "pair_weight_mean": safe_mean(weights),
        "pair_weight_p50": percentile(weights, 50),
        "pair_weight_p75": percentile(weights, 75),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Exp17-C0 pairwise-low quality separation.")
    parser.add_argument("--config_name", choices=sorted(C0_CONFIGS), required=True)
    parser.add_argument("--model_name_or_path", default="__tiny_random__")
    parser.add_argument("--init_checkpoint", type=Path, default=DEFAULT_INIT_CKPT)
    parser.add_argument("--train_path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--dev_path", type=Path, default=DEFAULT_DEV_PATH)
    parser.add_argument("--a0_dir", type=Path, default=DEFAULT_A0_DIR)
    parser.add_argument("--d1_dir", type=Path, default=DEFAULT_D1_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_length_quality", type=int, default=2048)
    parser.add_argument("--max_length_boundary", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--pair_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=32)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--precision", choices=["auto", "fp16", "bf16", "fp32"], default="fp32")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--save_best_by", choices=["dev_mae", "dev_qwk", "latest"], default="dev_mae")
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--max_train_steps", type=int, default=0)
    parser.add_argument("--log_steps", type=int, default=5)
    parser.add_argument("--allow_empty_pairs", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    return parser


def train_one(args: Namespace) -> dict[str, Any]:
    cfg = C0_CONFIGS[args.config_name]
    gamma = float(cfg["gamma"])
    margin = float(cfg["margin"])
    temperature = float(cfg["temperature"])
    pair_source = str(cfg["pair_source"])
    run_dir = Path(args.output_dir) / "runs" / args.config_name / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(args.seed))
    tokenizer = make_tokenizer(str(args.model_name_or_path), args.local_files_only, args.trust_remote_code)
    model = BoundaryLinkingOrdinalModel.from_model_name(
        str(args.model_name_or_path),
        variant="qmr",
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=bool(args.local_files_only),
    )
    init_status = load_init_checkpoint(model, Path(args.init_checkpoint))
    if bool(args.gradient_checkpointing):
        if hasattr(model.encoder.config, "use_cache"):
            model.encoder.config.use_cache = False
        if hasattr(model.encoder, "gradient_checkpointing_enable"):
            model.encoder.gradient_checkpointing_enable()

    train_samples = load_samples(args.train_path, "qmr", None, limit=int(args.max_train_samples) or None)
    dev_samples = load_samples(args.dev_path, "qmr", None, limit=int(args.max_eval_samples) or None)
    collator = BoundaryLinkingCollator(tokenizer, args.max_length_quality, args.max_length_boundary)
    train_loader = DataLoader(BoundaryLinkingDataset(train_samples), batch_size=int(args.batch_size), shuffle=True, collate_fn=collator)
    dev_loader = DataLoader(BoundaryLinkingDataset(dev_samples), batch_size=int(args.eval_batch_size), shuffle=False, collate_fn=collator)
    train_pairs = load_a0_pairs(args.a0_dir, pair_source, train_samples, int(args.seed))
    allow_empty_pairs = bool(args.allow_empty_pairs or cfg.get("allow_empty_pairs", False))
    if gamma > 0.0 and not train_pairs and not allow_empty_pairs:
        raise RuntimeError(
            f"Exp17-C0 config {args.config_name} pair_source={pair_source} produced 0 train pairs. "
            "Use --allow_empty_pairs only for diagnostics."
        )
    pair_loader = (
        DataLoader(PairDataset(train_pairs), batch_size=int(args.pair_batch_size), shuffle=True, collate_fn=PairCollator(collator))
        if train_pairs
        else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))

    best_state: dict[str, torch.Tensor] | None = None
    best_score: float | None = None
    best_metrics: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    global_step = 0
    max_steps = int(args.max_train_steps) if int(args.max_train_steps) > 0 else None
    epochs = int(np.ceil(float(args.epochs)))
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = {"loss": 0.0, "ord": 0.0, "sep": 0.0}
        batch_count = 0
        pair_iter = iter(pair_loader) if pair_loader is not None else None
        start = time.time()
        total_batches = len(train_loader)
        print(
            f"[exp17-c0] {args.config_name} epoch {epoch}/{epochs} start batches={total_batches} "
            f"batch_size={args.batch_size} pair_batch_size={args.pair_batch_size} grad_accum={args.grad_accum_steps} "
            f"gamma={gamma} pair_source={pair_source} pairs={len(train_pairs)}",
            flush=True,
        )
        for batch_idx, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, device)
            with autocast_context(args, device):
                outputs = model(
                    quality_input_ids=batch["quality_input_ids"],
                    quality_attention_mask=batch["quality_attention_mask"],
                    boundary_input_ids=batch["boundary_input_ids"],
                    boundary_attention_mask=batch["boundary_attention_mask"],
                )
                loss_ord = ordinal_bce_loss(outputs["logits"], batch["targets"], batch["labels"], class_weights=None)
            if gamma > 0.0 and pair_loader is not None and pair_iter is not None:
                try:
                    pair_batch = next(pair_iter)
                except StopIteration:
                    pair_iter = iter(pair_loader)
                    pair_batch = next(pair_iter)
                loss_sep = pair_sep_loss(model, pair_batch, device, args, margin, temperature)
            else:
                loss_sep = outputs["quality_score_s"].sum() * 0.0
            loss = loss_ord + gamma * loss_sep
            (loss / int(args.grad_accum_steps)).backward()
            running["loss"] += float(loss.detach().cpu())
            running["ord"] += float(loss_ord.detach().cpu())
            running["sep"] += float(loss_sep.detach().cpu())
            batch_count += 1
            if batch_idx % int(args.grad_accum_steps) == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if max_steps is not None and global_step >= max_steps:
                    break
            if batch_idx == 1 or batch_idx % int(args.log_steps) == 0 or batch_idx == total_batches:
                elapsed = time.time() - start
                rate = batch_idx / max(elapsed, 1e-6)
                eta = (total_batches - batch_idx) / max(rate, 1e-6)
                print(
                    f"[exp17-c0] {args.config_name} epoch {epoch}/{epochs} batch {batch_idx}/{total_batches} "
                    f"loss={float(loss.detach().cpu()):.4f} ord={float(loss_ord.detach().cpu()):.4f} "
                    f"sep={float(loss_sep.detach().cpu()):.4f} global_step={global_step} "
                    f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
                    flush=True,
                )
        if batch_count % int(args.grad_accum_steps) != 0 and (max_steps is None or global_step < max_steps):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
        dev_metrics, _ = evaluate_model(model, dev_loader, device, args)
        epoch_metrics = {
            **dev_metrics,
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": running["loss"] / max(1, batch_count),
            "train_ord_loss": running["ord"] / max(1, batch_count),
            "train_sep_loss": running["sep"] / max(1, batch_count),
        }
        history.append(epoch_metrics)
        print(
            f"[exp17-c0] {args.config_name} epoch {epoch}/{epochs} dev MAE={float(dev_metrics['MAE']):.4f} "
            f"QWK={float(dev_metrics['QWK']):.4f} low_to_high={dev_metrics['low_to_high_count']}/{dev_metrics['true_low_n']}",
            flush=True,
        )
        better, score = score_is_better(epoch_metrics, best_score, str(args.save_best_by))
        if better:
            best_score = score
            best_metrics = epoch_metrics
            best_state = copy.deepcopy(model.state_dict())
        if max_steps is not None and global_step >= max_steps:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    final_metrics, dev_predictions = evaluate_model(model, dev_loader, device, args)
    train_pair_samples: list[dict[str, Any]] = []
    seen_pair_ids: set[str] = set()
    for pair in train_pairs:
        for sample in [pair["low"], pair["high"]]:
            if sample["sample_id"] not in seen_pair_ids:
                train_pair_samples.append(sample)
                seen_pair_ids.add(sample["sample_id"])
    train_pair_loader = DataLoader(BoundaryLinkingDataset(train_pair_samples), batch_size=int(args.eval_batch_size), shuffle=False, collate_fn=collator)
    _, train_pair_preds = evaluate_model(model, train_pair_loader, device, args) if train_pair_samples else ({}, [])
    train_s = {row["sample_id"]: float(row["quality_score_s"]) for row in train_pair_preds}
    pair_row = pair_eval_row(
        args.config_name,
        int(args.seed),
        pair_source,
        margin,
        train_pairs,
        train_s,
        dev_predictions,
        load_dev_d1_pairs(args.d1_dir),
    )
    dev_row = dev_metric_row(args.config_name, int(args.seed), cfg, dev_predictions)

    write_json(run_dir / "metrics_dev.json", final_metrics)
    write_json(run_dir / "training_history.json", history)
    write_json(
        run_dir / "config.json",
        {
            **vars(args),
            "output_dir": relpath(args.output_dir),
            "run_dir": relpath(run_dir),
            "train_path": relpath(args.train_path),
            "dev_path": relpath(args.dev_path),
            "a0_dir": relpath(args.a0_dir),
            "d1_dir": relpath(args.d1_dir),
            "init_checkpoint": relpath(args.init_checkpoint),
            "init_status": init_status,
            "c0_config": cfg,
            "train_pair_count": len(train_pairs),
            "available_pair_count": len(train_pairs),
            "best_metrics": best_metrics,
            "test_read": False,
            "dev_annotation_used_as_train_label": False,
            "human_rationale_used_as_ranker_input": False,
            "hidden_failure_evidence_head": False,
            "suppression": False,
            "pair_loss_applies_to": "quality_score_s_only",
        },
    )
    write_csv(run_dir / "exp17_c0_dev_metrics.csv", [dev_row], fieldnames=DEV_FIELDS)
    write_csv(run_dir / "exp17_c0_pair_eval.csv", [pair_row], fieldnames=PAIR_FIELDS)
    write_text_report = [
        "# Exp17-C0 Pairwise Separation Run",
        "",
        f"- config_name: `{args.config_name}`",
        f"- gamma: {gamma}",
        f"- margin: {margin}",
        f"- temperature: {temperature}",
        f"- pair_source: `{pair_source}`",
        f"- train_pair_count: {len(train_pairs)}",
        f"- selected_by: `{args.save_best_by}`",
        "",
        "## Dev Metrics",
        "",
        f"- MAE: {dev_row['MAE']}",
        f"- QWK: {dev_row['QWK']}",
        f"- low_to_high_rate: {dev_row['low_to_high_rate']}",
        f"- label2_recall: {dev_row['label2_recall']}",
        f"- mean_g_i3_label2: {dev_row['mean_g_i3_label2']}",
        "",
        "## Pair Metrics",
        "",
        f"- train_pair_gap_mean: {pair_row['train_pair_gap_mean']}",
        f"- dev_d1_s_gap_control_minus_hidden_mean: {pair_row['dev_d1_s_gap_control_minus_hidden_mean']}",
        f"- d1_hidden_vs_control_s_auc: {pair_row['d1_hidden_vs_control_s_auc']}",
        "",
        "## Guardrails",
        "",
        "- Test split is not read.",
        "- Dev D1 annotations are used only for evaluation.",
        "- Human rationale text is not used as ranker input.",
        "- Pairwise separation loss applies only to quality score `s`.",
    ]
    (run_dir / "exp17_c0_report.md").write_text("\n".join(write_text_report), encoding="utf-8")
    print(json.dumps({"status": "COMPLETED", "config_name": args.config_name, "run_dir": relpath(run_dir)}, ensure_ascii=False, sort_keys=True))
    return {"status": "COMPLETED", "run_dir": relpath(run_dir)}


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    train_one(args)


if __name__ == "__main__":
    main()
