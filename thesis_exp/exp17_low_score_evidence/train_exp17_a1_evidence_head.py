"""Train Exp17-A1 hidden-failure evidence head under boundary linking.

Exp17-A1 keeps the Exp16A boundary-linking scoring path unchanged. The
additional evidence head is trained from train-side A0 weak labels only and is
used for diagnosis, not for score suppression.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

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
DEFAULT_A0_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp17_a0_train_hidden_failure_signals_seed42")
DEFAULT_D1_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/d1_hidden_failure_audit_seed42_dev")
DEFAULT_OUTPUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp17_a1_evidence_head_seed42")
DEFAULT_INIT_CKPT = Path("thesis_exp/outputs/exp16_boundary_linking/runs/qmr/seed_42/checkpoint_best/state_dict.pt")

SCOUT_CONFIGS: dict[str, dict[str, Any]] = {
    "A1_0_baseline": {"beta": 0.0, "neg_ratio": 4, "positive_mode": "a0_weak"},
    "A1_1": {"beta": 0.05, "neg_ratio": 4, "positive_mode": "a0_weak"},
    "A1_2": {"beta": 0.10, "neg_ratio": 4, "positive_mode": "a0_weak"},
    "A1_3": {"beta": 0.20, "neg_ratio": 4, "positive_mode": "a0_weak"},
    "A1_4": {"beta": 0.10, "neg_ratio": 2, "positive_mode": "a0_weak"},
    "A1_5_all_low_aux_baseline": {"beta": 0.10, "neg_ratio": 4, "positive_mode": "all_low_aux"},
    "A1_6_random_positive_control": {"beta": 0.10, "neg_ratio": 4, "positive_mode": "random_positive_control"},
}

DEV_METRIC_FIELDS = [
    "config_name",
    "seed",
    "beta",
    "neg_ratio",
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

EVIDENCE_FIELDS = [
    "config_name",
    "seed",
    "h_auc_d1_hidden_vs_controls",
    "mean_h_d1_hidden",
    "mean_h_d1_controls",
    "mean_h_d1_possible_conflict",
    "mean_h_marketing_group",
    "mean_h_non_marketing_group",
    "mean_h_train_weak_positive",
    "mean_h_train_clean_high",
    "evidence_delta_hidden_minus_control",
]

CASE_SCORE_FIELDS = [
    "sample_id",
    "gold_label",
    "pred_label",
    "s",
    "tau2",
    "tau3",
    "tau4",
    "alpha",
    "g_i3",
    "h",
    "question_key",
    "metric",
    "boundary_key",
]


class EvidenceBoundaryLinkingModel(nn.Module):
    """Exp16A scorer plus an evidence head on the quality representation."""

    def __init__(self, base: BoundaryLinkingOrdinalModel) -> None:
        super().__init__()
        self.base = base
        self.evidence_head = nn.Linear(base.hidden_size, 1)

    def forward(
        self,
        quality_input_ids: torch.Tensor,
        quality_attention_mask: torch.Tensor,
        boundary_input_ids: torch.Tensor,
        boundary_attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        quality_h = self.base.encode(quality_input_ids, quality_attention_mask)
        if self.base.variant == "global":
            boundary_h = torch.zeros_like(quality_h)
        else:
            boundary_h = self.base.encode(boundary_input_ids, boundary_attention_mask)
        quality_score_s = self.base.quality_head(quality_h).squeeze(-1)
        thresholds_tau, scale_alpha = self.base.ordered_thresholds(boundary_h)
        logits = scale_alpha.unsqueeze(-1) * (quality_score_s.unsqueeze(-1) - thresholds_tau)
        probs = torch.sigmoid(logits)
        pred_label = 1 + (probs > 0.5).sum(dim=-1)
        evidence_logit = self.evidence_head(quality_h).squeeze(-1)
        evidence_h = torch.sigmoid(evidence_logit)
        return {
            "logits": logits,
            "probs": probs,
            "pred_label": pred_label,
            "quality_score_s": quality_score_s,
            "thresholds_tau": thresholds_tau,
            "scale_alpha": scale_alpha,
            "evidence_logit": evidence_logit,
            "evidence_h": evidence_h,
        }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except Exception:
        return default


def safe_mean(values: list[float]) -> float:
    clean = [float(v) for v in values if not math.isnan(float(v))]
    return float(np.mean(clean)) if clean else float("nan")


def safe_rate(num: int, den: int) -> float:
    return float(num / den) if den else float("nan")


def rank_auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    total = 0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
            total += 1
    return float(wins / total) if total else float("nan")


def sample_id(row: dict[str, Any]) -> str:
    return str(row.get("sample_id") or row.get("record_id") or row.get("id") or "")


def build_evidence_labels(
    train_rows: list[dict[str, Any]],
    a0_dir: Path,
    positive_mode: str,
    neg_ratio: int,
    seed: int,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    candidates = read_csv_rows(a0_dir / "train_hidden_failure_candidates.csv")
    controls = read_csv_rows(a0_dir / "train_clean_high_controls.csv")
    train_by_id = {sample_id(row): row for row in train_rows}

    a0_weak = [
        row
        for row in candidates
        if row.get("hidden_failure_candidate_type") == "weak_evidence_positive"
        and row.get("recommended_training_use") == "weak_evidence_positive"
    ]
    rng = random.Random(seed)

    if positive_mode == "a0_weak":
        positive_rows = a0_weak
        positive_ids = [row["sample_id"] for row in positive_rows if row.get("sample_id") in train_by_id]
        positive_weights = {
            row["sample_id"]: max(0.05, safe_float(row.get("confidence_weight"), 1.0))
            for row in positive_rows
            if row.get("sample_id") in train_by_id
        }
    elif positive_mode == "all_low_aux":
        positive_ids = [sid for sid, row in train_by_id.items() if int(float(row.get("label_5", row.get("label", 0)))) <= 2]
        positive_weights = {sid: 1.0 for sid in positive_ids}
    elif positive_mode == "random_positive_control":
        low_ids = [sid for sid, row in train_by_id.items() if int(float(row.get("label_5", row.get("label", 0)))) <= 2]
        rng.shuffle(low_ids)
        positive_ids = low_ids[: len(a0_weak)]
        positive_weights = {sid: 1.0 for sid in positive_ids}
    else:
        raise ValueError(f"Unsupported positive_mode: {positive_mode}")

    positive_ids = sorted(set(positive_ids))
    desired_negatives = max(1, len(positive_ids) * int(neg_ratio))
    control_rows = [row for row in controls if row.get("sample_id") in train_by_id]
    grouped_controls: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in control_rows:
        key = (row.get("metric", ""), row.get("language", ""), row.get("subject", ""))
        grouped_controls.setdefault(key, []).append(row)

    selected_negatives: list[dict[str, str]] = []
    used_negatives: set[str] = set()
    if positive_mode == "all_low_aux":
        high_ids = [sid for sid, row in train_by_id.items() if int(float(row.get("label_5", row.get("label", 0)))) >= 4]
        rng.shuffle(high_ids)
        selected_negatives = [{"sample_id": sid, "clean_high_confidence_weight": "1.0"} for sid in high_ids[:desired_negatives]]
    else:
        positives_for_matching = [
            row
            for row in candidates
            if row.get("sample_id") in positive_ids
        ]
        rng.shuffle(positives_for_matching)
        per_positive = max(1, int(neg_ratio))
        for pos in positives_for_matching:
            key = (pos.get("metric", ""), pos.get("language", ""), pos.get("subject", ""))
            pool = list(grouped_controls.get(key) or control_rows)
            rng.shuffle(pool)
            added_for_positive = 0
            for control in pool:
                cid = control.get("sample_id", "")
                if cid and cid not in used_negatives:
                    selected_negatives.append(control)
                    used_negatives.add(cid)
                    added_for_positive += 1
                if len(selected_negatives) >= desired_negatives:
                    break
                if added_for_positive >= per_positive:
                    break
            if len(selected_negatives) >= desired_negatives:
                break
        if len(selected_negatives) < desired_negatives:
            pool = list(control_rows)
            rng.shuffle(pool)
            for control in pool:
                cid = control.get("sample_id", "")
                if cid and cid not in used_negatives:
                    selected_negatives.append(control)
                    used_negatives.add(cid)
                if len(selected_negatives) >= desired_negatives:
                    break

    labels: dict[str, dict[str, float]] = {}
    for sid in positive_ids:
        labels[sid] = {"target": 1.0, "weight": float(positive_weights.get(sid, 1.0))}
    for row in selected_negatives:
        sid = row.get("sample_id", "")
        if sid and sid not in labels:
            labels[sid] = {
                "target": 0.0,
                "weight": max(0.05, safe_float(row.get("clean_high_confidence_weight"), 1.0)),
            }
    info = {
        "positive_mode": positive_mode,
        "positive_count": len(positive_ids),
        "negative_count": sum(1 for item in labels.values() if item["target"] == 0.0),
        "requested_neg_ratio": neg_ratio,
        "a0_weak_positive_count": len(a0_weak),
    }
    return labels, info


def add_evidence_to_samples(samples: list[dict[str, Any]], evidence_labels: dict[str, dict[str, float]]) -> None:
    for sample in samples:
        item = evidence_labels.get(sample["sample_id"])
        if item is None:
            sample["evidence_mask"] = 0.0
            sample["evidence_target"] = 0.0
            sample["evidence_weight"] = 0.0
        else:
            sample["evidence_mask"] = 1.0
            sample["evidence_target"] = float(item["target"])
            sample["evidence_weight"] = float(item["weight"])


class EvidenceCollator(BoundaryLinkingCollator):
    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        batch = super().__call__(rows)
        batch["evidence_mask"] = torch.tensor([float(row.get("evidence_mask", 0.0)) for row in rows], dtype=torch.float32)
        batch["evidence_targets"] = torch.tensor([float(row.get("evidence_target", 0.0)) for row in rows], dtype=torch.float32)
        batch["evidence_weights"] = torch.tensor([float(row.get("evidence_weight", 0.0)) for row in rows], dtype=torch.float32)
        return batch


def move_evidence_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    batch = move_batch(batch, device)
    for key in ["evidence_mask", "evidence_targets", "evidence_weights"]:
        batch[key] = batch[key].to(device)
    return batch


def evidence_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    mask = batch["evidence_mask"]
    if float(mask.sum().detach().cpu()) <= 0:
        return outputs["evidence_logit"].sum() * 0.0
    raw = F.binary_cross_entropy_with_logits(outputs["evidence_logit"], batch["evidence_targets"], reduction="none")
    weighted = raw * batch["evidence_weights"] * mask
    return weighted.sum() / (batch["evidence_weights"] * mask).sum().clamp_min(1e-8)


def prediction_rows(outputs: dict[str, torch.Tensor], labels: torch.Tensor, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    probs = outputs["probs"].detach().float().cpu().numpy()
    pred = outputs["pred_label"].detach().cpu().numpy()
    s = outputs["quality_score_s"].detach().float().cpu().numpy()
    tau = outputs["thresholds_tau"].detach().float().cpu().numpy()
    alpha = outputs["scale_alpha"].detach().float().cpu().numpy()
    h = outputs["evidence_h"].detach().float().cpu().numpy()
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
            "evidence_h": float(h[idx]),
        }
        row["is_low_to_high"] = bool(row["gold_label"] <= 2 and row["pred_label"] >= 4)
        rows.append(row)
    return rows


@torch.no_grad()
def evaluate_model(
    model: EvidenceBoundaryLinkingModel,
    loader: DataLoader,
    device: torch.device,
    split: str,
    args: Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        batch = move_evidence_batch(batch, device)
        with autocast_context(args, device):
            outputs = model(
                quality_input_ids=batch["quality_input_ids"],
                quality_attention_mask=batch["quality_attention_mask"],
                boundary_input_ids=batch["boundary_input_ids"],
                boundary_attention_mask=batch["boundary_attention_mask"],
            )
        rows.extend(prediction_rows(outputs, batch["labels"], batch["samples"]))
    return compute_metrics(rows, split=split), rows


def dev_metric_row(config_name: str, seed: int, beta: float, neg_ratio: int, predictions: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = compute_metrics(predictions, split="dev")
    label2 = [row for row in predictions if int(row["gold_label"]) == 2]
    high = [row for row in predictions if int(row["gold_label"]) >= 4]
    label2_pred_ge4 = sum(1 for row in label2 if int(row["pred_label"]) >= 4)
    return {
        "config_name": config_name,
        "seed": seed,
        "beta": beta,
        "neg_ratio": f"1:{neg_ratio}",
        "MAE": metrics.get("MAE"),
        "QWK": metrics.get("QWK"),
        "accuracy": metrics.get("Accuracy"),
        "low_to_high_count": metrics.get("low_to_high_count"),
        "low_to_high_rate": metrics.get("low_to_high_rate"),
        "label2_recall": metrics.get("label2_recall"),
        "label2_pred_ge4_rate": safe_rate(label2_pred_ge4, len(label2)),
        "monotonic_violation_rate": metrics.get("monotonic_violation_rate"),
        "mean_s_label2": safe_mean([float(row["quality_score_s"]) for row in label2]),
        "mean_s_label4_5": safe_mean([float(row["quality_score_s"]) for row in high]),
        "mean_g_i3_label2": safe_mean([float(row["margin_tau3"]) for row in label2]),
    }


def load_dev_eval_sets(d1_dir: Path) -> dict[str, set[str]]:
    annotation_path = d1_dir / "d1_hidden_failure_annotation_template_filled_human_rationale_recovered.csv"
    controls_path = d1_dir / "d1_matched_case_control_review.csv"
    hidden: set[str] = set()
    possible_conflict: set[str] = set()
    marketing: set[str] = set()
    non_marketing: set[str] = set()
    if annotation_path.exists():
        for row in read_csv_rows(annotation_path):
            sid = row.get("sample_id", "")
            if not sid:
                continue
            is_conflict = str(row.get("possible_label_conflict_manual", "")).strip() == "1"
            use = row.get("recommended_training_use_manual", "")
            if is_conflict:
                possible_conflict.add(sid)
            elif use in {"evidence_positive", "pairwise_low", "format_auxiliary"}:
                hidden.add(sid)
            question = (row.get("question", "") + " " + row.get("subject", "")).lower()
            if "marketing manager" in question or "marketing" in question:
                marketing.add(sid)
            else:
                non_marketing.add(sid)
    controls: set[str] = set()
    if controls_path.exists():
        for row in read_csv_rows(controls_path):
            if row.get("case_sample_id") in hidden and row.get("control_sample_id"):
                controls.add(row["control_sample_id"])
    return {
        "hidden": hidden,
        "controls": controls,
        "possible_conflict": possible_conflict,
        "marketing": marketing,
        "non_marketing": non_marketing,
    }


def mean_h(pred_by_id: dict[str, dict[str, Any]], ids: set[str]) -> float:
    return safe_mean([float(pred_by_id[sid]["evidence_h"]) for sid in ids if sid in pred_by_id])


def evidence_eval_row(
    config_name: str,
    seed: int,
    dev_predictions: list[dict[str, Any]],
    train_predictions: list[dict[str, Any]],
    d1_sets: dict[str, set[str]],
    evidence_labels: dict[str, dict[str, float]],
) -> dict[str, Any]:
    dev_by_id = {row["sample_id"]: row for row in dev_predictions}
    train_by_id = {row["sample_id"]: row for row in train_predictions}
    hidden_values = [float(dev_by_id[sid]["evidence_h"]) for sid in d1_sets["hidden"] if sid in dev_by_id]
    control_values = [float(dev_by_id[sid]["evidence_h"]) for sid in d1_sets["controls"] if sid in dev_by_id]
    train_pos = {sid for sid, item in evidence_labels.items() if item["target"] == 1.0}
    train_neg = {sid for sid, item in evidence_labels.items() if item["target"] == 0.0}
    return {
        "config_name": config_name,
        "seed": seed,
        "h_auc_d1_hidden_vs_controls": rank_auc(hidden_values, control_values),
        "mean_h_d1_hidden": safe_mean(hidden_values),
        "mean_h_d1_controls": safe_mean(control_values),
        "mean_h_d1_possible_conflict": mean_h(dev_by_id, d1_sets["possible_conflict"]),
        "mean_h_marketing_group": mean_h(dev_by_id, d1_sets["marketing"]),
        "mean_h_non_marketing_group": mean_h(dev_by_id, d1_sets["non_marketing"]),
        "mean_h_train_weak_positive": mean_h(train_by_id, train_pos),
        "mean_h_train_clean_high": mean_h(train_by_id, train_neg),
        "evidence_delta_hidden_minus_control": safe_mean(hidden_values) - safe_mean(control_values),
    }


def case_score_rows(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in predictions:
        rows.append(
            {
                "sample_id": row["sample_id"],
                "gold_label": row["gold_label"],
                "pred_label": row["pred_label"],
                "s": row["quality_score_s"],
                "tau2": row["tau2"],
                "tau3": row["tau3"],
                "tau4": row["tau4"],
                "alpha": row["scale_alpha"],
                "g_i3": row["margin_tau3"],
                "h": row["evidence_h"],
                "question_key": row["question_key"],
                "metric": row["metric"],
                "boundary_key": row["boundary_key"],
            }
        )
    return rows


def score_is_better(metrics: dict[str, Any], best: float | None, save_best_by: str) -> tuple[bool, float]:
    if save_best_by == "dev_qwk":
        score = float(metrics["QWK"])
        return best is None or score > best, score
    if save_best_by == "dev_mae":
        score = float(metrics["MAE"])
        return best is None or score < best, score
    raise ValueError(f"Unsupported save_best_by: {save_best_by}")


def load_init_checkpoint(model: EvidenceBoundaryLinkingModel, path: Path, allow_missing: bool) -> str:
    if not path.exists():
        if allow_missing:
            return "missing_allowed"
        raise FileNotFoundError(
            f"Missing Exp16A qmr init checkpoint: {path}. "
            "Run/sync Exp16A qmr seed42 first or pass --allow_missing_init_checkpoint for tiny smoke only."
        )
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    try:
        missing, unexpected = model.base.load_state_dict(state, strict=False)
    except RuntimeError:
        if allow_missing:
            return "incompatible_checkpoint_allowed_for_smoke"
        raise
    if unexpected:
        raise RuntimeError(f"Unexpected keys while loading base checkpoint: {unexpected[:10]}")
    return f"loaded missing_base_keys={len(missing)}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Exp17-A1 evidence head without score suppression.")
    parser.add_argument("--config_name", choices=sorted(SCOUT_CONFIGS), required=True)
    parser.add_argument("--model_name_or_path", default="__tiny_random__")
    parser.add_argument("--init_checkpoint", type=Path, default=DEFAULT_INIT_CKPT)
    parser.add_argument("--allow_missing_init_checkpoint", action="store_true")
    parser.add_argument("--train_path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--dev_path", type=Path, default=DEFAULT_DEV_PATH)
    parser.add_argument("--a0_dir", type=Path, default=DEFAULT_A0_DIR)
    parser.add_argument("--d1_dir", type=Path, default=DEFAULT_D1_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variant", default="qmr", choices=["qmr"])
    parser.add_argument("--max_length_quality", type=int, default=2048)
    parser.add_argument("--max_length_boundary", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=4)
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
    parser.add_argument("--save_best_by", choices=["dev_mae", "dev_qwk"], default="dev_mae")
    parser.add_argument("--max_train_steps", type=int, default=0)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--log_steps", type=int, default=5)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    return parser


def train_one(args: Namespace) -> dict[str, Any]:
    cfg = SCOUT_CONFIGS[args.config_name]
    beta = float(cfg["beta"])
    neg_ratio = int(cfg["neg_ratio"])
    positive_mode = str(cfg["positive_mode"])
    run_dir = Path(args.output_dir) / "runs" / args.config_name / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(args.seed))
    tokenizer = make_tokenizer(str(args.model_name_or_path), args.local_files_only, args.trust_remote_code)
    base = BoundaryLinkingOrdinalModel.from_model_name(
        str(args.model_name_or_path),
        variant="qmr",
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=bool(args.local_files_only),
    )
    model = EvidenceBoundaryLinkingModel(base)
    init_status = load_init_checkpoint(model, Path(args.init_checkpoint), bool(args.allow_missing_init_checkpoint))
    if bool(args.gradient_checkpointing):
        if hasattr(model.base.encoder.config, "use_cache"):
            model.base.encoder.config.use_cache = False
        if hasattr(model.base.encoder, "gradient_checkpointing_enable"):
            model.base.encoder.gradient_checkpointing_enable()

    train_rows = read_jsonl(args.train_path)
    if int(args.max_train_samples) > 0:
        train_rows = train_rows[: int(args.max_train_samples)]
    dev_limit = int(args.max_eval_samples) or None
    train_samples = load_samples(args.train_path, variant="qmr", boundary_fields=None, limit=int(args.max_train_samples) or None)
    dev_samples = load_samples(args.dev_path, variant="qmr", boundary_fields=None, limit=dev_limit)
    evidence_labels, evidence_info = build_evidence_labels(train_rows, args.a0_dir, positive_mode, neg_ratio, int(args.seed))
    add_evidence_to_samples(train_samples, evidence_labels)
    add_evidence_to_samples(dev_samples, {})

    collator = EvidenceCollator(tokenizer, args.max_length_quality, args.max_length_boundary)
    train_loader = DataLoader(
        BoundaryLinkingDataset(train_samples),
        batch_size=int(args.batch_size),
        shuffle=True,
        collate_fn=collator,
    )
    dev_loader = DataLoader(BoundaryLinkingDataset(dev_samples), batch_size=int(args.eval_batch_size), shuffle=False, collate_fn=collator)
    signal_samples = [sample for sample in train_samples if float(sample.get("evidence_mask", 0.0)) > 0]
    signal_loader = DataLoader(BoundaryLinkingDataset(signal_samples), batch_size=int(args.eval_batch_size), shuffle=False, collate_fn=collator)

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
        running = {"loss": 0.0, "ord": 0.0, "hfe": 0.0}
        batch_count = 0
        start = time.time()
        total_batches = len(train_loader)
        print(
            f"[exp17-a1] {args.config_name} epoch {epoch}/{epochs} start "
            f"batches={total_batches} batch_size={args.batch_size} grad_accum={args.grad_accum_steps} "
            f"beta={beta} neg_ratio=1:{neg_ratio}",
            flush=True,
        )
        for batch_idx, batch in enumerate(train_loader, start=1):
            batch = move_evidence_batch(batch, device)
            with autocast_context(args, device):
                outputs = model(
                    quality_input_ids=batch["quality_input_ids"],
                    quality_attention_mask=batch["quality_attention_mask"],
                    boundary_input_ids=batch["boundary_input_ids"],
                    boundary_attention_mask=batch["boundary_attention_mask"],
                )
                loss_ord = ordinal_bce_loss(outputs["logits"], batch["targets"], batch["labels"], class_weights=None)
                loss_hfe = evidence_loss(outputs, batch)
                loss = loss_ord + beta * loss_hfe
            (loss / int(args.grad_accum_steps)).backward()
            running["loss"] += float(loss.detach().cpu())
            running["ord"] += float(loss_ord.detach().cpu())
            running["hfe"] += float(loss_hfe.detach().cpu())
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
                    f"[exp17-a1] {args.config_name} epoch {epoch}/{epochs} batch {batch_idx}/{total_batches} "
                    f"loss={float(loss.detach().cpu()):.4f} ord={float(loss_ord.detach().cpu()):.4f} "
                    f"hfe={float(loss_hfe.detach().cpu()):.4f} global_step={global_step} "
                    f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
                    flush=True,
                )
        if batch_count % int(args.grad_accum_steps) != 0 and (max_steps is None or global_step < max_steps):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
        dev_metrics, dev_predictions = evaluate_model(model, dev_loader, device, split="dev", args=args)
        dev_metrics = {
            **dev_metrics,
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": running["loss"] / max(1, batch_count),
            "train_ord_loss": running["ord"] / max(1, batch_count),
            "train_hfe_loss": running["hfe"] / max(1, batch_count),
        }
        print(
            f"[exp17-a1] {args.config_name} epoch {epoch}/{epochs} dev "
            f"MAE={float(dev_metrics['MAE']):.4f} QWK={float(dev_metrics['QWK']):.4f} "
            f"low_to_high={dev_metrics['low_to_high_count']}/{dev_metrics['true_low_n']}",
            flush=True,
        )
        history.append(dev_metrics)
        better, score = score_is_better(dev_metrics, best_score, args.save_best_by)
        if better:
            best_score = score
            best_metrics = dev_metrics
            best_state = copy.deepcopy(model.state_dict())
            ckpt = run_dir / "checkpoint_best"
            ckpt.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, ckpt / "state_dict.pt")
            write_json(ckpt / "best_metrics.json", dev_metrics)
        if max_steps is not None and global_step >= max_steps:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    final_dev_metrics, final_dev_predictions = evaluate_model(model, dev_loader, device, split="dev", args=args)
    _, signal_predictions = evaluate_model(model, signal_loader, device, split="train_signal", args=args)
    d1_sets = load_dev_eval_sets(args.d1_dir)
    dev_row = dev_metric_row(args.config_name, int(args.seed), beta, neg_ratio, final_dev_predictions)
    evidence_row = evidence_eval_row(args.config_name, int(args.seed), final_dev_predictions, signal_predictions, d1_sets, evidence_labels)

    write_json(run_dir / "metrics_dev.json", final_dev_metrics)
    write_json(run_dir / "evidence_eval.json", evidence_row)
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
            "scout_config": cfg,
            "evidence_info": evidence_info,
            "best_metrics": best_metrics,
            "test_read": False,
            "dev_annotation_used_as_train_label": False,
            "human_rationale_used_as_model_input": False,
        },
    )
    write_csv(run_dir / "exp17_a1_dev_metrics.csv", [dev_row], fieldnames=DEV_METRIC_FIELDS)
    write_csv(run_dir / "exp17_a1_evidence_eval.csv", [evidence_row], fieldnames=EVIDENCE_FIELDS)
    write_csv(run_dir / "exp17_a1_case_scores.csv", case_score_rows(final_dev_predictions), fieldnames=CASE_SCORE_FIELDS)
    report_lines = [
        "# Exp17-A1 Evidence Head Run Report",
        "",
        f"- config_name: `{args.config_name}`",
        f"- seed: {args.seed}",
        f"- beta: {beta}",
        f"- neg_ratio: 1:{neg_ratio}",
        f"- positive_mode: `{positive_mode}`",
        f"- init_checkpoint: `{relpath(args.init_checkpoint)}`",
        f"- init_status: `{init_status}`",
        f"- positive_count: {evidence_info['positive_count']}",
        f"- negative_count: {evidence_info['negative_count']}",
        "",
        "## Dev Metrics",
        "",
        f"- MAE: {dev_row['MAE']}",
        f"- QWK: {dev_row['QWK']}",
        f"- low_to_high: {dev_row['low_to_high_count']} ({dev_row['low_to_high_rate']})",
        f"- label2_recall: {dev_row['label2_recall']}",
        f"- monotonic_violation_rate: {dev_row['monotonic_violation_rate']}",
        "",
        "## Evidence Metrics",
        "",
        f"- h_auc_d1_hidden_vs_controls: {evidence_row['h_auc_d1_hidden_vs_controls']}",
        f"- mean_h_d1_hidden: {evidence_row['mean_h_d1_hidden']}",
        f"- mean_h_d1_controls: {evidence_row['mean_h_d1_controls']}",
        f"- evidence_delta_hidden_minus_control: {evidence_row['evidence_delta_hidden_minus_control']}",
        "",
        "## Guardrails",
        "",
        "- No test split is read.",
        "- Dev D1 annotation is used only for dev evidence evaluation, never as train labels.",
        "- Human rationale text is not used as model input.",
        "- The evidence head does not modify the ordinal scoring path.",
    ]
    (run_dir / "exp17_a1_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return {
        "status": "COMPLETED",
        "config_name": args.config_name,
        "run_dir": relpath(run_dir),
        "dev_MAE": dev_row["MAE"],
        "dev_QWK": dev_row["QWK"],
        "h_auc_d1_hidden_vs_controls": evidence_row["h_auc_d1_hidden_vs_controls"],
        "test_read": False,
    }


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = train_one(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
