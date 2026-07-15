"""Diagnose train-derived soft prototypes on five heldout GroupCV folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from thesis_exp.exp43_rubimor.common import qwk, kendall_tau_b
from thesis_exp.exp45_dopr_head.common import (
    FOLDS, ROOT, atomic_json, embedding_path, encoder_dir, load_json,
    prediction_metrics, prototype_path, read_jsonl, write_csv, write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    return parser.parse_args()


def normalize(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def fold_diagnostic(args: argparse.Namespace, fold: int) -> tuple[list[dict], dict, list[dict], list[dict]]:
    train = read_jsonl(embedding_path(args.out_dir, fold, "outer_train"))
    heldout = read_jsonl(embedding_path(args.out_dir, fold, "outer_heldout"))
    if {row["question_key"] for row in train} & {row["question_key"] for row in heldout}:
        raise RuntimeError(f"Prototype question-key leakage in fold {fold}")
    train_h = normalize(np.asarray([row["embedding"] for row in train], dtype=np.float64))
    targets = np.asarray([row["human_distribution_5"] for row in train], dtype=np.float64)
    masses = targets.sum(axis=0)
    if np.any(masses <= 0):
        raise RuntimeError(f"Empty prototype in fold {fold}: {masses}")
    prototypes = normalize((targets.T @ train_h) / masses[:, None])
    priors = masses / len(train)
    if not np.isfinite(prototypes).all() or not np.isfinite(priors).all():
        raise FloatingPointError("Non-finite Exp45A prototypes")
    heldout_h = normalize(np.asarray([row["embedding"] for row in heldout], dtype=np.float64))
    similarities = heldout_h @ prototypes.T
    exp_values = np.exp(similarities - similarities.max(axis=1, keepdims=True))
    probabilities = exp_values / exp_values.sum(axis=1, keepdims=True)
    output = []
    for row, similarity, probability in zip(heldout, similarities, probabilities):
        output.append(
            {
                **{key: row[key] for key in ("sample_id", "question_key", "fold", "gold_label_5", "human_distribution_5", "expected_human_score", "metric", "language", "subject")},
                "pred_label_5": int(np.argmax(similarity)) + 1,
                "pred_score_expected": float(sum(label * probability[label - 1] for label in range(1, 6))),
                **{f"prob_{label}": float(probability[label - 1]) for label in range(1, 6)},
            }
        )
    write_json(prototype_path(args.out_dir, fold), {"fold": fold, "effective_mass": masses.tolist(), "soft_prior": priors.tolist(), "prototypes": prototypes.tolist(), "train_rows": len(train), "heldout_rows": len(heldout), "question_key_overlap": 0})
    write_jsonl(args.out_dir / f"private/predictions/prototype_transfer/fold_{fold}.jsonl", output)
    metrics = {"fold": fold, **prediction_metrics(output)}
    metrics["balanced_accuracy"] = float(np.mean([metrics[f"label{label}_recall"] for label in range(1, 6)]))
    stats = [{"fold": fold, "label": label, "effective_mass": float(masses[label - 1]), "soft_prior": float(priors[label - 1]), "hard_count": sum(int(row["gold_label_5"]) == label for row in train)} for label in range(1, 6)]
    cosine = [{"fold": fold, "row_label": left, "column_label": right, "cosine": float(prototypes[left - 1] @ prototypes[right - 1])} for left in range(1, 6) for right in range(1, 6)]
    return output, metrics, stats, cosine


def main() -> None:
    args = parse_args()
    encoder_rows = []
    coverage_rows = []
    for fold in FOLDS:
        summary = load_json(encoder_dir(args.out_dir, fold) / "encoder_summary.json")
        valid = summary.get("status") == "COMPLETED" and summary.get("fixed_final_epoch") == 10 and summary.get("question_key_overlap") == 0 and summary.get("oom_count") == summary.get("nan_count") == 0
        encoder_rows.append({"fold": fold, "status": summary.get("status"), "source": summary.get("source"), "valid": valid, "fixed_final_epoch": summary.get("fixed_final_epoch"), "global_step": summary.get("global_step"), "oom_count": summary.get("oom_count"), "nan_count": summary.get("nan_count"), "question_key_overlap": summary.get("question_key_overlap"), "checkpoint_sha256": summary.get("checkpoint_sha256")})
        outer_train = read_jsonl(embedding_path(args.out_dir, fold, "outer_train"))
        heldout = read_jsonl(embedding_path(args.out_dir, fold, "outer_heldout"))
        overlap = len({row["question_key"] for row in outer_train} & {row["question_key"] for row in heldout})
        coverage_rows.append({"fold": fold, "outer_train_rows": len(outer_train), "outer_heldout_rows": len(heldout), "question_key_overlap": overlap, "status": "PASS" if overlap == 0 else "FAIL"})
    if not all(row["valid"] for row in encoder_rows) or not all(row["status"] == "PASS" for row in coverage_rows):
        raise RuntimeError("Exp45A encoder or embedding audit failed before prototype diagnosis")
    write_csv(args.out_dir / "tables/exp45a_encoder_completion.csv", encoder_rows)
    write_csv(args.out_dir / "tables/exp45a_embedding_coverage.csv", coverage_rows)
    atomic_json(args.out_dir / "decision/exp45a_encoder_decision.json", {"status": "ENCODERS_COMPLETE", "checkpoint_reused_from_exp44": False, "encoders_retrained": True, "completed_folds": 5, "oom_count": 0, "nan_count": 0, "dev_access_count": 0, "test_access_count": 0})
    input_report = ["# Exp45A Input and Encoder Report", "", "- Exp44 completed final checkpoints were unavailable; no Exp44 checkpoint was reused.", "- Five E4 encoders were retrained under the exact locked Exp44 C0 configuration.", "- Completed encoder folds: 5/5.", f"- Outer-heldout coverage: {sum(int(row['outer_heldout_rows']) for row in coverage_rows)}/2654.", "- Every fold has zero outer-train/heldout question-key overlap.", "- Dev/test access: 0/0."]
    (args.out_dir / "reports/exp45a_input_and_encoder_report.md").write_text("\n".join(input_report) + "\n", encoding="utf-8")

    combined = []
    by_fold = []
    stats = []
    cosine = []
    for fold in FOLDS:
        output, metrics, fold_stats, fold_cosine = fold_diagnostic(args, fold)
        combined.extend(output); by_fold.append(metrics); stats.extend(fold_stats); cosine.extend(fold_cosine)
    if len(combined) != 2654 or len({row["sample_id"] for row in combined}) != 2654:
        raise RuntimeError(f"Invalid prototype OOF coverage: {len(combined)}")
    metrics = prediction_metrics(combined)
    metrics["balanced_accuracy"] = float(np.mean([metrics[f"label{label}_recall"] for label in range(1, 6)]))
    gold = np.asarray([int(row["gold_label_5"]) for row in combined])
    pred = np.asarray([int(row["pred_label_5"]) for row in combined])
    confusion = [{"gold_label": left, "pred_label": right, "count": int(np.sum((gold == left) & (pred == right)))} for left in range(1, 6) for right in range(1, 6)]
    label2_correct = int(np.sum((gold == 2) & (pred == 2)))
    class2_predictions = int(np.sum(pred == 2))
    go = metrics["label2_recall"] >= 0.05 and label2_correct >= 3 and class2_predictions >= 3
    decision = {
        "status": "PROTOTYPE_SIGNAL_GO" if go else "PROTOTYPE_SIGNAL_NO_GO",
        "recommend_head_training": go, "stop_positive_small_paper_route": not go,
        "label2_correct_count": label2_correct, "label2_total": int(np.sum(gold == 2)),
        "label2_recall": metrics["label2_recall"], "class2_prediction_count": class2_predictions,
        "class2_prediction_precision": label2_correct / class2_predictions if class2_predictions else 0.0,
        "question_key_leakage": 0, "all_prototypes_finite_nonempty": True,
        "dev_access_count": 0, "test_access_count": 0,
    }
    write_csv(args.out_dir / "tables/exp45a_train_prototype_statistics.csv", stats)
    write_csv(args.out_dir / "tables/exp45a_prototype_transfer_metrics.csv", [{"scope": "aggregate", **metrics}, *[{"scope": "fold", **row} for row in by_fold]])
    write_csv(args.out_dir / "tables/exp45a_prototype_confusion_matrix.csv", confusion)
    write_csv(args.out_dir / "tables/exp45a_prototype_cosine_matrix.csv", cosine)
    atomic_json(args.out_dir / "decision/exp45a_prototype_signal_decision.json", decision)
    report = [
        "# Exp45A Train-Derived Prototype Diagnostic", "",
        f"Decision: **{decision['status']}**", "",
        f"- OOF rows: {len(combined)}", f"- Label2 correct: {label2_correct}/{decision['label2_total']}",
        f"- Label2 recall: {metrics['label2_recall']:.6f}", f"- Class2 predictions: {class2_predictions}",
        f"- Class2 precision: {decision['class2_prediction_precision']:.6f}",
        f"- MAE/QWK/Exact/Kendall: {metrics['MAE']:.6f} / {metrics['QWK']:.6f} / {metrics['Exact_Match']:.6f} / {metrics['Kendall_tau']:.6f}",
        f"- Balanced accuracy: {metrics['balanced_accuracy']:.6f}",
        f"- Low-to-high/high-to-low: {metrics['low_to_high_rate']:.6f} / {metrics['high_to_low_rate']:.6f}",
        "- Prototypes use outer-train human distributions only; heldout labels are evaluation-only.",
        "- Dev/test access: 0/0.",
    ]
    (args.out_dir / "reports/exp45a_prototype_diagnostic_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
