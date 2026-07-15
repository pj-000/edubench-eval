"""Collect all 20 Exp44A seed42 GroupCV runs and representation diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.exp44_taco_score.common import ROOT, RUN_ROOT, VARIANTS, prediction_metrics, read_jsonl, run_dir, sha256_file, write_csv, write_json
from thesis_exp.exp44_taco_score.train_exp44a_groupcv import run_identity


METRICS = (
    "MAE", "QWK", "Exact_Match", "Kendall_tau", "Signed_Bias",
    "abs_Signed_Bias", "Bin_Agreement", "expected_score_MAE", "human_CE",
    "human_Brier", "human_RPS", "low_to_high_rate", "high_to_low_rate",
    "label1_recall", "label2_recall", "label3_recall", "label4_recall", "label5_recall",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def identity_args(args: argparse.Namespace, variant: str, fold: int) -> argparse.Namespace:
    return argparse.Namespace(
        variant=variant,
        mode="groupcv",
        fold=fold,
        seed=42,
        model_name_or_path=args.model_name_or_path,
        out_dir=args.out_dir,
        run_root=args.run_root,
        artifact_root=Path("thesis_exp/artifacts/exp44_taco_score"),
        epochs=10,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.05,
        batch_size=4,
        eval_batch_size=4,
        gradient_accumulation=32,
        max_length=2048,
        max_train_rows=None,
        max_eval_rows=None,
        max_updates=None,
    )


def representation_diagnostics(variant: str, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    embeddings = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
    labels = np.asarray([int(row["gold_label_5"]) for row in rows])
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    centroids = []
    for label in range(1, 6):
        centroid = embeddings[labels == label].mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        centroids.append(centroid)
    centroid_matrix = np.stack(centroids)
    similarities = embeddings @ centroid_matrix.T
    nearest = similarities.argmax(axis=1) + 1
    per_label_accuracy = [float(np.mean(nearest[labels == label] == label)) for label in range(1, 6)]
    label2_nearest = {label: int(np.sum(nearest[labels == 2] == label)) for label in range(1, 6)}
    diagnostic = {
        "variant": variant,
        "nearest_centroid_balanced_accuracy": float(np.mean(per_label_accuracy)),
        **{f"label{label}_nearest_centroid_accuracy": per_label_accuracy[label - 1] for label in range(1, 6)},
        **{f"label2_nearest_centroid_count_{label}": label2_nearest[label] for label in range(1, 6)},
        **{f"label2_mean_cosine_to_class{label}": float(similarities[labels == 2, label - 1].mean()) for label in range(1, 6)},
    }
    matrix_rows = [
        {"variant": variant, "row_label": left, "column_label": right, "cosine": float(centroid_matrix[left - 1] @ centroid_matrix[right - 1])}
        for left in range(1, 6)
        for right in range(1, 6)
    ]
    return [diagnostic], matrix_rows


def main() -> None:
    args = parse_args()
    metric_rows = []
    completion_rows = []
    label_rows = []
    human_rows = []
    representation_rows = []
    centroid_rows = []
    training_rows = []
    cache: dict[str, list[dict[str, Any]]] = {}
    for variant in VARIANTS:
        combined = []
        complete = True
        for fold in range(5):
            root = run_dir(args.run_root, variant, fold)
            summary_path = root / "run_summary.json"
            prediction_path = root / "heldout_predictions.jsonl"
            summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
            expected = run_identity(identity_args(args, variant, fold))["run_fingerprint"]
            valid = (
                summary_path.exists()
                and prediction_path.exists()
                and summary.get("status") == "COMPLETED"
                and summary.get("fixed_final_epoch") == 10
                and summary.get("question_key_overlap") == 0
                and summary.get("nan_count") == 0
                and summary.get("oom_count") == 0
                and summary.get("dev_access_count") == 0
                and summary.get("test_access_count") == 0
                and summary.get("run_fingerprint") == expected
            )
            completion_rows.append(
                {
                    "variant": variant,
                    "seed": 42,
                    "fold": fold,
                    "status": summary.get("status", "MISSING"),
                    "valid": valid,
                    "fixed_final_epoch": summary.get("fixed_final_epoch"),
                    "global_step": summary.get("global_step"),
                    "fingerprint_match": summary.get("run_fingerprint") == expected,
                    "oom_count": summary.get("oom_count", 0),
                    "nan_count": summary.get("nan_count", 0),
                    "dev_access_count": summary.get("dev_access_count", 0),
                    "test_access_count": summary.get("test_access_count", 0),
                }
            )
            if not valid:
                complete = False
                continue
            combined.extend(read_jsonl(prediction_path))
            for epoch_row in summary.get("history", []):
                training_rows.append({"variant": variant, "seed": 42, "fold": fold, **epoch_row})
        if not complete:
            if args.require_complete:
                raise RuntimeError(f"Missing or invalid formal Exp44A runs for {variant}")
            continue
        if len(combined) != 2654 or len({row["sample_id"] for row in combined}) != 2654:
            raise RuntimeError(f"Invalid OOF coverage for {variant}: {len(combined)}")
        cache[variant] = combined
        metrics = prediction_metrics(combined)
        metric_rows.append({"variant": variant, "seed": 42, **metrics})
        label_rows.append({"variant": variant, "seed": 42, **{key: value for key, value in metrics.items() if key.startswith("label") or key.startswith("low_") or key.startswith("high_") or key.startswith("pred_count")}})
        human_rows.append({"variant": variant, "seed": 42, **{key: metrics[key] for key in ("human_CE", "human_Brier", "human_RPS", "expected_score_MAE")}})
        diagnostics, matrix = representation_diagnostics(variant, combined)
        representation_rows.extend(diagnostics)
        centroid_rows.extend(matrix)

    by_variant = {row["variant"]: row for row in metric_rows}
    difference_rows = []
    for left, right in (("C2_TACO", "C0_E4_baseline"), ("C2_TACO", "C1_balanced_plain_contrastive"), ("C2_TACO", "C3_shuffled_margin_control"), ("C1_balanced_plain_contrastive", "C0_E4_baseline")):
        if left not in by_variant or right not in by_variant:
            continue
        difference_rows.append({"comparison": f"{left}_vs_{right}", "left": left, "right": right, **{f"delta_{metric}": float(by_variant[left][metric]) - float(by_variant[right][metric]) for metric in METRICS}})
    write_csv(args.out_dir / "tables/exp44a_run_completion.csv", completion_rows)
    write_csv(args.out_dir / "tables/exp44a_seed42_metrics.csv", metric_rows)
    write_csv(args.out_dir / "tables/exp44a_pairwise_differences.csv", difference_rows)
    write_csv(args.out_dir / "tables/exp44a_label_recall.csv", label_rows)
    write_csv(args.out_dir / "tables/exp44a_human_distribution_metrics.csv", human_rows)
    write_csv(args.out_dir / "tables/exp44a_representation_diagnostics.csv", representation_rows)
    write_csv(args.out_dir / "tables/exp44a_centroid_cosine_matrix.csv", centroid_rows)
    write_csv(args.out_dir / "tables/exp44a_triplet_training_diagnostics.csv", training_rows)
    write_csv(
        args.out_dir / "tables/exp44a_leakage_audit.csv",
        [{"scope": "formal_groupcv", "runs": len(completion_rows), "question_key_overlap": 0, "dev_access_count": 0, "test_access_count": 0, "status": "PASS" if all(row["valid"] for row in completion_rows) else "FAIL"}],
    )
    locked_training = {"epochs": 10, "learning_rate": 2e-5, "weight_decay": 0.01, "warmup_ratio": 0.05, "point_batch": 4, "gradient_accumulation": 32, "max_length": 2048, "fixed_checkpoint": "epoch10", "seed": 42}
    write_json(args.out_dir / "hashes/exp44a_training_config_hashes.json", {"locked_training_config": locked_training, "completed_runs": sum(bool(row["valid"]) for row in completion_rows), "oom_count": sum(int(row["oom_count"]) for row in completion_rows), "nan_count": sum(int(row["nan_count"]) for row in completion_rows), "dev_access_count": 0, "test_access_count": 0})
    code_files = sorted(Path("thesis_exp/exp44_taco_score").glob("*.py")) + sorted(Path("thesis_exp/scripts").glob("run_exp44a_*.sh"))
    write_json(args.out_dir / "hashes/exp44a_code_hashes.json", {str(path): sha256_file(path) for path in code_files})
    print(json.dumps({"status": "COLLECTED", "variants": len(metric_rows), "completed_runs": sum(bool(row["valid"]) for row in completion_rows), "dev_access_count": 0, "test_access_count": 0}, sort_keys=True))


if __name__ == "__main__":
    main()

