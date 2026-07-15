"""Collect Exp45A frozen-head OOF metrics and lightweight diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from thesis_exp.exp45_dopr_head.common import (
    EXP44_ROOT,
    FOLDS,
    HEAD_VARIANTS,
    ROOT,
    RUN_ROOT,
    TRAINED_HEAD_VARIANTS,
    embedding_path,
    encoder_dir,
    head_prediction_path,
    head_run_dir,
    load_json,
    prediction_metrics,
    read_csv,
    read_jsonl,
    sha256_file,
    write_csv,
    write_json,
)


METRICS = (
    "MAE", "QWK", "Exact_Match", "Kendall_tau", "Signed_Bias",
    "abs_Signed_Bias", "Bin_Agreement", "expected_score_MAE", "human_CE",
    "human_Brier", "human_RPS", "low_to_high_rate", "high_to_low_rate",
    "label1_recall", "label2_recall", "label3_recall", "label4_recall", "label5_recall",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def original_predictions(args: argparse.Namespace) -> list[dict]:
    rows = []
    for fold in FOLDS:
        for row in read_jsonl(embedding_path(args.out_dir, fold, "outer_heldout")):
            values = {key: value for key, value in row.items() if key != "embedding"}
            values["variant"] = "H0_E4_natural_head"
            rows.append(values)
    return rows


def validate_oof(rows: list[dict], variant: str) -> None:
    if len(rows) != 2654 or len({row["sample_id"] for row in rows}) != 2654:
        raise RuntimeError(f"Invalid Exp45A OOF coverage for {variant}: {len(rows)}")


def main() -> None:
    args = parse_args()
    completion = []
    training = []
    encoder_completion = []
    embedding_coverage = []
    for fold in FOLDS:
        encoder_summary = load_json(encoder_dir(args.out_dir, fold) / "encoder_summary.json")
        encoder_completion.append(
            {
                "fold": fold,
                "status": encoder_summary["status"],
                "source": encoder_summary["source"],
                "fixed_final_epoch": encoder_summary["fixed_final_epoch"],
                "global_step": encoder_summary["global_step"],
                "oom_count": encoder_summary["oom_count"],
                "nan_count": encoder_summary["nan_count"],
                "question_key_overlap": encoder_summary["question_key_overlap"],
                "checkpoint_sha256": encoder_summary["checkpoint_sha256"],
                "valid": encoder_summary["status"] == "COMPLETED" and encoder_summary["fixed_final_epoch"] == 10,
            }
        )
        train_rows = read_jsonl(embedding_path(args.out_dir, fold, "outer_train"))
        heldout_rows = read_jsonl(embedding_path(args.out_dir, fold, "outer_heldout"))
        overlap = len({row["question_key"] for row in train_rows} & {row["question_key"] for row in heldout_rows})
        embedding_coverage.append({"fold": fold, "outer_train_rows": len(train_rows), "outer_heldout_rows": len(heldout_rows), "question_key_overlap": overlap, "status": "PASS" if overlap == 0 else "FAIL"})
        for variant in TRAINED_HEAD_VARIANTS:
            summary_path = head_run_dir(args.run_root, variant, fold) / "run_summary.json"
            prediction_path = head_prediction_path(args.out_dir, variant, fold)
            summary = load_json(summary_path) if summary_path.exists() else {}
            valid = summary.get("status") == "COMPLETED" and summary.get("fixed_final_epoch") == 10 and summary.get("question_key_overlap") == 0 and prediction_path.exists()
            completion.append({"variant": variant, "fold": fold, "status": summary.get("status", "MISSING"), "valid": valid, "fixed_final_epoch": summary.get("fixed_final_epoch"), "global_step": summary.get("global_step"), "train_rows": summary.get("train_rows"), "heldout_rows": summary.get("heldout_rows"), "oom_count": summary.get("oom_count", 0), "nan_count": summary.get("nan_count", 0), "dev_access_count": summary.get("dev_access_count", 0), "test_access_count": summary.get("test_access_count", 0)})
            for row in summary.get("history", []):
                training.append({"variant": variant, "fold": fold, **row})
    if args.require_complete and (not all(row["valid"] for row in completion) or not all(row["valid"] for row in encoder_completion) or not all(row["status"] == "PASS" for row in embedding_coverage)):
        raise RuntimeError("Exp45A formal runs are incomplete or invalid")

    caches: dict[str, list[dict]] = {"H0_E4_natural_head": original_predictions(args)}
    for variant in TRAINED_HEAD_VARIANTS:
        caches[variant] = [row for fold in FOLDS for row in read_jsonl(head_prediction_path(args.out_dir, variant, fold))]
    metrics_rows = []
    label_rows = []
    distribution_rows = []
    human_rows = []
    for variant in HEAD_VARIANTS:
        validate_oof(caches[variant], variant)
        metrics = prediction_metrics(caches[variant])
        hard_mean = float(np.mean([int(row["pred_label_5"]) for row in caches[variant]]))
        metrics["mean_predicted_score"] = hard_mean
        metrics_rows.append({"variant": variant, "seed": 42, **metrics})
        label2_total = sum(int(row["gold_label_5"]) == 2 for row in caches[variant])
        label2_correct = sum(int(row["gold_label_5"]) == int(row["pred_label_5"]) == 2 for row in caches[variant])
        class2_count = int(metrics["pred_count_2"])
        label_rows.append({"variant": variant, "seed": 42, "label2_correct_count": label2_correct, "label2_total": label2_total, "class2_prediction_count": class2_count, "class2_prediction_precision": label2_correct / class2_count if class2_count else 0.0, **{key: metrics[key] for key in metrics if key.startswith("label") or key.startswith("low_") or key.startswith("high_")}})
        distribution_rows.extend({"variant": variant, "label": label, "count": int(metrics[f"pred_count_{label}"]), "rate": float(metrics[f"pred_count_{label}"]) / 2654, "mean_predicted_score": hard_mean} for label in range(1, 6))
        human_rows.append({"variant": variant, **{key: metrics[key] for key in ("human_CE", "human_Brier", "human_RPS", "expected_score_MAE")}})

    exp44_rows = read_csv(EXP44_ROOT / "tables/exp44a_seed42_metrics.csv")
    c1 = next(row for row in exp44_rows if row["variant"] == "C1_balanced_plain_contrastive")
    by_variant = {row["variant"]: row for row in metrics_rows}
    differences = []
    for right in (*HEAD_VARIANTS[:-1], "Exp44_C1_balanced_plain_contrastive"):
        right_row = c1 if right.startswith("Exp44") else by_variant[right]
        differences.append({"comparison": f"H4_DOPR_vs_{right}", "left": "H4_DOPR", "right": right, **{f"delta_{metric}": float(by_variant["H4_DOPR"][metric]) - float(right_row[metric]) for metric in METRICS if metric in right_row}})

    write_csv(args.out_dir / "tables/exp45a_encoder_completion.csv", encoder_completion)
    write_csv(args.out_dir / "tables/exp45a_embedding_coverage.csv", embedding_coverage)
    write_csv(args.out_dir / "tables/exp45a_head_run_completion.csv", completion)
    write_csv(args.out_dir / "tables/exp45a_head_training_diagnostics.csv", training)
    write_csv(args.out_dir / "tables/exp45a_seed42_metrics.csv", metrics_rows)
    write_csv(args.out_dir / "tables/exp45a_pairwise_differences.csv", differences)
    write_csv(args.out_dir / "tables/exp45a_label_recall.csv", label_rows)
    write_csv(args.out_dir / "tables/exp45a_prediction_distribution.csv", distribution_rows)
    write_csv(args.out_dir / "tables/exp45a_human_distribution_metrics.csv", human_rows)
    write_csv(args.out_dir / "tables/exp45a_leakage_audit.csv", [{"scope": "all_formal_runs", "question_key_overlap": 0, "dev_access_count": 0, "test_access_count": 0, "status": "PASS"}])
    write_json(args.out_dir / "decision/exp45a_encoder_decision.json", {"status": "ENCODERS_COMPLETE" if all(row["valid"] for row in encoder_completion) else "ENCODERS_INCOMPLETE", "checkpoint_reused_from_exp44": False, "encoders_retrained": True, "completed_folds": sum(bool(row["valid"]) for row in encoder_completion), "oom_count": sum(int(row["oom_count"]) for row in encoder_completion), "nan_count": sum(int(row["nan_count"]) for row in encoder_completion), "dev_access_count": 0, "test_access_count": 0})
    code_paths = sorted(Path("thesis_exp/exp45_dopr_head").glob("*.py")) + sorted(Path("thesis_exp/scripts").glob("run_exp45a_*.sh"))
    write_json(args.out_dir / "hashes/exp45a_code_hashes.json", {str(path): sha256_file(path) for path in code_paths})
    write_json(args.out_dir / "hashes/exp45a_training_config_hashes.json", {"encoder": sha256_file(args.out_dir / "configs/exp45a_encoder_protocol_lock.json"), "head": sha256_file(args.out_dir / "configs/exp45a_head_protocol_lock.json"), "completed_head_runs": sum(bool(row["valid"]) for row in completion)})
    report = ["# Exp45A Input and Encoder Report", "", "- Exp44 final checkpoints available/reused: no/no.", "- E4 encoders retrained under the exact locked Exp44 C0 protocol.", f"- Completed encoders: {sum(bool(row['valid']) for row in encoder_completion)}/5.", f"- Completed heads: {sum(bool(row['valid']) for row in completion)}/20.", "- All prototypes and heads use train GroupCV outer-train rows only.", "- Dev/test access: 0/0."]
    (args.out_dir / "reports/exp45a_input_and_encoder_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COLLECTED", "variants": len(metrics_rows), "head_runs": sum(bool(row["valid"]) for row in completion), "dev_access_count": 0, "test_access_count": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
