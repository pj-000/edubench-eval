"""Collect Exp36A seed42 metrics, bootstrap comparisons, report, and GO/STOP decision."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp36_safer_score.bootstrap_exp36a_dev import bootstrap
from thesis_exp.exp36_safer_score.common import ROOT, metrics, read_jsonl, write_csv, write_json


FORMAL = ("v0_original_hard", "v0h_human_soft", "v1_qwen_hard", "v2_qwen_range_soft", "v3_naive_human_qwen", "v4_human_soft_logit_adjustment", "v5_safer_score", "v7_shuffled_teacher_control")


def load_summary(run_root: Path, variant: str) -> dict[str, Any]:
    path = run_root / variant / "seed_42/run_summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "COMPLETED":
        raise ValueError(f"Incomplete Exp36A run: {variant}")
    return value


def strong_baseline(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in rows if row["variant"] in {"v0_original_hard", "v0h_human_soft", "v4_human_soft_logit_adjustment"}]
    minimum = min(float(row["MAE_argmax"]) for row in candidates)
    eligible = [row for row in candidates if float(row["MAE_argmax"]) <= minimum + 0.005 + 1e-12]
    return min(eligible, key=lambda row: (-float(row["Exact_Match"]), float(row["MAE_argmax"]), row["variant"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--run-root", type=Path, default=Path("thesis_exp/runs/exp36_safer_score"))
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = parser.parse_args()
    selected = []
    epoch_rows = []
    predictions = {}
    failure_rows = []
    curriculum_rows = []
    label_rows = []
    for variant in FORMAL:
        summary = load_summary(args.run_root, variant)
        row = {"variant": variant, "seed": 42, "selected_epoch": summary["selected_epoch"], **summary["selected_metrics"]}
        selected.append(row)
        history = json.loads((args.run_root / variant / "seed_42/epoch_metrics.json").read_text(encoding="utf-8"))
        epoch_rows.extend(history)
        failure_rows.append({"variant": variant, "seed": 42, **summary["failure_head_train_metrics"]})
        curriculum_rows.extend({
            "variant": variant, "epoch": item["epoch"], "curriculum_rho": item["curriculum_rho"],
            "mean_teacher_lambda_seen": item["mean_teacher_lambda_seen"],
            "train_score_loss": item["train_score_loss"], "train_failure_loss": item["train_failure_loss"],
            "failure_mask_exposures": item["failure_mask_exposures"],
        } for item in history)
        predictions[variant] = read_jsonl(args.out_dir / "private/dev_predictions" / variant / "seed_42.jsonl")
        for label in range(1, 6):
            subset = [item for item in predictions[variant] if int(item["gold_label_5"]) == label]
            label_rows.append({"variant": variant, "seed": 42, "label": label, "n": len(subset),
                               "recall": sum(int(item["pred_label_5"]) == label for item in subset) / len(subset) if subset else float("nan")})
    base = strong_baseline(selected)
    v5 = next(row for row in selected if row["variant"] == "v5_safer_score")
    v3 = next(row for row in selected if row["variant"] == "v3_naive_human_qwen")
    v7 = next(row for row in selected if row["variant"] == "v7_shuffled_teacher_control")
    checks = {
        "mae_guard": float(v5["MAE_argmax"]) <= float(base["MAE_argmax"]) + 0.01,
        "exact_guard": float(v5["Exact_Match"]) >= float(base["Exact_Match"]) - 0.01,
        "qwk_guard": float(v5["QWK"]) >= float(base["QWK"]) - 0.02,
        "label5_guard": float(v5["label5_recall"]) >= float(base["label5_recall"]) - 0.03,
        "high_to_low_guard": float(v5["high_to_low_rate"]) <= float(base["high_to_low_rate"]) + 0.02,
        "low_tail_improvement": (
            float(v5["low_to_high_rate"]) <= float(base["low_to_high_rate"]) - 0.05
            or float(v5["label2_recall"]) >= float(base["label2_recall"]) + 0.05
        ),
        "beats_naive_or_shuffle": (
            float(v5["MAE_argmax"]) < float(v3["MAE_argmax"])
            or float(v5["low_to_high_rate"]) < float(v3["low_to_high_rate"])
            or float(v5["MAE_argmax"]) < float(v7["MAE_argmax"])
            or float(v5["low_to_high_rate"]) < float(v7["low_to_high_rate"])
        ),
        "no_nan_or_oom": all(math.isfinite(float(v5[key])) for key in ("MAE_argmax", "Exact_Match", "QWK")),
        "no_test_access": True,
    }
    go = all(checks.values())
    pairwise = []
    bootstrap_rows = []
    for right in (base["variant"], "v3_naive_human_qwen", "v7_shuffled_teacher_control"):
        right_metrics = next(row for row in selected if row["variant"] == right)
        pairwise.append({
            "left_variant": "v5_safer_score", "right_variant": right,
            **{f"delta_{key}": float(v5[key]) - float(right_metrics[key]) for key in
               ("MAE_argmax", "Exact_Match", "QWK", "low_to_high_rate", "label2_recall", "label5_recall", "high_to_low_rate")},
        })
        rows = bootstrap(predictions["v5_safer_score"], predictions[right], args.bootstrap_resamples, 42)
        for row in rows:
            row.update({"left_variant": "v5_safer_score", "right_variant": right})
        bootstrap_rows.extend(rows)
    write_csv(args.out_dir / "tables/exp36a_seed42_selected_metrics.csv", selected)
    write_csv(args.out_dir / "tables/exp36a_seed42_epoch_metrics.csv", epoch_rows)
    write_csv(args.out_dir / "tables/exp36a_seed42_pairwise_differences.csv", pairwise)
    write_csv(args.out_dir / "tables/exp36a_seed42_question_key_bootstrap_ci.csv", bootstrap_rows)
    write_csv(args.out_dir / "tables/exp36a_label_recall.csv", label_rows)
    write_csv(args.out_dir / "tables/exp36a_failure_head_metrics.csv", failure_rows)
    write_csv(args.out_dir / "tables/exp36a_train_curriculum_diagnostics.csv", curriculum_rows)
    decision = {
        "status": "GO_MULTISEED" if go else "STOP_SAFER_SCORE",
        "strong_baseline": base["variant"], "checks": checks,
        "recommend_run_seeds_43_44": go, "stop_safer_score": not go,
        "api_calls": 0, "test_access_count": 0,
    }
    write_json(args.out_dir / "decision/exp36a_seed42_decision.json", decision)
    lines = [
        "# Exp36A SAFER-Score Seed42 Report", "",
        "## Locked scope", "", "Train-only OOF supervision and one dev-only seed42 scout. No API and no test access.", "",
        "## Selected metrics", "",
        "| Variant | Epoch | MAE | Exact | QWK | L2H | Label2 recall | Label5 recall | H2L |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(f"| {row['variant']} | {row['selected_epoch']} | {row['MAE_argmax']:.4f} | {row['Exact_Match']:.4f} | {row['QWK']:.4f} | {row['low_to_high_rate']:.4f} | {row['label2_recall']:.4f} | {row['label5_recall']:.4f} | {row['high_to_low_rate']:.4f} |")
    lines += ["", "## Gate", "", f"Strong baseline: `{base['variant']}`.", ""]
    lines.extend(f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in checks.items())
    lines += ["", f"Decision: **{decision['status']}**.", "",
              f"recommend_run_seeds_43_44 = `{str(go).lower()}`", f"stop_safer_score = `{str(not go).lower()}`", "",
              "No API, no test access, and no heavy/private artifacts are intended for commit."]
    report = args.out_dir / "reports/exp36a_safer_score_seed42_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(decision)


if __name__ == "__main__":
    main()
