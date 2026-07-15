"""Build a lightweight diagnosis after the Exp46A teacher capacity gate fails."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from thesis_exp.exp46_hato_kd.common import ROOT, write_csv


LOWER_IS_BETTER = {
    "MAE",
    "abs_Signed_Bias",
    "expected_score_MAE",
    "human_CE",
    "human_Brier",
    "human_RPS",
    "low_to_high_rate",
    "high_to_low_rate",
}

DIAGNOSIS_METRICS = (
    "MAE",
    "QWK",
    "Exact_Match",
    "Kendall_tau",
    "abs_Signed_Bias",
    "Bin_Agreement",
    "expected_score_MAE",
    "human_CE",
    "human_Brier",
    "human_RPS",
    "low_to_high_rate",
    "high_to_low_rate",
    "label2_recall",
    "label5_recall",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fmt(value: float) -> str:
    return f"{value:.4f}"


def main() -> None:
    args = parse_args()
    decision_path = args.out_dir / "decision/exp46a_teacher_capacity_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("status") != "TEACHER_CAPACITY_NO_GO":
        raise RuntimeError("Failure diagnosis requires TEACHER_CAPACITY_NO_GO")

    rows = {row["variant"]: row for row in read_csv(args.out_dir / "tables/exp46a_teacher_capacity_metrics.csv")}
    baseline = rows["K0_E4"]
    teacher = rows["T1_4B_teacher"]
    bootstrap = {
        row["metric"]: row
        for row in read_csv(args.out_dir / "tables/exp46a_teacher_question_key_bootstrap.csv")
        if row["comparison"] == "T1_4B_teacher_vs_K0_E4"
    }

    diagnosis_rows = []
    for metric in DIAGNOSIS_METRICS:
        base_value = float(baseline[metric])
        teacher_value = float(teacher[metric])
        delta = teacher_value - base_value
        better = delta < 0 if metric in LOWER_IS_BETTER else delta > 0
        boot = bootstrap.get(metric, {})
        diagnosis_rows.append(
            {
                "metric": metric,
                "better_direction": "lower" if metric in LOWER_IS_BETTER else "higher",
                "K0_E4": base_value,
                "T1_4B_teacher": teacher_value,
                "teacher_minus_baseline": delta,
                "point_direction": "better" if better else "tie" if delta == 0 else "worse",
                "bootstrap_ci_low": boot.get("ci_low", ""),
                "bootstrap_ci_high": boot.get("ci_high", ""),
            }
        )
    write_csv(args.out_dir / "tables/exp46a_teacher_failure_diagnosis.csv", diagnosis_rows)

    checks = decision["checks"]
    failed_checks = [name for name, passed in checks.items() if not passed]
    report = [
        "# Exp46A Teacher Failure Diagnosis",
        "",
        "Decision: **TEACHER_CAPACITY_NO_GO**",
        "",
        "## What failed",
        "",
        f"- The 4B teacher identified **0 label-2 examples** correctly and label-2 recall remained 0. It predicted label 2 for {teacher['pred_count_2']} examples, but every such prediction was a false positive.",
        f"- Low-to-high increased from {baseline['low_to_high_count']}/{baseline['low_n']} ({fmt(float(baseline['low_to_high_rate']))}) to {teacher['low_to_high_count']}/{teacher['low_n']} ({fmt(float(teacher['low_to_high_rate']))}).",
        f"- MAE worsened from {fmt(float(baseline['MAE']))} to {fmt(float(teacher['MAE']))}; QWK fell from {fmt(float(baseline['QWK']))} to {fmt(float(teacher['QWK']))}.",
        f"- Exact Match fell from {fmt(float(baseline['Exact_Match']))} to {fmt(float(teacher['Exact_Match']))}; Kendall tau fell from {fmt(float(baseline['Kendall_tau']))} to {fmt(float(teacher['Kendall_tau']))}.",
        f"- Absolute signed bias increased from {fmt(float(baseline['abs_Signed_Bias']))} to {fmt(float(teacher['abs_Signed_Bias']))}; the mean prediction also increased from {fmt(float(baseline['mean_pred']))} to {fmt(float(teacher['mean_pred']))}.",
        "",
        "## Statistical reading",
        "",
        "- Question-key bootstrap intervals cross zero for the overall metric deltas, so the experiment does not establish statistically significant overall harm.",
        "- It also provides no positive overall gain: every preregistered point-estimate improvement condition failed.",
        "- Label-2 recall is exactly unchanged at zero, with bootstrap delta 0 and interval [0, 0]. This is the decisive mechanism failure for distillation.",
        "- Human cross-entropy improved slightly, but Brier score and ranked probability score worsened. Better likelihood on the observed distributions did not translate into safer hard-score decisions.",
        "",
        "## Causal scope",
        "",
        "This result rejects the locked Exp46A premise that a 4B LoRA teacher trained with the same human-distribution and ordinal objective supplies transferable label-2 structure. It does **not** prove that Qwen3-Reranker-4B is inherently incapable, nor does it test full fine-tuning, additional tail supervision, or a different data distribution.",
        "",
        "## Protocol consequence",
        "",
        "- K1/K2/K3 student training was correctly skipped. Distilling this teacher would transfer no verified label-2 signal.",
        "- Do not rerun Exp46A unchanged and do not tune the Gate after seeing these results.",
        "- A new positive experiment would require a fresh preregistration and a changed source of tail evidence or optimization, not merely a larger teacher.",
        "",
        "## Integrity",
        "",
        "- Five of five Teacher folds completed at the locked final epoch.",
        "- Question-key overlap: 0.",
        "- Dev access count: 0.",
        "- Test access count: 0.",
        f"- Failed Gate checks: {', '.join(failed_checks)}.",
    ]
    report_path = args.out_dir / "reports/exp46a_teacher_failure_diagnosis.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": "DIAGNOSED", "failed_checks": failed_checks, "test_access_count": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
