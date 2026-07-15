"""Apply the preregistered Exp44A seed42 GO/STOP gate and write the report."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from thesis_exp.exp44_taco_score.common import ROOT, atomic_json


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = {row["variant"]: row for row in read_csv(args.out_dir / "tables/exp44a_seed42_metrics.csv")}
    bootstrap = read_csv(args.out_dir / "tables/exp44a_question_key_bootstrap_ci.csv")
    required = {"C0_E4_baseline", "C1_balanced_plain_contrastive", "C2_TACO", "C3_shuffled_margin_control"}
    if set(metrics) != required:
        raise RuntimeError(f"Exp44A decision requires all variants: {set(metrics)}")
    c0, c1, c2, c3 = (metrics[name] for name in ("C0_E4_baseline", "C1_balanced_plain_contrastive", "C2_TACO", "C3_shuffled_margin_control"))
    value = lambda row, key: float(row[key])
    label2_correct = round(value(c2, "label2_recall") * 52)
    z = 1.959963984540054
    proportion = label2_correct / 52
    denominator = 1 + z * z / 52
    center = (proportion + z * z / (2 * 52)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / 52 + z * z / (4 * 52 * 52)) / denominator
    wilson = [max(0.0, center - radius), min(1.0, center + radius)]
    core = value(c2, "label2_recall") >= 0.05 and label2_correct >= 3
    low_guard = value(c2, "low_to_high_rate") <= value(c0, "low_to_high_rate")
    overall = (
        value(c0, "MAE") - value(c2, "MAE") >= 0.003
        or value(c2, "QWK") - value(c0, "QWK") >= 0.005
        or value(c2, "Kendall_tau") - value(c0, "Kendall_tau") >= 0.005
    )
    protection = (
        value(c2, "Exact_Match") >= value(c0, "Exact_Match") - 0.005
        and value(c2, "abs_Signed_Bias") <= value(c0, "abs_Signed_Bias") + 0.01
        and value(c2, "label5_recall") >= value(c0, "label5_recall") - 0.02
        and value(c2, "high_to_low_rate") <= value(c0, "high_to_low_rate") + 0.01
    )
    beats_c1_tail = value(c2, "label2_recall") > value(c1, "label2_recall") or value(c2, "low_to_high_rate") < value(c1, "low_to_high_rate")
    protects_c1 = value(c2, "MAE") <= value(c1, "MAE") + 0.005 and value(c2, "QWK") >= value(c1, "QWK") - 0.01 and value(c2, "Exact_Match") >= value(c1, "Exact_Match") - 0.005
    beats_c3 = any((
        value(c2, "label2_recall") > value(c3, "label2_recall"),
        value(c2, "low_to_high_rate") < value(c3, "low_to_high_rate"),
        value(c2, "MAE") < value(c3, "MAE"),
        value(c2, "QWK") > value(c3, "QWK"),
    ))
    bootstrap_by = {(row["comparison"], row["metric"]): row for row in bootstrap}
    comparison = "C2_TACO_vs_C0_E4_baseline"
    no_significant_harm = (
        float(bootstrap_by[(comparison, "MAE")]["ci_low"]) <= 0
        and float(bootstrap_by[(comparison, "QWK")]["ci_high"]) >= 0
        and float(bootstrap_by[(comparison, "Exact_Match")]["ci_high"]) >= 0
    )
    bootstrap_gate = no_significant_harm and low_guard
    checks = {
        "core_label2": core,
        "low_to_high_guard": low_guard,
        "overall_gain": overall,
        "protection": protection,
        "beats_C1_tail": beats_c1_tail,
        "protects_vs_C1": protects_c1,
        "beats_C3": beats_c3,
        "bootstrap_no_significant_harm": no_significant_harm,
        "bootstrap_gate": bootstrap_gate,
    }
    passed = all(checks.values())
    status = "TACO_SEED42_GO" if passed else "TACO_SEED42_STOP"
    decision = {
        "status": status,
        "checks": checks,
        "label2_correct_count": label2_correct,
        "label2_total": 52,
        "label2_recall": value(c2, "label2_recall"),
        "label2_wilson_95_ci": wilson,
        "class2_prediction_count": int(float(c2["pred_count_2"])),
        "recommend_multiseed": passed,
        "stop_positive_small_paper_route": not passed,
        "dev_access_count": 0,
        "test_access_count": 0,
    }
    atomic_json(args.out_dir / "decision/exp44a_seed42_decision.json", decision)
    diagnostics = read_csv(args.out_dir / "tables/exp44a_representation_diagnostics.csv")
    report = [
        "# Exp44A TACO-Score Seed42 Report",
        "",
        f"Final status: **{status}**",
        "",
        "## Gate checks",
        "",
        *[f"- {key}: {'PASS' if result else 'FAIL'}" for key, result in checks.items()],
        "",
        "## Metrics",
        "",
        "| Variant | MAE | QWK | Exact | Kendall | Bias | Bin agreement | L2H | H2L | Label1 | Label2 | Label5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("C0_E4_baseline", "C1_balanced_plain_contrastive", "C2_TACO", "C3_shuffled_margin_control"):
        row = metrics[name]
        report.append(f"| {name} | {value(row,'MAE'):.6f} | {value(row,'QWK'):.6f} | {value(row,'Exact_Match'):.6f} | {value(row,'Kendall_tau'):.6f} | {value(row,'Signed_Bias'):.6f} | {value(row,'Bin_Agreement'):.6f} | {value(row,'low_to_high_rate'):.6f} | {value(row,'high_to_low_rate'):.6f} | {value(row,'label1_recall'):.6f} | {value(row,'label2_recall'):.6f} | {value(row,'label5_recall'):.6f} |")
    report += [
        "",
        f"C2 label2 correct: {label2_correct}/52; Wilson 95% CI=[{wilson[0]:.6f}, {wilson[1]:.6f}]; class-2 predictions={int(float(c2['pred_count_2']))}.",
        "",
        "## Representation diagnostics",
        "",
    ]
    for row in diagnostics:
        report.append(f"- {row['variant']}: nearest-centroid balanced accuracy={float(row['nearest_centroid_balanced_accuracy']):.6f}; label2 nearest class2={row['label2_nearest_centroid_count_2']}/52")
    report += [
        "",
        "## Boundaries",
        "",
        "No API or teacher labels were used. Dev and test access counts are both zero. Raw predictions, triplets, checkpoints, embeddings, and logs remain private and ignored.",
    ]
    (args.out_dir / "reports/exp44a_taco_seed42_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
