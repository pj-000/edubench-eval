"""Apply the preregistered Exp45A DOPR Seed42 decision gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from thesis_exp.exp45_dopr_head.common import EXP44_ROOT, ROOT, atomic_json, read_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    return parser.parse_args()


def number(row: dict, key: str) -> float:
    return float(row[key])


def wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - radius, center + radius


def main() -> None:
    args = parse_args()
    metrics = {row["variant"]: row for row in read_csv(args.out_dir / "tables/exp45a_seed42_metrics.csv")}
    labels = {row["variant"]: row for row in read_csv(args.out_dir / "tables/exp45a_label_recall.csv")}
    c1 = next(row for row in read_csv(EXP44_ROOT / "tables/exp44a_seed42_metrics.csv") if row["variant"] == "C1_balanced_plain_contrastive")
    bootstrap = read_csv(args.out_dir / "tables/exp45a_question_key_bootstrap_ci.csv")
    h4 = metrics["H4_DOPR"]
    h1, h2, h3 = metrics["H1_vanilla_cRT"], metrics["H2_distributional_ordinal_cRT"], metrics["H3_prototype_cRT_no_prior"]
    correct = int(float(labels["H4_DOPR"]["label2_correct_count"]))
    total = int(float(labels["H4_DOPR"]["label2_total"]))
    wilson_low, wilson_high = wilson(correct, total)

    core = {
        "label2_recall": number(h4, "label2_recall") >= 0.05,
        "label2_correct_count": correct >= 3,
        "low_to_high": number(h4, "low_to_high_rate") <= 0.7368421053 + 1e-12,
    }
    overall = {
        "MAE": number(c1, "MAE") - number(h4, "MAE") >= 0.003,
        "QWK": number(h4, "QWK") - number(c1, "QWK") >= 0.005,
        "Kendall_tau": number(h4, "Kendall_tau") - number(c1, "Kendall_tau") >= 0.005,
    }
    protection = {
        "Exact": number(h4, "Exact_Match") >= number(c1, "Exact_Match") - 0.005,
        "abs_Signed_Bias": number(h4, "abs_Signed_Bias") <= number(c1, "abs_Signed_Bias") + 0.01,
        "label5_recall": number(h4, "label5_recall") >= number(c1, "label5_recall") - 0.02,
        "high_to_low": number(h4, "high_to_low_rate") <= number(c1, "high_to_low_rate") + 0.01,
        "mean_score_decline": number(c1, "Signed_Bias") - number(h4, "Signed_Bias") <= 0.10,
    }

    def tail_better(reference: dict) -> bool:
        return number(h4, "label2_recall") > number(reference, "label2_recall") or number(h4, "low_to_high_rate") < number(reference, "low_to_high_rate")

    def noninferior(reference: dict) -> bool:
        return number(h4, "MAE") <= number(reference, "MAE") + 0.005 and number(h4, "QWK") >= number(reference, "QWK") - 0.01 and number(h4, "Exact_Match") >= number(reference, "Exact_Match") - 0.005

    h3_stronger_tail = number(h3, "label2_recall") > number(h4, "label2_recall") or number(h3, "low_to_high_rate") < number(h4, "low_to_high_rate")
    method = {
        "tail_vs_H1": tail_better(h1),
        "tail_vs_H2": tail_better(h2),
        "noninferior_H1": noninferior(h1),
        "noninferior_H2": noninferior(h2),
        "H3_protection_if_needed": (not h3_stronger_tail) or number(h4, "label5_recall") > number(h3, "label5_recall") or number(h4, "high_to_low_rate") < number(h3, "high_to_low_rate"),
    }
    c1_rows = [row for row in bootstrap if row["comparison"] == "H4_DOPR_vs_Exp44_C1_balanced_plain_contrastive" and row["metric"] != "ROW_LEVEL_UNAVAILABLE"]
    if c1_rows:
        by_metric = {row["metric"]: row for row in c1_rows}
        bootstrap_gate = {
            "row_level_available": True,
            "MAE_no_significant_harm": float(by_metric["MAE"]["ci_lower"]) <= 0,
            "QWK_no_significant_harm": float(by_metric["QWK"]["ci_upper"]) >= 0,
            "Exact_no_significant_harm": float(by_metric["Exact_Match"]["ci_upper"]) >= 0,
            "low_to_high_point_not_worse": float(by_metric["low_to_high_rate"]["delta_point"]) <= 0,
        }
    else:
        # Missing C1 row-level predictions cannot be treated as statistical success.
        bootstrap_gate = {"row_level_available": False, "MAE_no_significant_harm": False, "QWK_no_significant_harm": False, "Exact_no_significant_harm": False, "low_to_high_point_not_worse": number(h4, "low_to_high_rate") <= number(c1, "low_to_high_rate")}
    passed = all(core.values()) and any(overall.values()) and all(protection.values()) and all(method.values()) and all(value for key, value in bootstrap_gate.items() if key != "row_level_available") and bootstrap_gate["row_level_available"]
    decision = {
        "status": "DOPR_SEED42_GO" if passed else "DOPR_SEED42_STOP",
        "recommend_full_multiseed": passed,
        "stop_positive_small_paper_route": not passed,
        "core_tail_gate": core,
        "overall_gain_gate": overall,
        "protection_gate": protection,
        "method_attribution_gate": method,
        "bootstrap_gate": bootstrap_gate,
        "label2_correct_count": correct,
        "label2_total": total,
        "label2_wilson_95_ci": [wilson_low, wilson_high],
        "class2_prediction_count": int(float(labels["H4_DOPR"]["class2_prediction_count"])),
        "class2_prediction_precision": float(labels["H4_DOPR"]["class2_prediction_precision"]),
        "dev_access_count": 0,
        "test_access_count": 0,
    }
    atomic_json(args.out_dir / "decision/exp45a_seed42_decision.json", decision)
    report = [
        "# Exp45A DOPR-Head Seed42 Report", "", f"Final status: **{decision['status']}**", "",
        "## Metrics", "",
        "| Variant | MAE | QWK | Exact | Kendall | Bias | L2H | H2L | L2 recall | L5 recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in ("H0_E4_natural_head", "H1_vanilla_cRT", "H2_distributional_ordinal_cRT", "H3_prototype_cRT_no_prior", "H4_DOPR"):
        row = metrics[variant]
        report.append(f"| {variant} | {number(row,'MAE'):.6f} | {number(row,'QWK'):.6f} | {number(row,'Exact_Match'):.6f} | {number(row,'Kendall_tau'):.6f} | {number(row,'Signed_Bias'):.6f} | {number(row,'low_to_high_rate'):.6f} | {number(row,'high_to_low_rate'):.6f} | {number(row,'label2_recall'):.6f} | {number(row,'label5_recall'):.6f} |")
    report.extend(["", f"- H4 label2: {correct}/{total}; Wilson 95% CI [{wilson_low:.6f}, {wilson_high:.6f}].", f"- H4 class2 predictions/precision: {decision['class2_prediction_count']} / {decision['class2_prediction_precision']:.6f}.", f"- Core gate: {core}.", f"- Overall gain gate: {overall}.", f"- Protection gate: {protection}.", f"- Method attribution: {method}.", f"- Bootstrap gate: {bootstrap_gate}.", "- No API; no teacher labels; no dev/test access; private artifacts remain ignored."])
    (args.out_dir / "reports/exp45a_dopr_seed42_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
