"""Compute Exp42A factorial contrasts and apply the preregistered decision gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp42_rubidist.common import METRICS, ROOT, SEEDS, write_csv, write_json  # noqa: E402

V00 = "v00_hard_no_rubric"
V01 = "v01_soft_no_rubric"
V10 = "v10_hard_raw_rubric"
V11 = "v11_soft_raw_rubric"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def f(value: Any) -> float:
    return float(value)


def main() -> None:
    args = parse_args()
    seed_rows = read_csv(args.out_dir / "tables/exp42a_selected_metrics_by_seed.csv")
    by_seed = {(row["variant"], int(row["seed"])): row for row in seed_rows}
    means = {row["variant"]: row for row in read_csv(args.out_dir / "tables/exp42a_multiseed_summary.csv")}
    crossed = {
        (row["left_variant"], row["right_variant"], row["metric"]): row
        for row in read_csv(args.out_dir / "tables/exp42a_crossed_seed_question_bootstrap_ci.csv")
    }

    effects = []
    for seed in (*SEEDS, "mean"):
        def value(variant: str, metric: str) -> float:
            if seed == "mean":
                return f(means[variant][f"{metric}_mean"])
            return f(by_seed[(variant, int(seed))][metric])

        for metric in METRICS:
            soft_without = value(V01, metric) - value(V00, metric)
            soft_with = value(V11, metric) - value(V10, metric)
            rubric_hard = value(V10, metric) - value(V00, metric)
            rubric_soft = value(V11, metric) - value(V01, metric)
            effects.append(
                {
                    "seed": seed,
                    "metric": metric,
                    "soft_effect_without_rubric_v01_minus_v00": soft_without,
                    "soft_effect_with_rubric_v11_minus_v10": soft_with,
                    "rubric_effect_with_hard_v10_minus_v00": rubric_hard,
                    "rubric_effect_with_soft_v11_minus_v01": rubric_soft,
                    "interaction": soft_with - soft_without,
                    "direction_note": "raw signed difference; negative favors left for lower-is-better metrics",
                }
            )
    write_csv(args.out_dir / "tables/exp42a_factorial_effects.csv", effects)

    def mean(variant: str, metric: str) -> float:
        return f(means[variant][f"{metric}_mean"])

    main_mae_delta = mean(V11, "MAE") - mean(V01, "MAE")
    main_qwk_delta = mean(V11, "QWK") - mean(V01, "QWK")
    mae_ci = crossed[(V11, V01, "MAE")]
    qwk_ci = crossed[(V11, V01, "QWK")]
    rubric_gain_mae = main_mae_delta <= -0.01 and f(mae_ci["ci_upper"]) < 0
    rubric_gain_qwk = main_qwk_delta >= 0.02 and f(qwk_ci["ci_lower"]) > 0

    soft_noninferiority = {
        "mae": mean(V11, "MAE") <= mean(V10, "MAE") + 0.005,
        "qwk": mean(V11, "QWK") >= mean(V10, "QWK") - 0.01,
        "exact": mean(V11, "Exact_Match") >= mean(V10, "Exact_Match") - 0.005,
        "brier_or_rps": mean(V11, "human_Brier") < mean(V10, "human_Brier")
        or mean(V11, "human_RPS") < mean(V10, "human_RPS"),
    }
    guards = {
        "exact": mean(V11, "Exact_Match") >= mean(V01, "Exact_Match") - 0.005,
        "kendall": mean(V11, "Kendall_tau") >= mean(V01, "Kendall_tau") - 0.01,
        "absolute_bias": abs(mean(V11, "Signed_Bias")) <= abs(mean(V01, "Signed_Bias")) + 0.01,
        "low_to_high": mean(V11, "low_to_high_rate") <= mean(V01, "low_to_high_rate"),
        "high_to_low": mean(V11, "high_to_low_rate") <= mean(V01, "high_to_low_rate") + 0.01,
        "label5": mean(V11, "label5_recall") >= mean(V01, "label5_recall") - 0.02,
    }
    seed_stability = []
    for seed in SEEDS:
        delta_mae = f(by_seed[(V11, seed)]["MAE"]) - f(by_seed[(V01, seed)]["MAE"])
        delta_qwk = f(by_seed[(V11, seed)]["QWK"]) - f(by_seed[(V01, seed)]["QWK"])
        delta_label5 = f(by_seed[(V11, seed)]["label5_recall"]) - f(by_seed[(V01, seed)]["label5_recall"])
        seed_stability.append(
            {
                "seed": seed,
                "delta_MAE": delta_mae,
                "delta_QWK": delta_qwk,
                "delta_label5_recall": delta_label5,
                "favorable_mae_or_qwk": delta_mae < 0 or delta_qwk > 0,
                "mae_guard": delta_mae <= 0.02,
                "label5_guard": delta_label5 >= -0.04,
            }
        )
    stability = {
        "at_least_two_favorable": sum(bool(row["favorable_mae_or_qwk"]) for row in seed_stability) >= 2,
        "all_seed_mae_guards": all(bool(row["mae_guard"]) for row in seed_stability),
        "all_seed_label5_guards": all(bool(row["label5_guard"]) for row in seed_stability),
    }
    gate_a = rubric_gain_mae or rubric_gain_qwk
    gate_b = all(soft_noninferiority.values())
    gate_c = all(guards.values())
    gate_d = all(stability.values())
    main_go = gate_a and gate_b and gate_c and gate_d

    secondary_mae_delta = mean(V10, "MAE") - mean(V00, "MAE")
    secondary_qwk_delta = mean(V10, "QWK") - mean(V00, "QWK")
    secondary_mae_ci = crossed[(V10, V00, "MAE")]
    secondary_qwk_ci = crossed[(V10, V00, "QWK")]
    secondary_gain = (
        secondary_mae_delta <= -0.01 and f(secondary_mae_ci["ci_upper"]) < 0
    ) or (
        secondary_qwk_delta >= 0.02 and f(secondary_qwk_ci["ci_lower"]) > 0
    )
    secondary_guards = {
        "exact": mean(V10, "Exact_Match") >= mean(V00, "Exact_Match") - 0.005,
        "kendall": mean(V10, "Kendall_tau") >= mean(V00, "Kendall_tau") - 0.01,
        "absolute_bias": abs(mean(V10, "Signed_Bias")) <= abs(mean(V00, "Signed_Bias")) + 0.01,
        "low_to_high": mean(V10, "low_to_high_rate") <= mean(V00, "low_to_high_rate"),
        "high_to_low": mean(V10, "high_to_low_rate") <= mean(V00, "high_to_low_rate") + 0.01,
        "label5": mean(V10, "label5_recall") >= mean(V00, "label5_recall") - 0.02,
    }
    secondary_seed_rows = []
    for seed in SEEDS:
        delta_mae = f(by_seed[(V10, seed)]["MAE"]) - f(by_seed[(V00, seed)]["MAE"])
        delta_qwk = f(by_seed[(V10, seed)]["QWK"]) - f(by_seed[(V00, seed)]["QWK"])
        secondary_seed_rows.append({"seed": seed, "delta_MAE": delta_mae, "delta_QWK": delta_qwk, "favorable": delta_mae < 0 or delta_qwk > 0})
    secondary_stable = sum(bool(row["favorable"]) for row in secondary_seed_rows) >= 2
    secondary_go = secondary_gain and all(secondary_guards.values()) and secondary_stable

    if main_go:
        status = "RUBIDIST_GO"
    elif secondary_go:
        status = "RAW_RUBRIC_ONLY_SIGNAL"
    else:
        status = "FACTORIAL_STOP"
    decision = {
        "status": status,
        "main_method": V11,
        "main_gates": {
            "A_rubric_gain_relative_to_v01": gate_a,
            "A_mae_path": rubric_gain_mae,
            "A_qwk_path": rubric_gain_qwk,
            "B_soft_noninferiority_relative_to_v10": gate_b,
            "B_details": soft_noninferiority,
            "C_overall_tail_guards_relative_to_v01": gate_c,
            "C_details": guards,
            "D_stability": gate_d,
            "D_details": stability,
            "D_per_seed": seed_stability,
        },
        "secondary_raw_rubric_only": {
            "passed": secondary_go,
            "gain_with_crossed_ci": secondary_gain,
            "guards": secondary_guards,
            "at_least_two_favorable_seeds": secondary_stable,
            "per_seed": secondary_seed_rows,
        },
        "rubric_conditioning_supported": status in {"RUBIDIST_GO", "RAW_RUBRIC_ONLY_SIGNAL"},
        "human_soft_interaction_supported": status == "RUBIDIST_GO",
        "recommend_final_dev_campaign": status == "RUBIDIST_GO",
        "stop_positive_small_paper_route": status == "FACTORIAL_STOP",
        "no_posthoc_variant_changes": True,
        "dev_access_count": 0,
        "test_access_count": 0,
    }
    write_json(args.out_dir / "decision/exp42a_rubidist_decision.json", decision)

    report = [
        "# Exp42A RubiDist multiseed report",
        "",
        f"- Final status: **{status}**",
        "- Formal matrix: 4 variants x 5 question-key folds x 3 seeds = 60 runs.",
        "- All models used fixed final epoch 10; no checkpoint selection was performed.",
        f"- Main rubric gain gate: `{str(gate_a).lower()}` (MAE delta={main_mae_delta:.6f}; QWK delta={main_qwk_delta:.6f}).",
        f"- Soft-target non-inferiority gate: `{str(gate_b).lower()}`.",
        f"- Overall/tail guard gate: `{str(gate_c).lower()}`.",
        f"- Three-seed stability gate: `{str(gate_d).lower()}`.",
        f"- Secondary raw-rubric-only signal: `{str(secondary_go).lower()}`.",
        f"- Recommend final dev campaign: `{str(decision['recommend_final_dev_campaign']).lower()}`.",
        f"- Stop positive small-paper route: `{str(decision['stop_positive_small_paper_route']).lower()}`.",
        "",
        "## Boundaries",
        "",
        "- No API or teacher labeling was used.",
        "- Training used standard hard/soft cross-entropy only, with no custom loss or sample weighting.",
        "- No paper-like dev/test data or historical evaluation predictions were accessed.",
        "- Exp41 post-fit audit is descriptive and does not affect this decision.",
    ]
    path = args.out_dir / "reports/exp42a_rubidist_multiseed_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
