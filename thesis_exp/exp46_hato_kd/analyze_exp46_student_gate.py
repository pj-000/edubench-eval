"""Apply the preregistered Exp46A student transfer gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from thesis_exp.exp46_hato_kd.common import ROOT, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def n(row: dict[str, str], key: str) -> float:
    return float(row[key])


def better(main: dict[str, str], control: dict[str, str]) -> bool:
    return (
        n(main, "label2_recall") > n(control, "label2_recall")
        or n(main, "low_to_high_rate") < n(control, "low_to_high_rate")
        or n(main, "MAE") < n(control, "MAE")
        or n(main, "QWK") > n(control, "QWK")
        or n(main, "Kendall_tau") > n(control, "Kendall_tau")
    )


def main() -> None:
    args = parse_args()
    rows = {row["variant"]: row for row in read_csv(args.out_dir / "tables/exp46a_student_metrics.csv")}
    baseline = rows["C1_strongest_point"]
    k1, k2, k3 = rows["K1_standard_kd"], rows["K2_hato_kd"], rows["K3_shuffled_hato_control"]
    collection = json.loads((args.out_dir / "hashes/exp46a_student_collection.json").read_text(encoding="utf-8"))
    bootstrap_rows = read_csv(args.out_dir / "tables/exp46a_student_question_key_bootstrap.csv")
    bootstrap = {row["metric"]: row for row in bootstrap_rows if row["comparison"] == "K2_hato_kd_vs_C1_strongest_point"}
    significant_harm = {
        "MAE": float(bootstrap["MAE"]["ci_low"]) > 0,
        "QWK": float(bootstrap["QWK"]["ci_high"]) < 0,
        "Exact_Match": float(bootstrap["Exact_Match"]["ci_high"]) < 0,
    }
    checks = {
        "label2_recall": n(k2, "label2_recall") >= 0.05,
        "label2_correct": n(k2, "label2_correct") >= 3,
        "low_to_high": n(k2, "low_to_high_rate") <= 0.7368421052631579,
        "overall_gain": n(k2, "MAE") <= n(baseline, "MAE") - 0.003 or n(k2, "QWK") >= n(baseline, "QWK") + 0.005 or n(k2, "Kendall_tau") >= n(baseline, "Kendall_tau") + 0.005,
        "exact_protection": n(k2, "Exact_Match") >= n(baseline, "Exact_Match") - 0.005,
        "label5_protection": n(k2, "label5_recall") >= n(baseline, "label5_recall") - 0.02,
        "high_to_low_protection": n(k2, "high_to_low_rate") <= n(baseline, "high_to_low_rate") + 0.01,
        "mean_score_downshift": n(baseline, "mean_pred") - n(k2, "mean_pred") <= 0.10,
        "bias_protection": n(k2, "abs_Signed_Bias") <= n(baseline, "abs_Signed_Bias") + 0.01,
        "better_than_standard_kd": better(k2, k1),
        "better_than_shuffled_control": better(k2, k3),
        "shuffled_donor_change": float(collection["k3_mean_donor_change_rate"]) >= 0.80,
        "bootstrap_no_significant_overall_harm": not any(significant_harm.values()),
    }
    status = "HATO_STUDENT_GO" if all(checks.values()) else "HATO_STUDENT_NO_GO"
    decision = {"status": status, "checks": checks, "significant_harm": significant_harm, "baseline": baseline, "K1": k1, "K2": k2, "K3": k3, "stop_positive_small_paper_route": status == "HATO_STUDENT_NO_GO", "open_test": False, "dev_access_count": 0, "test_access_count": 0}
    write_json(args.out_dir / "decision/exp46a_student_transfer_decision.json", decision)
    report = ["# Exp46A HATO-KD Student Transfer Gate", "", f"Decision: **{status}**", "", "## Gate checks", ""]
    report.extend(f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in checks.items())
    report.extend(["", "## K2 key metrics", "", f"- Label2: {int(n(k2, 'label2_correct'))}/52 correct; recall={n(k2, 'label2_recall'):.4f}.", f"- Low-to-high={n(k2, 'low_to_high_rate'):.4f}; MAE={n(k2, 'MAE'):.4f}; QWK={n(k2, 'QWK'):.4f}; Exact={n(k2, 'Exact_Match'):.4f}; Kendall={n(k2, 'Kendall_tau'):.4f}.", "", "K3 is a same-hard-label and same-language shuffled teacher-logit control. No dev or test data were read."])
    report_path = args.out_dir / "reports/exp46a_student_transfer_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "checks": checks, "test_access_count": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
