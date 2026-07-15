"""Apply the preregistered Exp46A teacher capacity gate."""

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


def number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> None:
    args = parse_args()
    rows = {row["variant"]: row for row in read_csv(args.out_dir / "tables/exp46a_teacher_capacity_metrics.csv")}
    baseline, teacher = rows["K0_E4"], rows["T1_4B_teacher"]
    bootstrap = read_csv(args.out_dir / "tables/exp46a_teacher_question_key_bootstrap.csv")
    bootstrap_by = {row["metric"]: row for row in bootstrap if row["comparison"] == "T1_4B_teacher_vs_K0_E4"}
    checks = {
        "label2_recall": number(teacher, "label2_recall") >= 0.10,
        "label2_correct": number(teacher, "label2_correct") >= 6,
        "label2_precision": number(teacher, "label2_precision") >= 0.10,
        "low_to_high": number(teacher, "low_to_high_rate") <= 0.90 * number(baseline, "low_to_high_rate"),
        "overall_gain": number(teacher, "MAE") <= number(baseline, "MAE") - 0.01 or number(teacher, "QWK") >= number(baseline, "QWK") + 0.015 or number(teacher, "Kendall_tau") >= number(baseline, "Kendall_tau") + 0.01,
        "exact_protection": number(teacher, "Exact_Match") >= number(baseline, "Exact_Match") - 0.01,
        "label5_protection": number(teacher, "label5_recall") >= number(baseline, "label5_recall") - 0.02,
        "high_to_low_protection": number(teacher, "high_to_low_rate") <= number(baseline, "high_to_low_rate") + 0.01,
        "bias_protection": number(teacher, "abs_Signed_Bias") <= number(baseline, "abs_Signed_Bias") + 0.01,
    }
    significant_harm = {
        "MAE": float(bootstrap_by["MAE"]["ci_low"]) > 0,
        "QWK": float(bootstrap_by["QWK"]["ci_high"]) < 0,
        "Exact_Match": float(bootstrap_by["Exact_Match"]["ci_high"]) < 0,
        "Kendall_tau": float(bootstrap_by["Kendall_tau"]["ci_high"]) < 0,
    }
    checks["bootstrap_no_significant_overall_harm"] = not any(significant_harm.values())
    status = "TEACHER_CAPACITY_GO" if all(checks.values()) else "TEACHER_CAPACITY_NO_GO"
    decision = {"status": status, "checks": checks, "significant_harm": significant_harm, "baseline": baseline, "teacher": teacher, "next_stage": "student_hato" if status.endswith("_GO") and status != "TEACHER_CAPACITY_NO_GO" else "stop_positive_small_paper_route", "dev_access_count": 0, "test_access_count": 0}
    write_json(args.out_dir / "decision/exp46a_teacher_capacity_decision.json", decision)
    report = ["# Exp46A Teacher Capacity Gate", "", f"Decision: **{status}**", "", "The 4B teacher uses only the locked human score distribution and the same question-key GroupCV split; no teacher relabeling, dev, or test data are used.", "", "## Gate checks", ""]
    report.extend(f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in checks.items())
    report.extend(["", "## Key metrics", "", f"- Label2: {int(number(teacher, 'label2_correct'))}/52 correct; recall={number(teacher, 'label2_recall'):.4f}; precision={number(teacher, 'label2_precision'):.4f}.", f"- Low-to-high: teacher={number(teacher, 'low_to_high_rate'):.4f}; 0.6B E4={number(baseline, 'low_to_high_rate'):.4f}.", f"- MAE/QWK/Exact/Kendall: {number(teacher, 'MAE'):.4f} / {number(teacher, 'QWK'):.4f} / {number(teacher, 'Exact_Match'):.4f} / {number(teacher, 'Kendall_tau'):.4f}.", "", "No test data were read."])
    report_path = args.out_dir / "reports/exp46a_teacher_capacity_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "checks": checks, "test_access_count": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
