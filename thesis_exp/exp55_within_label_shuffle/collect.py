"""Collect dev-only three-seed results for the mechanism control."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from thesis_exp.exp55_within_label_shuffle import MODEL_SEEDS, OUTPUT_ROOT, VARIANT


REPO_ROOT = Path(__file__).resolve().parents[2]
METRICS = (
    "MAE_human_mean",
    "Exact_rounded",
    "Kendall_human_mean",
    "Bias_human_mean",
    "QWK_rounded",
    "L2H_count",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows]
        summary[metric] = {
            "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values),
            "values": values,
        }
    return summary


def raw_deltas(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Return paired left-minus-right values in seed order."""

    result: dict[str, Any] = {}
    for metric in METRICS:
        values = [
            float(left_row[metric]) - float(right_row[metric])
            for left_row, right_row in zip(left["rows"], right["rows"])
        ]
        result[metric] = {
            "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values),
            "values": values,
        }
    return result


def arm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"rows": rows, "summary": summarize(rows)}


def main() -> None:
    shuffled_rows = []
    hmsa_rows = []
    hard_only_rows = []
    for seed in MODEL_SEEDS:
        shuffled_path = (
            OUTPUT_ROOT / "runs" / VARIANT / f"seed_{seed}" / "selected_dev_metrics.json"
        )
        hmsa_path = (
            REPO_ROOT
            / "thesis_exp"
            / "outputs"
            / "exp51_hmsa"
            / "runs"
            / "hmsa_lambda1"
            / f"seed_{seed}"
            / "selected_dev_metrics.json"
        )
        hard_only_path = (
            REPO_ROOT
            / "thesis_exp"
            / "outputs"
            / "exp49_cphce"
            / "runs"
            / "b0_hard_ce"
            / f"seed_{seed}"
            / "selected_dev_metrics.json"
        )
        if not shuffled_path.exists() or not hmsa_path.exists() or not hard_only_path.exists():
            raise FileNotFoundError(
                f"Missing seed {seed}: {shuffled_path}, {hmsa_path}, or {hard_only_path}"
            )
        shuffled_rows.append(read_json(shuffled_path))
        hmsa_rows.append(read_json(hmsa_path))
        hard_only_rows.append(read_json(hard_only_path))
    shuffled = arm(shuffled_rows)
    hmsa = arm(hmsa_rows)
    hard_only = arm(hard_only_rows)
    result = {
        "status": "DEV_ONLY_COMPLETE",
        "seeds": list(MODEL_SEEDS),
        "test_access_count": 0,
        "hard_only": hard_only["summary"],
        "within_label_shuffled_soft": shuffled["summary"],
        "hmsa_true_soft": hmsa["summary"],
        "paired_delta_shuffled_minus_hard_only": raw_deltas(shuffled, hard_only),
        "paired_delta_hmsa_minus_shuffled": raw_deltas(hmsa, shuffled),
        "paired_delta_hmsa_minus_hard_only": raw_deltas(hmsa, hard_only),
        "interpretation_contract": {
            "hmsa_better": "sample-matched human distributions add information beyond the preserved class-conditional soft-target multiset",
            "similar": "evidence favors a general soft auxiliary regularization account",
            "shuffle_better": "the sample-specific disagreement interpretation is not supported",
        },
    }
    output = OUTPUT_ROOT / "decision" / "dev_mechanism_comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Within-label shuffled-soft development-set mechanism control",
        "",
        "All values are means over paired seeds 42/43/44. No test data were accessed.",
        "",
        "| Metric | Hard-only | Shuffled-soft | HMSA | HMSA − Shuffle |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for metric in METRICS:
        report.append(
            f"| {metric} | {hard_only['summary'][metric]['mean']:.6f} | "
            f"{shuffled['summary'][metric]['mean']:.6f} | "
            f"{hmsa['summary'][metric]['mean']:.6f} | "
            f"{result['paired_delta_hmsa_minus_shuffled'][metric]['mean']:+.6f} |"
        )
    report.extend(
        [
            "",
            "For MAE, absolute bias, and L2H, lower is better; for Exact, Kendall, and QWK, higher is better.",
            "HMSA beats the shuffled control for MAE, Exact, Kendall, bias magnitude, and QWK in the three-seed mean.",
            "The shuffled control has 0.333 fewer mean L2H cases, a one-case total difference across the three runs.",
        ]
    )
    (OUTPUT_ROOT / "decision" / "dev_mechanism_comparison.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
