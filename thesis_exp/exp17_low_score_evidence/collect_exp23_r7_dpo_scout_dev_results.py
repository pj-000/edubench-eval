"""Collect Exp23 R7 DPO scout dev predictions.

The main comparison is R7D vs R7E:

- R7D: chosen = recovered human reason + gold score.
- R7E: chosen = gold score only, same R7D source pair pool.

R7F is reported as an auxiliary score-reason consistency scout and should not
be over-interpreted as a natural real-error DPO result.
"""

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

from thesis_exp.exp17_low_score_evidence import collect_exp19_sft_first_round_dev_results as sft_collect  # noqa: E402
from thesis_exp.exp17_low_score_evidence.collect_exp19_r5_dpo_scout_dev_results import (  # noqa: E402
    D1_FIELDS,
    FAILURE_FIELDS,
    METRIC_FIELDS,
    copy_d1_fields,
    copy_failure_fields,
    copy_metric_fields,
    d1_score_cap_rate,
)
from thesis_exp.src.edujudge.utils.io import write_csv, write_json, write_text  # noqa: E402


DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp23_r7_dpo_scout")
DEFAULT_PREDICTION_ROOT = DEFAULT_OUT_DIR / "dev_predictions"
DEFAULT_REFERENCE = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp19_sft_dpo_datasets_seed42/tables/exp19_dev_reference.csv"
)
DEFAULT_D1_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/d1_hidden_failure_audit_seed42_dev/"
    "summary_human_rationale_recovered"
)
DEFAULT_BASELINE_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_sft_second_round")

PRIMARY_R7D = "r7d_reason_real_s100_b0p03_lr5em6"
PRIMARY_R7E = "r7e_matched_score_only_s100_b0p03_lr5em6"
AUX_R7F = "r7f_score_reason_consistency_s100_b0p03_lr5em6"

BASELINE_META = {
    "r2c_clean_reason_score_balanced": {
        "run_name": "r2c_clean_reason_score_balanced",
        "init_adapter": "r2c_clean_reason_score_balanced",
        "dpo_dataset": "none_init_baseline",
    },
    "r4b_shuffled_reason_balanced": {
        "run_name": "r4b_shuffled_reason_balanced",
        "init_adapter": "r4b_shuffled_reason_balanced",
        "dpo_dataset": "none_init_baseline",
    },
}


def read_csv_if_exists(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def fmt(value: Any, digits: int = 4) -> str:
    val = to_float(value)
    if math.isnan(val):
        return "nan"
    return f"{val:.{digits}f}"


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def load_run_matrix(path: Path) -> list[dict[str, str]]:
    rows = read_csv_if_exists(path)
    if not rows:
        raise SystemExit(f"Exp23 run matrix missing or empty: {path}")
    return rows


def collect_run_predictions(
    run_rows: list[dict[str, str]],
    prediction_root: Path,
    reference: list[dict[str, str]],
    d1_dir: Path,
    allow_missing_predictions: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    metric_rows: list[dict[str, Any]] = []
    d1_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for run in run_rows:
        run_name = run["run_name"]
        run_dir = prediction_root / run_name
        try:
            prediction_file = sft_collect.find_prediction_file(run_dir)
        except FileNotFoundError:
            missing.append(run_name)
            if allow_missing_predictions:
                continue
            raise
        predictions = sft_collect.load_prediction_records(prediction_file)
        parsed_rows = sft_collect.align_predictions(reference, predictions, run_name, run_name)
        meta = {
            "run_name": run_name,
            "init_adapter": "r2c_clean_reason_score_balanced",
            "dpo_dataset": run.get("dataset_family") or run.get("dataset") or "",
        }
        metric_rows.append(copy_metric_fields(sft_collect.metric_summary(parsed_rows, run_name, run_name, "dev"), meta))
        d1_rows.append(copy_d1_fields(sft_collect.d1_eval_row(parsed_rows, d1_dir, run_name, run_name), run_name))
        failure_rows.append(
            copy_failure_fields(
                sft_collect.failure_type_eval_row(parsed_rows, d1_dir, run_name, run_name),
                run_name,
                d1_score_cap_rate(parsed_rows, d1_dir),
            )
        )
    return metric_rows, d1_rows, failure_rows, missing


def include_baselines(
    baseline_dir: Path,
    metric_rows: list[dict[str, Any]],
    d1_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
) -> dict[str, list[str]]:
    metrics = {
        row.get("run_name", ""): row
        for row in read_csv_if_exists(baseline_dir / "tables" / "exp19_sft_second_round_dev_metrics.csv")
    }
    d1 = {
        row.get("run_name", ""): row
        for row in read_csv_if_exists(baseline_dir / "tables" / "exp19_sft_second_round_dev_d1_hidden_eval.csv")
    }
    failure = {
        row.get("run_name", ""): row
        for row in read_csv_if_exists(baseline_dir / "tables" / "exp19_sft_second_round_dev_failure_type_eval.csv")
    }
    included: list[str] = []
    missing: list[str] = []
    for run_name, meta in BASELINE_META.items():
        if run_name not in metrics:
            missing.append(run_name)
            continue
        metric_rows.append(copy_metric_fields(metrics[run_name], meta))
        if run_name in d1:
            d1_rows.append(copy_d1_fields(d1[run_name], run_name))
        if run_name in failure:
            failure_rows.append(copy_failure_fields(failure[run_name], run_name))
        included.append(run_name)
    return {"included": included, "missing": missing}


def by_run(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("run_name", "")): row for row in rows}


def compare_primary(metric_rows: list[dict[str, Any]], d1_rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = by_run(metric_rows)
    d1 = by_run(d1_rows)
    if PRIMARY_R7D not in metrics or PRIMARY_R7E not in metrics:
        return {
            "available": False,
            "recommendation": "wait_for_primary_predictions",
            "reason": "R7D and R7E predictions are both required for the primary Exp23 comparison.",
        }
    r7d = metrics[PRIMARY_R7D]
    r7e = metrics[PRIMARY_R7E]
    mae_delta = to_float(r7d.get("MAE")) - to_float(r7e.get("MAE"))
    qwk_delta = to_float(r7d.get("QWK")) - to_float(r7e.get("QWK"))
    l2h_delta = to_float(r7d.get("low_to_high_rate")) - to_float(r7e.get("low_to_high_rate"))
    label5_delta = to_float(r7d.get("label5_recall")) - to_float(r7e.get("label5_recall"))
    d1_delta = to_float(d1.get(PRIMARY_R7D, {}).get("pred_ge4_rate_d1_hidden")) - to_float(
        d1.get(PRIMARY_R7E, {}).get("pred_ge4_rate_d1_hidden")
    )

    reason_helps = (
        l2h_delta < 0
        and mae_delta <= 0.03
        and qwk_delta >= -0.03
        and label5_delta >= -0.05
    )
    if reason_helps:
        recommendation = "reason_dpo_promising_continue_orc_src"
        reason = "R7D improves low-to-high over the matched score-only R7E without breaking accuracy guardrails."
    else:
        recommendation = "reason_dpo_not_yet_supported"
        reason = "R7D does not clearly beat the exactly matched score-only R7E control under ordinary DPO."
    return {
        "available": True,
        "recommendation": recommendation,
        "reason": reason,
        "deltas_r7d_minus_r7e": {
            "MAE": mae_delta,
            "QWK": qwk_delta,
            "low_to_high_rate": l2h_delta,
            "label5_recall": label5_delta,
            "D1_hidden_pred_ge4_rate": d1_delta,
        },
        "guardrail": {
            "MAE_delta_max": 0.03,
            "QWK_delta_min": -0.03,
            "label5_recall_delta_min": -0.05,
            "requires_low_to_high_improvement": True,
        },
    }


def write_report(
    out_dir: Path,
    metric_rows: list[dict[str, Any]],
    d1_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    decision: dict[str, Any],
    baseline_summary: dict[str, list[str]],
    missing_predictions: list[str],
) -> None:
    d1_by_name = by_run(d1_rows)
    failure_by_name = by_run(failure_rows)
    lines = [
        "# Exp23 R7 DPO Scout Dev Evaluation",
        "",
        "Exp23 is an ordinary DPO sanity check for R7 data. The primary comparison is R7D",
        "human-reason chosen vs R7E exactly matched score-only control.",
        "",
        "## Dev Metrics",
        "",
        "| run | dataset | MAE | QWK | low-to-high | high-to-low | label2 recall | label5 recall | D1 pred>=4 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        d1 = d1_by_name.get(str(row["run_name"]), {})
        lines.append(
            f"| `{row['run_name']}` | `{row['dpo_dataset']}` | {fmt(row['MAE'])} | {fmt(row['QWK'])} | "
            f"{fmt(row['low_to_high_rate'])} | {fmt(row['high_to_low_rate'])} | "
            f"{fmt(row['label2_recall'])} | {fmt(row['label5_recall'])} | "
            f"{fmt(d1.get('pred_ge4_rate_d1_hidden'))} |"
        )
    lines.extend(
        [
            "",
            "## Structured Failure Diagnostics",
            "",
            "| run | failure micro-F1 | D1 nonempty failure | score_cap nonnull |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in metric_rows:
        failure = failure_by_name.get(str(row["run_name"]), {})
        lines.append(
            f"| `{row['run_name']}` | {fmt(failure.get('failure_type_micro_f1'))} | "
            f"{fmt(failure.get('major_failure_nonempty_rate_on_d1_hidden'))} | "
            f"{fmt(failure.get('score_cap_nonnull_rate_on_d1_hidden'))} |"
        )
    lines.extend(
        [
            "",
            "## Primary Decision",
            "",
            f"- recommendation: `{decision['recommendation']}`",
            f"- reason: {decision['reason']}",
            "",
            "R7F is an auxiliary consistency scout. It should not be used to claim that natural",
            "reason-aware real-error DPO works unless R7D also beats R7E.",
            "",
            "## Sources",
            "",
            f"- included baseline rows: {', '.join(baseline_summary.get('included') or []) or 'none'}",
            f"- missing baseline rows: {', '.join(baseline_summary.get('missing') or []) or 'none'}",
            f"- missing Exp23 predictions: {', '.join(missing_predictions) or 'none'}",
            "",
            "## Guardrails",
            "",
            "- Test split is not read.",
            "- Dev labels are used only for final dev evaluation, not for training.",
            "- D1 annotations are evaluation references only.",
            "- Raw predictions and logs should remain uncommitted.",
        ]
    )
    write_text(out_dir / "reports" / "exp23_r7_dpo_scout_report.md", "\n".join(lines))


def collect(args: argparse.Namespace) -> dict[str, Any]:
    run_rows = load_run_matrix(args.run_matrix)
    reference = sft_collect.read_csv_rows(args.reference_csv)
    metric_rows, d1_rows, failure_rows, missing_predictions = collect_run_predictions(
        run_rows,
        args.prediction_root,
        reference,
        args.d1_dir,
        args.allow_missing_predictions,
    )
    baseline_summary = include_baselines(args.baseline_dir, metric_rows, d1_rows, failure_rows)
    decision = json_safe(compare_primary(metric_rows, d1_rows))
    write_csv(args.out_dir / "tables" / "exp23_r7_dpo_scout_dev_metrics.csv", metric_rows, METRIC_FIELDS)
    write_csv(args.out_dir / "tables" / "exp23_r7_dpo_scout_d1_hidden_eval.csv", d1_rows, D1_FIELDS)
    write_csv(args.out_dir / "tables" / "exp23_r7_dpo_scout_failure_type_eval.csv", failure_rows, FAILURE_FIELDS)
    write_json(args.out_dir / "decision" / "exp23_r7_dpo_scout_decision.json", decision)
    write_report(args.out_dir, metric_rows, d1_rows, failure_rows, decision, baseline_summary, missing_predictions)
    return {
        "exp23_runs_collected": len([row for row in metric_rows if str(row.get("run_name", "")).startswith("r7")]),
        "missing_predictions": missing_predictions,
        "recommendation": decision["recommendation"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Exp23 R7 DPO scout dev predictions.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_PREDICTION_ROOT)
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--d1-dir", type=Path, default=DEFAULT_D1_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--run-matrix", type=Path, default=DEFAULT_OUT_DIR / "tables" / "exp23_run_matrix.csv")
    parser.add_argument("--allow-missing-predictions", action="store_true")
    args = parser.parse_args()
    print(json.dumps(collect(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
