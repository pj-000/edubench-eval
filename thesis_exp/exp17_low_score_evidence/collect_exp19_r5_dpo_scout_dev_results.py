"""Collect Exp19-R5 DPO scout dev predictions and compare with SFT baselines."""

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
from thesis_exp.src.edujudge.utils.io import write_csv, write_json, write_text  # noqa: E402


DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_r5_dpo_scout")
DEFAULT_PREDICTION_ROOT = DEFAULT_OUT_DIR / "dev_predictions"
DEFAULT_REFERENCE = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp19_sft_dpo_datasets_seed42/tables/exp19_dev_reference.csv"
)
DEFAULT_D1_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/d1_hidden_failure_audit_seed42_dev/"
    "summary_human_rationale_recovered"
)
DEFAULT_BASELINE_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_sft_second_round")

RUNS = [
    {
        "run_name": "r5c_from_r2c",
        "init_adapter": "r2c_clean_reason_score_balanced",
        "dpo_dataset": "r5c_score_risk",
    },
    {
        "run_name": "r5c_from_r1b",
        "init_adapter": "r1b_score_only_balanced",
        "dpo_dataset": "r5c_score_risk",
    },
    {
        "run_name": "r5d_from_r2c",
        "init_adapter": "r2c_clean_reason_score_balanced",
        "dpo_dataset": "r5d_evidence_consistency",
    },
    {
        "run_name": "r5e_from_r2c",
        "init_adapter": "r2c_clean_reason_score_balanced",
        "dpo_dataset": "r5e_hard_synthetic_control",
    },
]

BASELINES = [
    {
        "run_name": "r1b_score_only_balanced",
        "init_adapter": "r1b_score_only_balanced",
        "dpo_dataset": "none_init_baseline",
    },
    {
        "run_name": "r2c_clean_reason_score_balanced",
        "init_adapter": "r2c_clean_reason_score_balanced",
        "dpo_dataset": "none_init_baseline",
    },
    {
        "run_name": "r4b_shuffled_reason_balanced",
        "init_adapter": "r4b_shuffled_reason_balanced",
        "dpo_dataset": "none_init_baseline",
    },
]

METRIC_FIELDS = [
    "run_name",
    "init_adapter",
    "dpo_dataset",
    "n",
    "parse_success_rate",
    "MAE",
    "QWK",
    "Signed_Bias",
    "Exact_Match",
    "low_to_high_count",
    "low_to_high_rate",
    "high_to_low_count",
    "high_to_low_rate",
    "label2_recall",
    "label5_recall",
]
D1_FIELDS = [
    "run_name",
    "n_d1_cases",
    "mean_pred_d1_hidden",
    "pred_ge4_rate_d1_hidden",
    "pred_5_rate_d1_hidden",
    "label2_recall_d1",
    "matched_control_mean_pred",
    "hidden_control_score_gap",
]
FAILURE_FIELDS = [
    "run_name",
    "failure_type_micro_f1",
    "failure_type_macro_f1",
    "major_failure_nonempty_rate_on_d1_hidden",
    "no_major_failure_rate_on_controls",
    "score_cap_nonnull_rate_on_d1_hidden",
]


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


def copy_metric_fields(row: dict[str, Any], meta: dict[str, str]) -> dict[str, Any]:
    return {
        "run_name": meta["run_name"],
        "init_adapter": meta["init_adapter"],
        "dpo_dataset": meta["dpo_dataset"],
        "n": row.get("n", ""),
        "parse_success_rate": row.get("parse_success_rate", ""),
        "MAE": row.get("MAE", ""),
        "QWK": row.get("QWK", ""),
        "Signed_Bias": row.get("Signed_Bias", ""),
        "Exact_Match": row.get("Exact_Match", ""),
        "low_to_high_count": row.get("low_to_high_count", ""),
        "low_to_high_rate": row.get("low_to_high_rate", ""),
        "high_to_low_count": row.get("high_to_low_count", ""),
        "high_to_low_rate": row.get("high_to_low_rate", ""),
        "label2_recall": row.get("label2_recall", ""),
        "label5_recall": row.get("label5_recall", ""),
    }


def copy_d1_fields(row: dict[str, Any], run_name: str) -> dict[str, Any]:
    return {
        "run_name": run_name,
        "n_d1_cases": row.get("n_d1_cases", ""),
        "mean_pred_d1_hidden": row.get("mean_pred_d1_hidden", ""),
        "pred_ge4_rate_d1_hidden": row.get("pred_ge4_rate_d1_hidden", ""),
        "pred_5_rate_d1_hidden": row.get("pred_5_rate_d1_hidden", ""),
        "label2_recall_d1": row.get("label2_recall_d1", ""),
        "matched_control_mean_pred": row.get("matched_control_mean_pred", ""),
        "hidden_control_score_gap": row.get("hidden_control_score_gap", ""),
    }


def copy_failure_fields(row: dict[str, Any], run_name: str, score_cap_rate: Any = "") -> dict[str, Any]:
    return {
        "run_name": run_name,
        "failure_type_micro_f1": row.get("failure_type_micro_f1", ""),
        "failure_type_macro_f1": row.get("failure_type_macro_f1", ""),
        "major_failure_nonempty_rate_on_d1_hidden": row.get("major_failure_nonempty_rate_on_d1_hidden", ""),
        "no_major_failure_rate_on_controls": row.get("no_major_failure_rate_on_controls", ""),
        "score_cap_nonnull_rate_on_d1_hidden": score_cap_rate,
    }


def d1_score_cap_rate(parsed_rows: list[dict[str, Any]], d1_dir: Path) -> float:
    annotations, _pairs, _controls = sft_collect.load_d1_annotations(d1_dir)
    hidden_ids = set(annotations)
    hidden_rows = [row for row in parsed_rows if str(row.get("sample_id", "")) in hidden_ids]
    return sft_collect.safe_rate(sum(1 for row in hidden_rows if row.get("score_cap") is not None), len(hidden_rows))


def collect_dpo_rows(
    prediction_root: Path,
    reference: list[dict[str, str]],
    d1_dir: Path,
    allow_missing_predictions: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    metric_rows: list[dict[str, Any]] = []
    d1_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for meta in RUNS:
        run_name = meta["run_name"]
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
        metric = sft_collect.metric_summary(parsed_rows, run_name, run_name, "dev")
        metric_rows.append(copy_metric_fields(metric, meta))
        d1 = sft_collect.d1_eval_row(parsed_rows, d1_dir, run_name, run_name)
        failure = sft_collect.failure_type_eval_row(parsed_rows, d1_dir, run_name, run_name)
        d1_rows.append(copy_d1_fields(d1, run_name))
        failure_rows.append(copy_failure_fields(failure, run_name, d1_score_cap_rate(parsed_rows, d1_dir)))
    return metric_rows, d1_rows, failure_rows, missing


def include_baselines(
    baseline_dir: Path,
    metric_rows: list[dict[str, Any]],
    d1_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_metrics = {
        row.get("run_name", ""): row
        for row in read_csv_if_exists(baseline_dir / "tables" / "exp19_sft_second_round_dev_metrics.csv")
    }
    baseline_d1 = {
        row.get("run_name", ""): row
        for row in read_csv_if_exists(baseline_dir / "tables" / "exp19_sft_second_round_dev_d1_hidden_eval.csv")
    }
    baseline_failure = {
        row.get("run_name", ""): row
        for row in read_csv_if_exists(baseline_dir / "tables" / "exp19_sft_second_round_dev_failure_type_eval.csv")
    }
    included: list[str] = []
    missing: list[str] = []
    for meta in BASELINES:
        run_name = meta["run_name"]
        if run_name not in baseline_metrics:
            missing.append(run_name)
            continue
        metric_rows.append(copy_metric_fields(baseline_metrics[run_name], meta))
        if run_name in baseline_d1:
            d1_rows.append(copy_d1_fields(baseline_d1[run_name], run_name))
        if run_name in baseline_failure:
            failure_rows.append(copy_failure_fields(baseline_failure[run_name], run_name))
        included.append(run_name)
    return {"included": included, "missing": missing}


def by_run(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("run_name", "")): row for row in rows}


def r5c_success(dpo: dict[str, Any] | None, base: dict[str, Any] | None) -> dict[str, Any]:
    if not dpo or not base:
        return {"success": False, "reason": "missing DPO or baseline row"}
    low_to_high_improvement = to_float(base.get("low_to_high_rate")) - to_float(dpo.get("low_to_high_rate"))
    label2_delta = to_float(dpo.get("label2_recall")) - to_float(base.get("label2_recall"))
    high_to_low_delta = to_float(dpo.get("high_to_low_rate")) - to_float(base.get("high_to_low_rate"))
    mae_delta = to_float(dpo.get("MAE")) - to_float(base.get("MAE"))
    qwk_delta = to_float(dpo.get("QWK")) - to_float(base.get("QWK"))
    success = (
        low_to_high_improvement >= 0.05
        and label2_delta > 0
        and high_to_low_delta <= 0.05
        and mae_delta <= 0.05
        and qwk_delta >= -0.05
    )
    return {
        "success": success,
        "low_to_high_improvement": low_to_high_improvement,
        "label2_recall_delta": label2_delta,
        "high_to_low_delta": high_to_low_delta,
        "MAE_delta": mae_delta,
        "QWK_delta": qwk_delta,
    }


def r5d_success(
    r5d_metric: dict[str, Any] | None,
    r5d_failure: dict[str, Any] | None,
    base_metric: dict[str, Any] | None,
    base_failure: dict[str, Any] | None,
) -> dict[str, Any]:
    if not r5d_metric or not r5d_failure or not base_metric or not base_failure:
        return {"success": False, "reason": "missing DPO or baseline row"}
    nonempty_delta = to_float(r5d_failure.get("major_failure_nonempty_rate_on_d1_hidden")) - to_float(
        base_failure.get("major_failure_nonempty_rate_on_d1_hidden")
    )
    f1_delta = to_float(r5d_failure.get("failure_type_micro_f1")) - to_float(base_failure.get("failure_type_micro_f1"))
    mae_delta = to_float(r5d_metric.get("MAE")) - to_float(base_metric.get("MAE"))
    high_to_low_delta = to_float(r5d_metric.get("high_to_low_rate")) - to_float(base_metric.get("high_to_low_rate"))
    success = nonempty_delta > 0 and f1_delta > 0 and mae_delta <= 0.10 and high_to_low_delta <= 0.10
    return {
        "success": success,
        "major_failure_nonempty_delta": nonempty_delta,
        "failure_type_micro_f1_delta": f1_delta,
        "MAE_delta": mae_delta,
        "high_to_low_delta": high_to_low_delta,
    }


def similar_control(r5e: dict[str, Any] | None, r5c: dict[str, Any] | None) -> dict[str, Any]:
    if not r5e or not r5c:
        return {"similar": False, "reason": "missing R5E or R5C row"}
    low_to_high_gap = abs(to_float(r5e.get("low_to_high_rate")) - to_float(r5c.get("low_to_high_rate")))
    label2_gap = abs(to_float(r5e.get("label2_recall")) - to_float(r5c.get("label2_recall")))
    mae_gap = abs(to_float(r5e.get("MAE")) - to_float(r5c.get("MAE")))
    return {
        "similar": low_to_high_gap <= 0.02 and label2_gap <= 0.05 and mae_gap <= 0.05,
        "low_to_high_gap": low_to_high_gap,
        "label2_recall_gap": label2_gap,
        "MAE_gap": mae_gap,
    }


def make_decision(metric_rows: list[dict[str, Any]], failure_rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = by_run(metric_rows)
    failures = by_run(failure_rows)
    r5c_r2c = r5c_success(metrics.get("r5c_from_r2c"), metrics.get("r2c_clean_reason_score_balanced"))
    r5c_r1b = r5c_success(metrics.get("r5c_from_r1b"), metrics.get("r1b_score_only_balanced"))
    r5d = r5d_success(
        metrics.get("r5d_from_r2c"),
        failures.get("r5d_from_r2c"),
        metrics.get("r2c_clean_reason_score_balanced"),
        failures.get("r2c_clean_reason_score_balanced"),
    )
    r5e_vs_r5c = similar_control(metrics.get("r5e_from_r2c"), metrics.get("r5c_from_r2c"))

    if r5c_r2c.get("success") and not r5e_vs_r5c.get("similar"):
        decision = "proceed_full_r5c_from_r2c"
        reason = "R5C from R2c satisfies score-risk criteria and is not matched by the hard-synthetic control."
    elif r5c_r1b.get("success"):
        decision = "proceed_full_r5c_from_r1b"
        reason = "R5C from R1b satisfies score-risk criteria."
    elif r5d.get("success"):
        decision = "proceed_full_r5d"
        reason = "R5D improves structured failure behavior without obvious score collapse."
    elif r5c_r2c.get("low_to_high_improvement", 0) > 0 or r5c_r1b.get("low_to_high_improvement", 0) > 0:
        decision = "need_more_rejection_mining"
        reason = "There is some risk-side movement, but the scout does not meet the full success rule."
    else:
        decision = "stop_dpo"
        reason = "No DPO scout improves the required risk metrics against its init baseline."

    return {
        "decision": decision,
        "reason": reason,
        "r5c_from_r2c": r5c_r2c,
        "r5c_from_r1b": r5c_r1b,
        "r5d_from_r2c": r5d,
        "r5e_vs_r5c_from_r2c": r5e_vs_r5c,
        "guardrails": {
            "no_test_read": True,
            "d1_used_for_eval_only": True,
            "human_rationale_not_in_prompt": True,
        },
    }


def answer_line(question: str, answer: str) -> str:
    return f"- {question}: {answer}"


def bool_answer(condition: bool, yes: str, no: str) -> str:
    return yes if condition else no


def write_report(
    out_dir: Path,
    metric_rows: list[dict[str, Any]],
    d1_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    decision: dict[str, Any],
    baseline_summary: dict[str, Any],
    missing_predictions: list[str],
) -> None:
    metrics = by_run(metric_rows)
    failures = by_run(failure_rows)
    r5c_r2c = decision["r5c_from_r2c"]
    r5c_r1b = decision["r5c_from_r1b"]
    r5d = decision["r5d_from_r2c"]
    r5e = decision["r5e_vs_r5c_from_r2c"]
    high_damage = any(
        to_float(row.get("high_to_low_rate")) - to_float(metrics.get(row.get("init_adapter", ""), {}).get("high_to_low_rate"))
        > 0.05
        for row in metric_rows
        if str(row.get("run_name", "")).startswith("r5")
    )
    lines = [
        "# Exp19-R5 DPO Scout Dev Evaluation",
        "",
        "This report summarizes small-step DPO scout adapters on the original question-disjoint dev split.",
        "Raw predictions/logs/checkpoints remain gitignored. No test split is read.",
        "",
        "## Dev Metrics",
        "",
        "| run | init | dataset | MAE | QWK | low-to-high | high-to-low | label2 recall |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            f"| `{row['run_name']}` | `{row['init_adapter']}` | `{row['dpo_dataset']}` | "
            f"{fmt(row['MAE'])} | {fmt(row['QWK'])} | {fmt(row['low_to_high_rate'])} | "
            f"{fmt(row['high_to_low_rate'])} | {fmt(row['label2_recall'])} |"
        )
    lines.extend(
        [
            "",
            "## Required Questions",
            "",
            answer_line(
                "Does R5C from R2c reduce low-to-high vs R2c",
                bool_answer(
                    r5c_r2c.get("low_to_high_improvement", 0) > 0,
                    f"yes, delta={fmt(r5c_r2c.get('low_to_high_improvement'))}",
                    f"no, delta={fmt(r5c_r2c.get('low_to_high_improvement'))}",
                ),
            ),
            answer_line(
                "Does R5C from R1b reduce low-to-high vs R1b",
                bool_answer(
                    r5c_r1b.get("low_to_high_improvement", 0) > 0,
                    f"yes, delta={fmt(r5c_r1b.get('low_to_high_improvement'))}",
                    f"no, delta={fmt(r5c_r1b.get('low_to_high_improvement'))}",
                ),
            ),
            answer_line(
                "Does R5D improve structured failure outputs",
                bool_answer(
                    r5d.get("major_failure_nonempty_delta", 0) > 0 or r5d.get("failure_type_micro_f1_delta", 0) > 0,
                    (
                        "partly/yes, "
                        f"nonempty_delta={fmt(r5d.get('major_failure_nonempty_delta'))}, "
                        f"micro_f1_delta={fmt(r5d.get('failure_type_micro_f1_delta'))}"
                    ),
                    (
                        "no, "
                        f"nonempty_delta={fmt(r5d.get('major_failure_nonempty_delta'))}, "
                        f"micro_f1_delta={fmt(r5d.get('failure_type_micro_f1_delta'))}"
                    ),
                ),
            ),
            answer_line(
                "Does R5E hard-synthetic perform similarly to R5C",
                bool_answer(
                    bool(r5e.get("similar")),
                    (
                        "yes; if R5C is not clearly better than R5E, template effects remain a concern "
                        f"(low_to_high_gap={fmt(r5e.get('low_to_high_gap'))})"
                    ),
                    f"no; low_to_high_gap={fmt(r5e.get('low_to_high_gap'))}",
                ),
            ),
            answer_line(
                "Does any DPO run damage high-score protection",
                bool_answer(high_damage, "yes; inspect high-to-low rates before full training", "not by the >0.05 rule"),
            ),
            answer_line("Should we run full DPO", decision["decision"]),
            answer_line(
                "Which dataset/init pair is most promising",
                decision["decision"].replace("proceed_full_", "") if decision["decision"].startswith("proceed_full_") else "none locked yet",
            ),
            "",
            "## D1 Hidden And Failure-Type Tables",
            "",
            "| run | D1 pred>=4 | D1 label2 recall | failure micro-F1 | D1 nonempty failure | score_cap nonnull |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    d1_by_name = by_run(d1_rows)
    for row in failure_rows:
        name = str(row["run_name"])
        d1 = d1_by_name.get(name, {})
        lines.append(
            f"| `{name}` | {fmt(d1.get('pred_ge4_rate_d1_hidden'))} | {fmt(d1.get('label2_recall_d1'))} | "
            f"{fmt(row.get('failure_type_micro_f1'))} | {fmt(row.get('major_failure_nonempty_rate_on_d1_hidden'))} | "
            f"{fmt(row.get('score_cap_nonnull_rate_on_d1_hidden'))} |"
        )
    lines.extend(
        [
            "",
            "## Baselines",
            "",
            f"- included baselines: {', '.join(baseline_summary.get('included') or []) or 'none'}",
            f"- missing baselines: {', '.join(baseline_summary.get('missing') or []) or 'none'}",
            f"- missing DPO prediction runs: {', '.join(missing_predictions) or 'none'}",
            "",
            "## Decision",
            "",
            f"- decision: `{decision['decision']}`",
            f"- reason: {decision['reason']}",
            "",
            "## Guardrails",
            "",
            "- Evaluation uses the original dev split, not balanced train distribution.",
            "- Test split is not read.",
            "- D1 annotations are evaluation references only.",
            "- Human rationale is not included in the prediction prompt.",
        ]
    )
    write_text(out_dir / "reports" / "exp19_r5_dpo_scout_report.md", "\n".join(lines))


def collect(args: argparse.Namespace) -> dict[str, Any]:
    reference = sft_collect.read_csv_rows(args.reference_csv)
    metric_rows, d1_rows, failure_rows, missing_predictions = collect_dpo_rows(
        args.prediction_root,
        reference,
        args.d1_dir,
        args.allow_missing_predictions,
    )
    baseline_summary = include_baselines(args.baseline_dir, metric_rows, d1_rows, failure_rows)
    decision = make_decision(metric_rows, failure_rows)
    write_csv(args.out_dir / "tables" / "exp19_r5_dpo_scout_dev_metrics.csv", metric_rows, METRIC_FIELDS)
    write_csv(args.out_dir / "tables" / "exp19_r5_dpo_scout_d1_hidden_eval.csv", d1_rows, D1_FIELDS)
    write_csv(args.out_dir / "tables" / "exp19_r5_dpo_scout_failure_type_eval.csv", failure_rows, FAILURE_FIELDS)
    write_json(args.out_dir / "decision" / "exp19_r5_dpo_scout_decision.json", decision)
    write_report(args.out_dir, metric_rows, d1_rows, failure_rows, decision, baseline_summary, missing_predictions)
    return {
        "dpo_runs_collected": len([row for row in metric_rows if str(row.get("run_name", "")).startswith("r5")]),
        "baselines_included": baseline_summary.get("included", []),
        "missing_predictions": missing_predictions,
        "decision": decision["decision"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Exp19-R5 DPO scout dev predictions.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_PREDICTION_ROOT)
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--d1-dir", type=Path, default=DEFAULT_D1_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--allow-missing-predictions", action="store_true")
    args = parser.parse_args()
    summary = collect(args)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
