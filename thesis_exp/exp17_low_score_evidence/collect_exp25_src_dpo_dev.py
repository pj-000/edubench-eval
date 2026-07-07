"""Collect Exp25 Structured SRC-DPO dev predictions."""

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
from thesis_exp.src.edujudge.utils.io import read_csv, write_csv, write_json, write_text  # noqa: E402


DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp25_structured_src_dpo_seed42")
DEFAULT_PREDICTION_ROOT = DEFAULT_OUT_DIR / "dev_predictions"
DEFAULT_REFERENCE = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp19_sft_dpo_datasets_seed42/tables/exp19_dev_reference.csv"
)
DEFAULT_D1_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/d1_hidden_failure_audit_seed42_dev/"
    "summary_human_rationale_recovered"
)
DEFAULT_SFT_PREDICTION_ROOT = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_sft_second_round/dev_predictions")
DEFAULT_EXP23_PREDICTION_ROOT = Path("thesis_exp/exp17_low_score_evidence/outputs/exp23_r7_dpo_scout/dev_predictions")
DEFAULT_EXP24_PREDICTION_ROOT = Path("thesis_exp/exp17_low_score_evidence/outputs/exp24_orc_dpo_seed42/dev_predictions")
DEFAULT_TRAINING_SUMMARY_DIR = DEFAULT_OUT_DIR / "training_summaries"

RUNS = [
    {
        "run_name": "exp25_src_score_mismatch_r2c",
        "run_label": "Exp25 SRC score-mismatch R2C",
        "init_adapter": "r2c_clean_reason_score_balanced",
        "dataset": "r7h_score_mismatch_only",
        "method_role": "same_schema_score_mismatch",
    },
    {
        "run_name": "exp25_src_mixed_r2c",
        "run_label": "Exp25 SRC mixed R2C",
        "init_adapter": "r2c_clean_reason_score_balanced",
        "dataset": "r7h_structured_mixed",
        "method_role": "score_reason_failure_consistency",
    },
    {
        "run_name": "exp25_src_mixed_r4b",
        "run_label": "Exp25 SRC mixed R4B",
        "init_adapter": "r4b_shuffled_reason_balanced",
        "dataset": "r7h_structured_mixed",
        "method_role": "optional_r4b_init",
    },
    {
        "run_name": "exp25_r4_field_b1_ftx0_mixed_r2c",
        "run_label": "Exp25R4 field SRC mixed beta1 R2C",
        "init_adapter": "r2c_clean_reason_score_balanced",
        "dataset": "r7h_structured_mixed",
        "method_role": "field_masked_src_dpo_beta1",
    },
    {
        "run_name": "exp25_r4_field_b3_ftx0_mixed_r2c",
        "run_label": "Exp25R4 field SRC mixed beta3 R2C",
        "init_adapter": "r2c_clean_reason_score_balanced",
        "dataset": "r7h_structured_mixed",
        "method_role": "field_masked_src_dpo_beta3",
    },
    {
        "run_name": "exp25_r4_field_b1_ftx0_score_r2c",
        "run_label": "Exp25R4 field SRC score-only beta1 R2C",
        "init_adapter": "r2c_clean_reason_score_balanced",
        "dataset": "r7h_score_mismatch_only",
        "method_role": "field_masked_score_mismatch_beta1_optional",
    },
]


def parse_run_names(raw: str) -> list[str]:
    return [item for item in raw.replace(",", " ").split() if item]


def select_runs(raw_run_names: str) -> list[dict[str, str]]:
    names = parse_run_names(raw_run_names)
    if not names:
        return RUNS
    by_name = {run["run_name"]: run for run in RUNS}
    unknown = sorted(set(names) - set(by_name))
    if unknown:
        raise ValueError(f"Unknown Exp25 run name(s): {', '.join(unknown)}")
    return [by_name[name] for name in names]


BASELINE_SPECS = [
    ("r2c_clean_reason_score_balanced", DEFAULT_SFT_PREDICTION_ROOT / "r2c_clean_reason_score_balanced"),
    ("r4b_shuffled_reason_balanced", DEFAULT_SFT_PREDICTION_ROOT / "r4b_shuffled_reason_balanced"),
    ("r7d_reason_real_s100_b0p03_lr5em6", DEFAULT_EXP23_PREDICTION_ROOT / "r7d_reason_real_s100_b0p03_lr5em6"),
    ("r7f_score_reason_consistency_s100_b0p03_lr5em6", DEFAULT_EXP23_PREDICTION_ROOT / "r7f_score_reason_consistency_s100_b0p03_lr5em6"),
    ("exp24_dpo0_r2c", DEFAULT_EXP24_PREDICTION_ROOT / "exp24_dpo0_r2c"),
    ("exp24_orc_b_r2c", DEFAULT_EXP24_PREDICTION_ROOT / "exp24_orc_b_r2c"),
]


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


def safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        out = int(float(value))
        return out if 1 <= out <= 5 else None
    except Exception:
        return None


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def by_run(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("run_name", "")): row for row in rows}


def load_aligned(run_name: str, run_dir: Path, reference: list[dict[str, str]]) -> list[dict[str, Any]]:
    prediction_file = sft_collect.find_prediction_file(run_dir)
    records = sft_collect.load_prediction_records(prediction_file)
    return sft_collect.align_predictions(reference, records, run_name, run_name)


def invalid_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if safe_int(row.get("pred_label")) is None or not row.get("parse_success"))


def subset_extra_rows(
    run_name: str,
    rows: list[dict[str, Any]],
    d1_ids: set[str],
    baseline_predictions: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    low = [row for row in rows if int(row["gold_label"]) <= 2]
    d1 = [row for row in rows if row["sample_id"] in d1_ids]
    pred_by_id = {row["sample_id"]: row for row in rows}

    def baseline_decrease_count(base_name: str) -> int:
        base_by_id = baseline_predictions.get(base_name, {})
        count = 0
        for sid in d1_ids:
            cur = safe_int(pred_by_id.get(sid, {}).get("pred_label"))
            base = safe_int(base_by_id.get(sid, {}).get("pred_label"))
            if cur is not None and base is not None and cur < base:
                count += 1
        return count

    return {
        "run_name": run_name,
        "invalid_count": invalid_count(rows),
        "invalid_count_on_low": invalid_count(low),
        "invalid_count_on_d1_hidden": invalid_count(d1),
        "score_cap_nonnull_rate_on_low": sft_collect.safe_rate(
            sum(1 for row in low if row.get("score_cap") is not None), len(low)
        ),
        "score_cap_nonnull_rate_on_d1": sft_collect.safe_rate(
            sum(1 for row in d1 if row.get("score_cap") is not None), len(d1)
        ),
        "major_failure_nonempty_rate_on_low": sft_collect.safe_rate(
            sum(1 for row in low if sft_collect.has_substantive_failure(row)), len(low)
        ),
        "major_failure_nonempty_rate_on_d1": sft_collect.safe_rate(
            sum(1 for row in d1 if sft_collect.has_substantive_failure(row)), len(d1)
        ),
        "no_major_failure_rate_on_low": sft_collect.safe_rate(
            sum(1 for row in low if sft_collect.has_no_major_failure(row)), len(low)
        ),
        "d1_actual_scalar_decrease_vs_r2c": baseline_decrease_count("r2c_clean_reason_score_balanced"),
        "d1_actual_scalar_decrease_vs_exp24_dpo0": baseline_decrease_count("exp24_dpo0_r2c"),
    }


def transition_rows(
    run_name: str,
    rows: list[dict[str, Any]],
    baseline_predictions: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    pred_by_id = {row["sample_id"]: row for row in rows}
    out: list[dict[str, Any]] = []
    for base_name, base_by_id in baseline_predictions.items():
        n = changed = low_fixed = low_worsened = high_added = high_reduced = 0
        for sid, cur_row in pred_by_id.items():
            base_row = base_by_id.get(sid)
            if not base_row:
                continue
            gold = int(cur_row["gold_label"])
            cur = safe_int(cur_row.get("pred_label"))
            base = safe_int(base_row.get("pred_label"))
            if cur is None or base is None:
                continue
            n += 1
            changed += int(cur != base)
            low_fixed += int(gold <= 2 and base >= 4 and cur < 4)
            low_worsened += int(gold <= 2 and base < 4 and cur >= 4)
            high_added += int(gold >= 4 and base > 2 and cur <= 2)
            high_reduced += int(gold >= 4 and base <= 2 and cur > 2)
        out.append(
            {
                "run_name": run_name,
                "baseline_run": base_name,
                "n": n,
                "changed_count": changed,
                "changed_rate": sft_collect.safe_rate(changed, n),
                "low_to_high_fixed_count": low_fixed,
                "low_to_high_worsened_count": low_worsened,
                "high_to_low_added_count": high_added,
                "high_to_low_reduced_count": high_reduced,
            }
        )
    return out


def load_training_summaries(summary_dir: Path, runs: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for run in runs:
        path = summary_dir / f"{run['run_name']}.json"
        if not path.exists():
            out.append({"run_name": run["run_name"], "completed": False})
            continue
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        last = data.get("last_metrics") or {}
        out.append(
            {
                "run_name": data.get("run_name", run["run_name"]),
                "completed": int(data.get("completed_steps", 0)) >= int(data.get("max_steps", 0)),
                "output_dir": data.get("output_dir", ""),
                "data": data.get("data", ""),
                "rows": data.get("rows", ""),
                "max_steps": data.get("max_steps", ""),
                "completed_steps": data.get("completed_steps", ""),
                "learning_rate": data.get("learning_rate", ""),
                "beta": data.get("beta", ""),
                "margin": data.get("margin", ""),
                "pref_ftx": data.get("pref_ftx", ""),
                "logp_mode": data.get("logp_mode", ""),
                "length_normalized_logp": data.get("length_normalized_logp", ""),
                "initial_mean_delta_step1": data.get("initial_mean_delta_step1", ""),
                "mean_weight_all_pairs": data.get("mean_weight_all_pairs", ""),
                "max_weight_all_pairs": data.get("max_weight_all_pairs", ""),
                "cuda_device_name": data.get("cuda_device_name", ""),
                "cuda_memory_allocated_peak_mb": data.get("cuda_memory_allocated_peak_mb", ""),
                "last_loss": last.get("loss", ""),
                "last_src_loss": last.get("src_loss", ""),
                "last_chosen_nll": last.get("chosen_nll", ""),
                "last_mean_delta": last.get("mean_delta", ""),
                "elapsed_seconds": data.get("elapsed_seconds", ""),
            }
        )
    return out


def make_decision(
    metric_rows: list[dict[str, Any]],
    extra_rows: list[dict[str, Any]],
    d1_rows: list[dict[str, Any]],
    runs: list[dict[str, str]],
) -> dict[str, Any]:
    metrics = by_run(metric_rows)
    extra = by_run(extra_rows)
    d1_by_name = by_run(d1_rows)
    evaluations: dict[str, Any] = {}
    for run in runs:
        name = run["run_name"]
        if name not in metrics:
            continue
        metric = metrics[name]
        ext = extra.get(name, {})
        d1 = d1_by_name.get(name, {})
        minimum_checks = {
            "low_to_high_rate": to_float(metric.get("low_to_high_rate")) <= 0.8298,
            "d1_decrease_vs_exp24_dpo0": to_float(ext.get("d1_actual_scalar_decrease_vs_exp24_dpo0")) >= 3,
            "label2_recall": to_float(metric.get("label2_recall")) > 0.0526,
            "MAE": to_float(metric.get("MAE")) <= 0.50,
            "QWK": to_float(metric.get("QWK")) >= 0.35,
            "label5_recall": to_float(metric.get("label5_recall")) >= 0.78,
            "invalid_count_on_d1_hidden": to_float(ext.get("invalid_count_on_d1_hidden")) == 0,
        }
        strong_checks = {
            "low_to_high_rate": to_float(metric.get("low_to_high_rate")) <= 0.5965,
            "MAE": to_float(metric.get("MAE")) <= 0.45,
            "QWK": to_float(metric.get("QWK")) >= 0.48,
            "label2_recall": to_float(metric.get("label2_recall")) > 0.10,
            "label5_recall": to_float(metric.get("label5_recall")) >= 0.80,
            "d1_pred_ge4": to_float(d1.get("pred_ge4_rate_d1_hidden")) <= 0.8462,
            "d1_decrease_vs_r2c": to_float(ext.get("d1_actual_scalar_decrease_vs_r2c")) >= 6,
        }
        evaluations[name] = {
            "minimum_success": all(minimum_checks.values()),
            "strong_success": all(strong_checks.values()),
            "minimum_checks": minimum_checks,
            "strong_checks": strong_checks,
        }
    minimum = [name for name, row in evaluations.items() if row["minimum_success"]]
    strong = [name for name, row in evaluations.items() if row["strong_success"]]
    if strong:
        recommendation = "src_dpo_strong_success_consider_orc_weighted_src_or_multiseed"
        reason = f"{strong[0]} satisfies strong SRC-DPO success."
    elif minimum:
        recommendation = "src_dpo_minimum_success_continue"
        reason = f"{minimum[0]} satisfies minimum SRC-DPO success."
    elif evaluations:
        recommendation = "src_dpo_not_yet_successful_consider_hidden_failure_expansion"
        reason = "No Exp25 SRC-DPO run satisfies minimum success."
    else:
        recommendation = "wait_for_exp25_predictions"
        reason = "No Exp25 predictions were collected."
    return {
        "recommendation": recommendation,
        "reason": reason,
        "minimum_success_runs": minimum,
        "strong_success_runs": strong,
        "evaluations": json_safe(evaluations),
        "guardrails": {
            "test_read": False,
            "test_label_read": False,
            "dev_used_for_eval_only": True,
            "human_reason_not_in_prompt": True,
            "raw_predictions_committed": False,
        },
    }


def write_report(
    out_dir: Path,
    metric_rows: list[dict[str, Any]],
    d1_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    extra_rows: list[dict[str, Any]],
    transition: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
    decision: dict[str, Any],
    missing_predictions: list[str],
) -> None:
    d1_by_name = by_run(d1_rows)
    failure_by_name = by_run(failure_rows)
    extra_by_name = by_run(extra_rows)
    lines = [
        "# Exp25 Structured SRC-DPO Dev Evaluation",
        "",
        "Exp25 tests same-schema reason/score consistency preferences after Exp24 score-channel ORC-DPO failed to move hidden low-score predictions.",
        "",
        "## Dev Metrics",
        "",
        "| run | parse | MAE | QWK | low-to-high | label2 recall | label5 recall | D1 pred>=4 | D1 decrease vs DPO0 | invalid D1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        name = str(row["run_name"])
        if not name.startswith("exp25_"):
            continue
        d1 = d1_by_name.get(name, {})
        ext = extra_by_name.get(name, {})
        lines.append(
            f"| `{name}` | {fmt(row.get('parse_success_rate'))} | {fmt(row.get('MAE'))} | "
            f"{fmt(row.get('QWK'))} | {fmt(row.get('low_to_high_rate'))} | "
            f"{fmt(row.get('label2_recall'))} | {fmt(row.get('label5_recall'))} | "
            f"{fmt(d1.get('pred_ge4_rate_d1_hidden'))} | "
            f"{ext.get('d1_actual_scalar_decrease_vs_exp24_dpo0', '')} | "
            f"{ext.get('invalid_count_on_d1_hidden', '')} |"
        )
    lines.extend(["", "## Structured Behavior", "", "| run | D1 failure nonempty | D1 score_cap nonnull | low score_cap nonnull | no_major_failure on low |", "|---|---:|---:|---:|---:|"])
    for row in metric_rows:
        name = str(row["run_name"])
        if not name.startswith("exp25_"):
            continue
        failure = failure_by_name.get(name, {})
        ext = extra_by_name.get(name, {})
        lines.append(
            f"| `{name}` | {fmt(failure.get('major_failure_nonempty_rate_on_d1_hidden'))} | "
            f"{fmt(ext.get('score_cap_nonnull_rate_on_d1'))} | "
            f"{fmt(ext.get('score_cap_nonnull_rate_on_low'))} | "
            f"{fmt(ext.get('no_major_failure_rate_on_low'))} |"
        )
    lines.extend(["", "## Transition Diagnosis", ""])
    for row in transition:
        if not str(row.get("run_name", "")).startswith("exp25_"):
            continue
        lines.append(
            f"- `{row['run_name']}` vs `{row['baseline_run']}`: changed={row['changed_count']} "
            f"({fmt(row['changed_rate'])}), low_fixed={row['low_to_high_fixed_count']}, "
            f"low_worsened={row['low_to_high_worsened_count']}, high_added={row['high_to_low_added_count']}."
        )
    lines.extend(["", "## Training Summary", "", "| run | completed | steps | beta | pref_ftx | mean weight | peak MB | last loss |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in training_rows:
        lines.append(
            f"| `{row.get('run_name', '')}` | {row.get('completed', '')} | "
            f"{row.get('completed_steps', '')}/{row.get('max_steps', '')} | "
            f"{row.get('beta', '')} | {row.get('pref_ftx', '')} | "
            f"{fmt(row.get('mean_weight_all_pairs'))} | {fmt(row.get('cuda_memory_allocated_peak_mb'))} | "
            f"{fmt(row.get('last_loss'))} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- recommendation: `{decision['recommendation']}`",
            f"- reason: {decision['reason']}",
            f"- minimum_success_runs: {', '.join(decision.get('minimum_success_runs') or []) or 'none'}",
            f"- strong_success_runs: {', '.join(decision.get('strong_success_runs') or []) or 'none'}",
            f"- missing predictions: {', '.join(missing_predictions) or 'none'}",
            "",
            "## Guardrails",
            "",
            "- No test split is read in this collector.",
            "- Dev labels are used only for evaluation.",
            "- Human rationale is not included in prediction prompts.",
            "- Raw predictions, logs, checkpoints, and adapter weights must not be committed.",
        ]
    )
    write_text(out_dir / "reports" / "exp25_structured_src_dpo_report.md", "\n".join(lines))


def collect(args: argparse.Namespace) -> dict[str, Any]:
    runs = select_runs(args.run_names)
    reference = sft_collect.read_csv_rows(args.reference_csv)
    annotations, _pairs, _controls = sft_collect.load_d1_annotations(args.d1_dir)
    d1_ids = set(annotations)
    baseline_predictions: dict[str, dict[str, dict[str, Any]]] = {}
    missing_predictions: list[str] = []
    for name, path in BASELINE_SPECS:
        try:
            aligned = load_aligned(name, path, reference)
        except FileNotFoundError:
            missing_predictions.append(name)
            if not args.allow_missing_predictions:
                raise
            continue
        baseline_predictions[name] = {row["sample_id"]: row for row in aligned}

    metric_rows: list[dict[str, Any]] = []
    d1_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    extra_rows: list[dict[str, Any]] = []
    transition: list[dict[str, Any]] = []
    for run in runs:
        name = run["run_name"]
        try:
            aligned = load_aligned(name, args.prediction_root / name, reference)
        except FileNotFoundError:
            missing_predictions.append(name)
            if args.allow_missing_predictions:
                continue
            raise
        meta = {
            "run_name": name,
            "init_adapter": run["init_adapter"],
            "dpo_dataset": run["dataset"],
            "method_role": run["method_role"],
        }
        metric_rows.append(copy_metric_fields(sft_collect.metric_summary(aligned, name, run["run_label"], "dev"), meta))
        d1_rows.append(copy_d1_fields(sft_collect.d1_eval_row(aligned, args.d1_dir, name, run["run_label"]), name))
        failure_rows.append(
            copy_failure_fields(
                sft_collect.failure_type_eval_row(aligned, args.d1_dir, name, run["run_label"]),
                name,
                d1_score_cap_rate(aligned, args.d1_dir),
            )
        )
        extra_rows.append(subset_extra_rows(name, aligned, d1_ids, baseline_predictions))
        transition.extend(transition_rows(name, aligned, baseline_predictions))

    training_rows = load_training_summaries(args.training_summary_dir, runs)
    decision = json_safe(make_decision(metric_rows, extra_rows, d1_rows, runs))
    write_csv(args.out_dir / "tables" / "exp25_src_dpo_dev_metrics.csv", metric_rows, METRIC_FIELDS)
    write_csv(args.out_dir / "tables" / "exp25_src_dpo_d1_hidden_eval.csv", d1_rows, D1_FIELDS)
    write_csv(args.out_dir / "tables" / "exp25_src_dpo_failure_type_eval.csv", failure_rows, FAILURE_FIELDS)
    write_csv(args.out_dir / "tables" / "exp25_src_dpo_extra_behavior.csv", extra_rows)
    write_csv(args.out_dir / "tables" / "exp25_src_dpo_transition_diagnosis.csv", transition)
    write_csv(args.out_dir / "tables" / "exp25_src_dpo_training_summary.csv", training_rows)
    write_json(args.out_dir / "decision" / "exp25_structured_src_dpo_decision.json", decision)
    write_report(args.out_dir, metric_rows, d1_rows, failure_rows, extra_rows, transition, training_rows, decision, missing_predictions)
    return {
        "exp25_runs_collected": len(metric_rows),
        "selected_runs": [run["run_name"] for run in runs],
        "missing_predictions": missing_predictions,
        "recommendation": decision["recommendation"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Exp25 Structured SRC-DPO dev predictions.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_PREDICTION_ROOT)
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--d1-dir", type=Path, default=DEFAULT_D1_DIR)
    parser.add_argument("--training-summary-dir", type=Path, default=DEFAULT_TRAINING_SUMMARY_DIR)
    parser.add_argument(
        "--run-names",
        default="",
        help="Optional space/comma separated Exp25 run names to collect. Defaults to all registered Exp25 runs.",
    )
    parser.add_argument("--allow-missing-predictions", action="store_true")
    args = parser.parse_args()
    print(json.dumps(collect(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
