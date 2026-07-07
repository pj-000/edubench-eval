"""Aggregate Exp25R3 loss-scale and field-mask sanity results."""

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

from thesis_exp.src.edujudge.utils.io import read_csv, write_csv, write_json, write_text  # noqa: E402


DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp25r3_loss_scale_sanity_seed42")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


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


def flatten_summary(row: dict[str, Any]) -> dict[str, Any]:
    last = row.get("last_metrics") or {}
    return {
        "config_name": row.get("config_name", ""),
        "rows": row.get("rows", ""),
        "logp_mode": row.get("logp_mode", ""),
        "beta": row.get("beta", ""),
        "margin": row.get("margin", ""),
        "pref_ftx": row.get("pref_ftx", ""),
        "max_steps": row.get("max_steps", ""),
        "before_dpo_pref_acc": row.get("before_dpo_pref_acc", ""),
        "after_dpo_pref_acc": row.get("after_dpo_pref_acc", ""),
        "acc_gain": row.get("acc_gain", ""),
        "before_mean_delta": row.get("before_mean_delta", ""),
        "after_mean_delta": row.get("after_mean_delta", ""),
        "mean_delta_gain": row.get("mean_delta_gain", ""),
        "first_batch_delta_before": row.get("first_batch_delta_before", ""),
        "first_batch_delta_after": row.get("first_batch_delta_after", ""),
        "first_batch_delta_gain": row.get("first_batch_delta_gain", ""),
        "last_dpo_loss": last.get("dpo_loss", ""),
        "last_loss": last.get("loss", ""),
        "last_chosen_nll": last.get("chosen_nll", ""),
        "elapsed_seconds": row.get("elapsed_seconds", ""),
    }


def load_negative_rows(run_dir: Path, phase: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob(f"*_{phase}_by_negative_type.csv")):
        config = path.name.removesuffix(f"_{phase}_by_negative_type.csv")
        for row in read_csv(path):
            row = dict(row)
            row["config_name"] = config
            row["phase"] = phase
            out.append(row)
    return out


def make_decision(summary_rows: list[dict[str, Any]], negative_rows: list[dict[str, Any]]) -> dict[str, Any]:
    one_step_ok = any(to_float(row.get("first_batch_delta_gain")) > 0 for row in summary_rows)
    overfit_ok = any(to_float(row.get("after_dpo_pref_acc")) >= 0.75 for row in summary_rows)
    field_score_ok = any(
        row.get("logp_mode") == "field"
        and "score" in str(row.get("config_name", ""))
        and to_float(row.get("after_dpo_pref_acc")) >= 0.70
        for row in summary_rows
    )
    low_failure_after = [
        row
        for row in negative_rows
        if row.get("phase") == "after" and row.get("negative_type") == "low_failure_erasure_counterfactual"
    ]
    low_failure_ok = any(to_float(row.get("dpo_pref_acc")) >= 0.60 for row in low_failure_after)
    if not one_step_ok:
        recommendation = "inspect_trainer_gradient_or_label_mask"
        reason = "No config increased first-batch DPO delta after one update."
    elif overfit_ok and field_score_ok:
        recommendation = "field_mask_scale_sanity_passed_run_corrected_dev_scout"
        reason = "At least one field-masked config overfit train preferences and score mismatch improved."
    elif overfit_ok:
        recommendation = "scale_sanity_partial_field_mask_needs_refinement"
        reason = "A config overfit overall train preferences, but score/low-failure field behavior is not fully solved."
    else:
        recommendation = "increase_beta_steps_or_fix_field_mask_before_dev_scout"
        reason = "No config reached 0.75 train DPO preference accuracy."
    return {
        "recommendation": recommendation,
        "reason": reason,
        "checks": {
            "one_step_delta_gain_positive": one_step_ok,
            "any_after_dpo_pref_acc_ge_0p75": overfit_ok,
            "field_score_after_acc_ge_0p70": field_score_ok,
            "low_failure_after_acc_ge_0p60": low_failure_ok,
        },
        "test_read": False,
        "dev_read": False,
        "small_scale_sanity_only": True,
        "raw_predictions_committed": False,
    }


def write_report(out_dir: Path, summary_rows: list[dict[str, Any]], negative_rows: list[dict[str, Any]], decision: dict[str, Any]) -> None:
    lines = [
        "# Exp25R3 Loss-Scale + Field-Masked SRC-DPO Sanity",
        "",
        "This is a train-only sanity check. It does not read dev/test and does not generate predictions.",
        "",
        "## Overall",
        "",
        "| config | mode | beta | pref_ftx | n | before acc | after acc | acc gain | first-step delta gain | mean delta gain |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| `{row.get('config_name')}` | {row.get('logp_mode')} | {row.get('beta')} | "
            f"{row.get('pref_ftx')} | {row.get('rows')} | {fmt(row.get('before_dpo_pref_acc'))} | "
            f"{fmt(row.get('after_dpo_pref_acc'))} | {fmt(row.get('acc_gain'))} | "
            f"{fmt(row.get('first_batch_delta_gain'))} | {fmt(row.get('mean_delta_gain'))} |"
        )
    after_negative = [row for row in negative_rows if row.get("phase") == "after"]
    lines.extend(["", "## After Training By Negative Type", "", "| config | negative_type | n | dpo pref acc | mean delta |", "|---|---|---:|---:|---:|"])
    for row in after_negative:
        lines.append(
            f"| `{row.get('config_name')}` | `{row.get('negative_type')}` | {row.get('n')} | "
            f"{fmt(row.get('dpo_pref_acc'))} | {fmt(row.get('mean_delta'))} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- recommendation: `{decision['recommendation']}`",
            f"- reason: {decision['reason']}",
            "",
            "## Guardrails",
            "",
            "- No dev/test split is read.",
            "- No generated prediction JSONL is written.",
            "- This experiment is not a formal dev result.",
        ]
    )
    write_text(out_dir / "reports" / "exp25r3_loss_scale_sanity_report.md", "\n".join(lines))


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.out_dir / "run_summaries"
    summary_rows = [flatten_summary(read_json(path)) for path in sorted(run_dir.glob("*.json"))]
    negative_rows = load_negative_rows(run_dir, "before") + load_negative_rows(run_dir, "after")
    decision = make_decision(summary_rows, negative_rows)
    write_csv(args.out_dir / "tables" / "exp25r3_loss_scale_summary.csv", summary_rows)
    write_csv(args.out_dir / "tables" / "exp25r3_loss_scale_by_negative_type.csv", negative_rows)
    write_json(args.out_dir / "decision" / "exp25r3_loss_scale_decision.json", decision)
    write_report(args.out_dir, summary_rows, negative_rows, decision)
    return {
        "configs": len(summary_rows),
        "recommendation": decision["recommendation"],
        "out_dir": str(args.out_dir),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate Exp25R3 sanity results.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(diagnose(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
