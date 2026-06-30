"""Collect Exp17-A1 scout run outputs into lightweight CSV/MD summaries."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

from thesis_exp.exp17_low_score_evidence.train_exp17_a1_evidence_head import (
    DEV_METRIC_FIELDS,
    EVIDENCE_FIELDS,
    SCOUT_CONFIGS,
)
from thesis_exp.src.edujudge.utils.io import relpath, write_csv


DEFAULT_OUTPUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp17_a1_evidence_head_seed42")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value: Any) -> float:
    try:
        if value == "":
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def fmt(value: Any, digits: int = 4) -> str:
    number = safe_float(value)
    if math.isnan(number):
        return "NA"
    return f"{number:.{digits}f}"


def collect_runs(output_dir: Path, configs: list[str], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    dev_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for config in configs:
        run_dir = output_dir / "runs" / config / f"seed_{seed}"
        dev_path = run_dir / "exp17_a1_dev_metrics.csv"
        ev_path = run_dir / "exp17_a1_evidence_eval.csv"
        if not dev_path.exists() or not ev_path.exists():
            warnings.append(f"missing run outputs for {config} seed {seed}: {relpath(run_dir)}")
            continue
        dev_rows.extend(read_csv_rows(dev_path))
        evidence_rows.extend(read_csv_rows(ev_path))
    return dev_rows, evidence_rows, warnings


def best_config(dev_rows: list[dict[str, Any]], evidence_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    evidence_by_config = {row["config_name"]: row for row in evidence_rows}
    candidates = []
    for row in dev_rows:
        ev = evidence_by_config.get(row["config_name"])
        if not ev:
            continue
        auc = safe_float(ev.get("h_auc_d1_hidden_vs_controls"))
        delta = safe_float(ev.get("evidence_delta_hidden_minus_control"))
        mae = safe_float(row.get("MAE"))
        qwk = safe_float(row.get("QWK"))
        mono = safe_float(row.get("monotonic_violation_rate"))
        candidates.append((math.isnan(auc), -auc if not math.isnan(auc) else 0.0, -delta if not math.isnan(delta) else 0.0, mae, -qwk, mono, row, ev))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[:6])
    row = candidates[0][6]
    ev = candidates[0][7]
    return {**row, **{f"evidence_{k}": v for k, v in ev.items() if k not in row}}


def success_flags(dev_rows: list[dict[str, Any]], evidence_rows: list[dict[str, Any]]) -> dict[str, Any]:
    dev_by_config = {row["config_name"]: row for row in dev_rows}
    ev_by_config = {row["config_name"]: row for row in evidence_rows}
    baseline = dev_by_config.get("A1_0_baseline")
    filtered = [name for name in ["A1_1", "A1_2", "A1_3", "A1_4"] if name in dev_by_config and name in ev_by_config]
    controls = [name for name in ["A1_5_all_low_aux_baseline", "A1_6_random_positive_control"] if name in ev_by_config]
    best = best_config([dev_by_config[name] for name in filtered], [ev_by_config[name] for name in filtered]) if filtered else None
    if not best:
        return {"a1_success": False, "reason": "no completed A0-filtered A1 configs"}
    base_mae = safe_float(baseline.get("MAE")) if baseline else float("nan")
    base_qwk = safe_float(baseline.get("QWK")) if baseline else float("nan")
    mae = safe_float(best.get("MAE"))
    qwk = safe_float(best.get("QWK"))
    mono = safe_float(best.get("monotonic_violation_rate"))
    auc = safe_float(best.get("evidence_h_auc_d1_hidden_vs_controls"))
    hidden = safe_float(best.get("evidence_mean_h_d1_hidden"))
    control = safe_float(best.get("evidence_mean_h_d1_controls"))
    filtered_auc = auc
    control_aucs = [safe_float(ev_by_config[name].get("h_auc_d1_hidden_vs_controls")) for name in controls]
    beats_controls = all(math.isnan(value) or filtered_auc > value for value in control_aucs)
    return {
        "a1_success": (
            mono == 0.0
            and (math.isnan(base_mae) or mae <= base_mae + 0.02)
            and (math.isnan(base_qwk) or qwk >= base_qwk - 0.02)
            and auc >= 0.65
            and hidden > control
            and beats_controls
        ),
        "best_config": best.get("config_name"),
        "mae_degradation": mae - base_mae if not math.isnan(base_mae) else float("nan"),
        "qwk_degradation": base_qwk - qwk if not math.isnan(base_qwk) else float("nan"),
        "auc": auc,
        "hidden_minus_control": hidden - control,
        "beats_all_low_and_random_controls": beats_controls,
    }


def write_report(output_dir: Path, dev_rows: list[dict[str, Any]], evidence_rows: list[dict[str, Any]], warnings: list[str]) -> None:
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    flags = success_flags(dev_rows, evidence_rows)
    evidence_by_config = {row["config_name"]: row for row in evidence_rows}
    lines = [
        "# Exp17-A1 Evidence Head Scout Report",
        "",
        "Exp17-A1 keeps the Exp16A qmr ordinal decision function unchanged while adding a hidden-failure "
        "evidence head `h`. Joint A1 configs fine-tune the base model with an auxiliary evidence objective; "
        "A1F is a frozen-base probe that trains only the evidence head.",
        "",
        "## Guardrails",
        "",
        "- Test split is not read.",
        "- Dev D1 annotations are used only for dev evidence evaluation.",
        "- Human rationale text is not used as model input.",
        "- The evidence head does not suppress or alter `s`.",
        "- `A1_0_baseline` is a continued-training ordinal control, not the frozen original Exp16A checkpoint.",
        "- `A1_6_random_positive_control` samples random low-label positives, not arbitrary non-low positives.",
        "",
        "## Completed Configs",
        "",
        "| config | beta | neg_ratio | MAE | QWK | low-to-high | label2 recall | h AUC | hidden-control delta |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(dev_rows, key=lambda item: item.get("config_name", "")):
        ev = evidence_by_config.get(row["config_name"], {})
        lines.append(
            f"| `{row['config_name']}` | {fmt(row.get('beta'))} | {row.get('neg_ratio')} | "
            f"{fmt(row.get('MAE'))} | {fmt(row.get('QWK'))} | {fmt(row.get('low_to_high_rate'))} | "
            f"{fmt(row.get('label2_recall'))} | {fmt(ev.get('h_auc_d1_hidden_vs_controls'))} | "
            f"{fmt(ev.get('evidence_delta_hidden_minus_control'))} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- best_config: `{flags.get('best_config', 'NA')}`",
            f"- A1 success: `{flags.get('a1_success')}`",
            f"- MAE degradation vs A1_0_baseline: {fmt(flags.get('mae_degradation'))}",
            f"- QWK degradation vs A1_0_baseline: {fmt(flags.get('qwk_degradation'))}",
            f"- hidden-vs-control AUC: {fmt(flags.get('auc'))}",
            f"- hidden-control h delta: {fmt(flags.get('hidden_minus_control'))}",
            f"- beats all-low and random controls: `{flags.get('beats_all_low_and_random_controls')}`",
            "",
            "Proceed to C1 only if A1 succeeds. Keep B1 suppression delayed until the evidence head is reliable.",
        ]
    )
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    (reports_dir / "exp17_a1_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Exp17-A1 evidence-head scout results.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--configs", nargs="+", default=list(SCOUT_CONFIGS))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dev_rows, evidence_rows, warnings = collect_runs(args.output_dir, args.configs, int(args.seed))
    tables_dir = args.output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    write_csv(tables_dir / "exp17_a1_dev_metrics.csv", dev_rows, fieldnames=DEV_METRIC_FIELDS)
    write_csv(tables_dir / "exp17_a1_evidence_eval.csv", evidence_rows, fieldnames=EVIDENCE_FIELDS)
    write_report(args.output_dir, dev_rows, evidence_rows, warnings)
    print(
        {
            "dev_rows": len(dev_rows),
            "evidence_rows": len(evidence_rows),
            "warnings": len(warnings),
            "report": relpath(args.output_dir / "reports" / "exp17_a1_report.md"),
        }
    )


if __name__ == "__main__":
    main()
