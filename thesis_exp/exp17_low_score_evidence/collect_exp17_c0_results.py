"""Collect Exp17-C0 pairwise separation scout results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from thesis_exp.exp17_low_score_evidence.train_exp17_c0_pairwise_separation import C0_CONFIGS, DEV_FIELDS, PAIR_FIELDS
from thesis_exp.src.edujudge.utils.io import relpath, write_csv, write_json, write_text


DEFAULT_OUTPUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp17_c0_pairwise_separation_seed42")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value: Any) -> float:
    try:
        if value in {"", None}:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def fmt(value: Any, digits: int = 4) -> str:
    number = safe_float(value)
    return "NA" if math.isnan(number) else f"{number:.{digits}f}"


def collect_runs(output_dir: Path, configs: list[str], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    dev_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for config in configs:
        run_dir = output_dir / "runs" / config / f"seed_{seed}"
        dev_path = run_dir / "exp17_c0_dev_metrics.csv"
        pair_path = run_dir / "exp17_c0_pair_eval.csv"
        if not dev_path.exists() or not pair_path.exists():
            warnings.append(f"missing C0 outputs for {config} seed {seed}: {relpath(run_dir)}")
            continue
        dev_rows.extend(read_csv_rows(dev_path))
        pair_rows.extend(read_csv_rows(pair_path))
    return dev_rows, pair_rows, warnings


def by_config(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["config_name"]): row for row in rows}


def best_candidate(dev_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]]) -> str:
    pair = by_config(pair_rows)
    candidates = []
    for row in dev_rows:
        name = str(row["config_name"])
        if name == "C0_0_ordinal_continue" or name not in pair:
            continue
        l2h = safe_float(row.get("low_to_high_rate"))
        label2 = safe_float(row.get("label2_recall"))
        gap = safe_float(pair[name].get("dev_d1_s_gap_control_minus_hidden_mean"))
        mae = safe_float(row.get("MAE"))
        qwk = safe_float(row.get("QWK"))
        candidates.append((math.isnan(label2), -label2 if not math.isnan(label2) else 0, l2h, -gap if not math.isnan(gap) else 0, mae, -qwk, name))
    if not candidates:
        return ""
    candidates.sort()
    return candidates[0][-1]


def decision(dev_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    dev = by_config(dev_rows)
    pair = by_config(pair_rows)
    base = dev.get("C0_0_ordinal_continue", {})
    random_control = dev.get("C0_6_random_pair_control_gamma0p05_m0p2", {})
    random_pair = pair.get("C0_6_random_pair_control_gamma0p05_m0p2", {})
    best = best_candidate(dev_rows, pair_rows)
    best_dev = dev.get(best, {})
    best_pair = pair.get(best, {})
    base_mae = safe_float(base.get("MAE"))
    base_qwk = safe_float(base.get("QWK"))
    base_l2h = safe_float(base.get("low_to_high_rate"))
    base_g = safe_float(base.get("mean_g_i3_label2"))
    base_gap = safe_float(pair.get("C0_0_ordinal_continue", {}).get("dev_d1_s_gap_control_minus_hidden_mean"))
    mae = safe_float(best_dev.get("MAE"))
    qwk = safe_float(best_dev.get("QWK"))
    l2h = safe_float(best_dev.get("low_to_high_rate"))
    label2 = safe_float(best_dev.get("label2_recall"))
    mono = safe_float(best_dev.get("monotonic_violation_rate"))
    g = safe_float(best_dev.get("mean_g_i3_label2"))
    gap = safe_float(best_pair.get("dev_d1_s_gap_control_minus_hidden_mean"))
    random_l2h = safe_float(random_control.get("low_to_high_rate"))
    random_gap = safe_float(random_pair.get("dev_d1_s_gap_control_minus_hidden_mean"))
    a0_beats_random = (
        (not math.isnan(l2h) and not math.isnan(random_l2h) and l2h < random_l2h)
        or (not math.isnan(gap) and not math.isnan(random_gap) and gap > random_gap)
    )
    success = (
        best
        and mono == 0.0
        and (math.isnan(base_mae) or mae <= base_mae + 0.02)
        and (math.isnan(base_qwk) or qwk >= base_qwk - 0.02)
        and ((not math.isnan(base_l2h) and l2h <= base_l2h - 0.05) or label2 > 0)
        and (math.isnan(base_g) or g < base_g)
        and (math.isnan(base_gap) or gap > base_gap)
        and a0_beats_random
    )
    return {
        "final_decision": "C0_success_consider_C1_not_B1" if success else "C0_not_success_do_not_enter_B1",
        "c0_success": bool(success),
        "best_config": best,
        "enter_c1": bool(success),
        "enter_b1": False,
        "mae_degradation_vs_c0_0": mae - base_mae if not math.isnan(base_mae) else float("nan"),
        "qwk_degradation_vs_c0_0": base_qwk - qwk if not math.isnan(base_qwk) else float("nan"),
        "low_to_high_delta_vs_c0_0": l2h - base_l2h if not math.isnan(base_l2h) else float("nan"),
        "label2_recall": label2,
        "mean_g_i3_label2_delta_vs_c0_0": g - base_g if not math.isnan(base_g) else float("nan"),
        "dev_d1_s_gap_delta_vs_c0_0": gap - base_gap if not math.isnan(base_gap) else float("nan"),
        "a0_pairs_outperform_random_pair_control": a0_beats_random,
    }


def write_report(output_dir: Path, dev_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]], warnings: list[str]) -> dict[str, Any]:
    dec = decision(dev_rows, pair_rows)
    pair = by_config(pair_rows)
    lines = [
        "# Exp17-C0 Pairwise-Low Quality Separation Report",
        "",
        "C0 keeps Exp16A qmr Boundary Linking and adds a pairwise separation loss on `quality_score_s` only.",
        "",
        "## Guardrails",
        "",
        "- Test split is not read.",
        "- Dev D1 annotations are used only for evaluation.",
        "- Human rationale text is not used as ranker input.",
        "- Pairwise loss does not alter tau directly and does not use scalar `h`.",
        "",
        "## Completed Configs",
        "",
        "| config | MAE | QWK | low-to-high | label2 recall | mean g_i3 label2 | D1 s gap | train pair gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(dev_rows, key=lambda item: item.get("config_name", "")):
        name = row["config_name"]
        pr = pair.get(name, {})
        lines.append(
            f"| `{name}` | {fmt(row.get('MAE'))} | {fmt(row.get('QWK'))} | {fmt(row.get('low_to_high_rate'))} | "
            f"{fmt(row.get('label2_recall'))} | {fmt(row.get('mean_g_i3_label2'))} | "
            f"{fmt(pr.get('dev_d1_s_gap_control_minus_hidden_mean'))} | {fmt(pr.get('train_pair_gap_mean'))} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- final_decision: `{dec['final_decision']}`",
            f"- best_config: `{dec['best_config']}`",
            f"- c0_success: `{dec['c0_success']}`",
            f"- enter_c1: `{dec['enter_c1']}`",
            f"- enter_b1: `{dec['enter_b1']}`",
            f"- low_to_high_delta_vs_c0_0: {fmt(dec['low_to_high_delta_vs_c0_0'])}",
            f"- mean_g_i3_label2_delta_vs_c0_0: {fmt(dec['mean_g_i3_label2_delta_vs_c0_0'])}",
            f"- dev_d1_s_gap_delta_vs_c0_0: {fmt(dec['dev_d1_s_gap_delta_vs_c0_0'])}",
            f"- A0 pairs outperform random pair control: `{dec['a0_pairs_outperform_random_pair_control']}`",
            "",
            "C0 must pass the success gate before any B1 suppression experiment.",
        ]
    )
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    write_text(output_dir / "reports" / "exp17_c0_report.md", "\n".join(lines))
    write_json(output_dir / "decision" / "exp17_c0_decision.json", dec)
    return dec


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Exp17-C0 pairwise separation results.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--configs", nargs="+", default=list(C0_CONFIGS))
    args = parser.parse_args()

    dev_rows, pair_rows, warnings = collect_runs(args.output_dir, args.configs, int(args.seed))
    write_csv(args.output_dir / "tables" / "exp17_c0_dev_metrics.csv", dev_rows, fieldnames=DEV_FIELDS)
    write_csv(args.output_dir / "tables" / "exp17_c0_pair_eval.csv", pair_rows, fieldnames=PAIR_FIELDS)
    dec = write_report(args.output_dir, dev_rows, pair_rows, warnings)
    print(json.dumps({"decision": dec["final_decision"], "warnings": len(warnings), "out_dir": relpath(args.output_dir)}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
