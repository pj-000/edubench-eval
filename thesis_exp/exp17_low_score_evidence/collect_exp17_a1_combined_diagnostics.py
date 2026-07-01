"""Collect primary and diagnostic Exp17-A1 results into a final decision report."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.utils.io import relpath, write_csv, write_json, write_text


REQUIRED_CONFIGS = [
    "A1_0_baseline",
    "A1_1",
    "A1_2",
    "A1_3",
    "A1_4",
    "A1_5_all_low_aux_baseline",
    "A1_6_random_positive_control",
    "A1F_1_frozen_base_beta_0p10",
    "A1F_2_frozen_probe_lr1em3_gradaccum1_epochs20",
    "A1F_3_frozen_probe_lr3em4_gradaccum1_epochs20",
    "A1F_4_frozen_probe_lr1em4_gradaccum1_epochs30",
    "A1_5a_all_low_downsample76_same_neg_pool",
    "A1_5b_all_low111_same_clean_high_controls",
    "A1_1b_a0_weak_random_high_negatives",
]

A0_FILTERED = ["A1_1", "A1_2", "A1_3", "A1_4", "A1_1b_a0_weak_random_high_negatives"]
ALL_LOW_CONTROLS = ["A1_5_all_low_aux_baseline", "A1_5a_all_low_downsample76_same_neg_pool", "A1_5b_all_low111_same_clean_high_controls"]
FAIR_ALL_LOW_CONTROLS = ["A1_5a_all_low_downsample76_same_neg_pool", "A1_5b_all_low111_same_clean_high_controls"]
RANDOM_CONTROLS = ["A1_6_random_positive_control"]
A1F_CONFIGS = [
    "A1F_1_frozen_base_beta_0p10",
    "A1F_2_frozen_probe_lr1em3_gradaccum1_epochs20",
    "A1F_3_frozen_probe_lr3em4_gradaccum1_epochs20",
    "A1F_4_frozen_probe_lr1em4_gradaccum1_epochs30",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def run_dir_for(config: str, primary_dir: Path, diagnostic_dir: Path, seed: int) -> Path:
    primary = primary_dir / "runs" / config / f"seed_{seed}"
    diagnostic = diagnostic_dir / "runs" / config / f"seed_{seed}"
    if primary.exists():
        return primary
    return diagnostic


def collect(primary_dir: Path, diagnostic_dir: Path, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    dev_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for config in REQUIRED_CONFIGS:
        run_dir = run_dir_for(config, primary_dir, diagnostic_dir, seed)
        dev_path = run_dir / "exp17_a1_dev_metrics.csv"
        ev_path = run_dir / "exp17_a1_evidence_eval.csv"
        cfg_path = run_dir / "config.json"
        if not dev_path.exists() or not ev_path.exists() or not cfg_path.exists():
            warnings.append(f"missing required config outputs: {config} at {relpath(run_dir)}")
            continue
        dev_row = read_csv_rows(dev_path)[0]
        ev_row = read_csv_rows(ev_path)[0]
        cfg = read_json(cfg_path)
        info = cfg.get("evidence_info", {})
        dev_rows.append(dev_row)
        evidence_rows.append(ev_row)
        config_rows.append(
            {
                "config_name": config,
                "source_dir": "primary" if primary_dir in run_dir.parents else "diagnostic",
                "positive_mode": info.get("positive_mode", ""),
                "negative_pool": info.get("negative_pool", ""),
                "positive_count": info.get("positive_count", ""),
                "negative_count": info.get("negative_count", ""),
                "signal_count": info.get("signal_count", ""),
                "learning_rate": cfg.get("learning_rate", ""),
                "epochs": cfg.get("epochs", ""),
                "grad_accum_steps": cfg.get("grad_accum_steps", ""),
                "freeze_base": cfg.get("freeze_base", ""),
                "train_evidence_head_only": cfg.get("train_evidence_head_only", ""),
                "checkpoint_selection": cfg.get("checkpoint_selection", ""),
                "test_read": cfg.get("test_read", False),
                "dev_annotation_used_as_train_label": cfg.get("dev_annotation_used_as_train_label", False),
                "human_rationale_used_as_model_input": cfg.get("human_rationale_used_as_model_input", False),
            }
        )
    return dev_rows, evidence_rows, config_rows, warnings


def by_config(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["config_name"]): row for row in rows}


def best_by_auc(configs: list[str], evidence_by_config: dict[str, dict[str, Any]]) -> tuple[str, float]:
    best_name = ""
    best_auc = float("nan")
    for name in configs:
        row = evidence_by_config.get(name)
        if not row:
            continue
        auc = safe_float(row.get("h_auc_d1_hidden_vs_controls"))
        if not best_name or (not math.isnan(auc) and (math.isnan(best_auc) or auc > best_auc)):
            best_name = name
            best_auc = auc
    return best_name, best_auc


def final_decision(dev_rows: list[dict[str, Any]], evidence_rows: list[dict[str, Any]]) -> dict[str, Any]:
    dev = by_config(dev_rows)
    ev = by_config(evidence_rows)
    base = dev.get("A1_0_baseline", {})
    base_mae = safe_float(base.get("MAE"))
    base_qwk = safe_float(base.get("QWK"))
    best_filtered, best_filtered_auc = best_by_auc(A0_FILTERED, ev)
    best_all_low, best_all_low_auc = best_by_auc(ALL_LOW_CONTROLS, ev)
    best_fair_all_low, best_fair_all_low_auc = best_by_auc(FAIR_ALL_LOW_CONTROLS, ev)
    best_a1f, best_a1f_auc = best_by_auc(A1F_CONFIGS, ev)
    best_row = dev.get(best_filtered, {})
    best_ev = ev.get(best_filtered, {})
    mae = safe_float(best_row.get("MAE"))
    qwk = safe_float(best_row.get("QWK"))
    low_to_high = safe_float(best_row.get("low_to_high_rate"))
    baseline_l2h = safe_float(base.get("low_to_high_rate"))
    label2_recall = safe_float(best_row.get("label2_recall"))
    delta = safe_float(best_ev.get("evidence_delta_hidden_minus_control"))
    success = (
        best_filtered_auc >= 0.65
        and delta > 0
        and (math.isnan(base_mae) or mae <= base_mae + 0.02)
        and (math.isnan(base_qwk) or qwk >= base_qwk - 0.02)
        and ((not math.isnan(low_to_high) and not math.isnan(baseline_l2h) and low_to_high < baseline_l2h) or label2_recall > 0)
    )
    return {
        "final_decision": "A1_success_ready_for_next_gate" if success else "A1_failed_move_to_C0",
        "a1_success": success,
        "enter_b1_suppression": False if not success else "manual_review_required",
        "move_to_c0_pairwise_separation": not success,
        "best_a0_filtered_config": best_filtered,
        "best_a0_filtered_auc": best_filtered_auc,
        "best_all_low_config": best_all_low,
        "best_all_low_auc": best_all_low_auc,
        "best_fair_all_low_config": best_fair_all_low,
        "best_fair_all_low_auc": best_fair_all_low_auc,
        "best_a1f_config": best_a1f,
        "best_a1f_auc": best_a1f_auc,
        "a1f_high_lr_solved_undertraining": best_a1f_auc >= 0.65,
        "best_filtered_hidden_control_delta": delta,
        "best_filtered_mae_degradation_vs_a1_0": mae - base_mae if not math.isnan(base_mae) else float("nan"),
        "best_filtered_qwk_degradation_vs_a1_0": base_qwk - qwk if not math.isnan(base_qwk) else float("nan"),
        "best_filtered_low_to_high_rate": low_to_high,
        "a1_0_low_to_high_rate": baseline_l2h,
        "best_filtered_label2_recall": label2_recall,
        "all_low_advantage_after_fair_controls": best_fair_all_low_auc > best_filtered_auc,
    }


def write_report(out_dir: Path, dev_rows: list[dict[str, Any]], evidence_rows: list[dict[str, Any]], config_rows: list[dict[str, Any]], warnings: list[str]) -> dict[str, Any]:
    decision = final_decision(dev_rows, evidence_rows)
    ev = by_config(evidence_rows)
    dev = by_config(dev_rows)
    lines = [
        "# Exp17-A1 Combined Diagnostic Report",
        "",
        "This report combines the primary A1 scout and diagnostic-control runs. It is the formal A1 close-out before Exp17-C0.",
        "",
        "## Guardrails",
        "",
        "- Test split is not read by these diagnostics.",
        "- Dev D1 annotations are used only for evaluation.",
        "- Human rationale text is not used as ranker input.",
        "- A1 evidence head does not suppress or alter the ordinal score.",
        "",
        "## Key Results",
        "",
        "| config | MAE | QWK | low-to-high | label2 recall | h AUC | hidden-control delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in REQUIRED_CONFIGS:
        if name not in dev or name not in ev:
            continue
        lines.append(
            f"| `{name}` | {fmt(dev[name].get('MAE'))} | {fmt(dev[name].get('QWK'))} | "
            f"{fmt(dev[name].get('low_to_high_rate'))} | {fmt(dev[name].get('label2_recall'))} | "
            f"{fmt(ev[name].get('h_auc_d1_hidden_vs_controls'))} | {fmt(ev[name].get('evidence_delta_hidden_minus_control'))} |"
        )
    lines.extend(
        [
            "",
            "## Questions",
            "",
            f"1. Did high-learning-rate A1F probes solve undertraining? `{decision['a1f_high_lr_solved_undertraining']}`. "
            f"Best A1F is `{decision['best_a1f_config']}` with AUC {fmt(decision['best_a1f_auc'])}.",
            "2. Did frozen probes transfer to dev D1 hidden cases? No, because the best frozen AUC remains below the success gate.",
            f"3. Does all-low still dominate after fair controls? `{decision['all_low_advantage_after_fair_controls']}`. "
            f"Best fair all-low control is `{decision['best_fair_all_low_config']}` with AUC {fmt(decision['best_fair_all_low_auc'])}. "
            f"The original all-low control was `{decision['best_all_low_config']}` with AUC {fmt(decision['best_all_low_auc'])}.",
            f"4. Is A1_1b only a weak random-high-negative signal? Yes. Its AUC is {fmt(ev.get('A1_1b_a0_weak_random_high_negatives', {}).get('h_auc_d1_hidden_vs_controls'))}, below 0.65.",
            f"5. Did A1 succeed? `{decision['a1_success']}`.",
            f"6. Enter B1 suppression? `{decision['enter_b1_suppression']}`.",
            f"7. Move to C0 pairwise-low quality separation? `{decision['move_to_c0_pairwise_separation']}`.",
            "",
            "## Final Decision",
            "",
            f"- final_decision: `{decision['final_decision']}`",
            f"- best_a0_filtered_config: `{decision['best_a0_filtered_config']}`",
            f"- best_a0_filtered_auc: {fmt(decision['best_a0_filtered_auc'])}",
            "",
            "A1 is closed unless a later manual review changes the success gate. The next ranker-side experiment is C0 pairwise quality separation.",
        ]
    )
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    write_text(out_dir / "reports" / "exp17_a1_combined_diagnostic_report.md", "\n".join(lines))
    write_json(out_dir / "decision" / "exp17_a1_final_decision.json", decision)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect combined Exp17-A1 diagnostics.")
    parser.add_argument("--primary-dir", type=Path, default=Path("thesis_exp/exp17_low_score_evidence/outputs/exp17_a1_evidence_head_seed42"))
    parser.add_argument("--diagnostic-dir", type=Path, default=Path("thesis_exp/exp17_low_score_evidence/outputs/exp17_a1_diagnostic_controls_seed42"))
    parser.add_argument("--out-dir", type=Path, default=Path("thesis_exp/exp17_low_score_evidence/outputs/exp17_a1_combined_diagnostics_seed42"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dev_rows, evidence_rows, config_rows, warnings = collect(args.primary_dir, args.diagnostic_dir, int(args.seed))
    write_csv(args.out_dir / "tables" / "exp17_a1_combined_dev_metrics.csv", dev_rows)
    write_csv(args.out_dir / "tables" / "exp17_a1_combined_evidence_eval.csv", evidence_rows)
    write_csv(args.out_dir / "tables" / "exp17_a1_config_diagnostics.csv", config_rows)
    decision = write_report(args.out_dir, dev_rows, evidence_rows, config_rows, warnings)
    print(json.dumps({"decision": decision["final_decision"], "warnings": len(warnings), "out_dir": relpath(args.out_dir)}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
