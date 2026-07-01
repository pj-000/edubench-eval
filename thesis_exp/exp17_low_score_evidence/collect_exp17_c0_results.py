"""Collect Exp17-C0 pairwise separation scout results."""

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

from thesis_exp.src.edujudge.utils.io import relpath, write_csv, write_json, write_text


DEFAULT_OUTPUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp17_c0_pairwise_separation_seed42")
C0_CONFIG_NAMES = [
    "C0_0_ordinal_continue",
    "C0_1_all_pairs_gamma0p02_m0p2",
    "C0_2_all_pairs_gamma0p05_m0p2",
    "C0_3_all_pairs_gamma0p10_m0p2",
    "C0_4_pairwise_low_only_gamma0p05_m0p2",
    "C0_5_evidence_positive_plus_pairwise_low_gamma0p05_m0p2",
    "C0_6_random_pair_control_gamma0p05_m0p2",
    "C0_7_same_subject_only_gamma0p05_m0p2",
    "C0_8_high_weight_only_gamma0p05_m0p2",
    "C0_9_same_subject_high_weight_gamma0p05_m0p2",
    "C0_10_exclude_format_auxiliary_gamma0p05_m0p2",
    "C0_11_exclude_answer_key_dependent_gamma0p05_m0p2",
    "C0_12_random_matched_metric_rubric_gamma0p05_m0p2",
    "C0_13_random_matched_metric_rubric_subject_gamma0p05_m0p2",
    "C0_14_same_question_group_upper_bound_gamma0p05_m0p2",
]
DEV_FIELDS = [
    "config_name",
    "seed",
    "gamma",
    "margin",
    "temperature",
    "pair_source",
    "MAE",
    "QWK",
    "accuracy",
    "low_to_high_count",
    "low_to_high_rate",
    "label2_recall",
    "label2_pred_ge4_rate",
    "monotonic_violation_rate",
    "mean_s_label2",
    "mean_s_label4_5",
    "mean_g_i3_label2",
]
PAIR_FIELDS = [
    "config_name",
    "seed",
    "pair_source",
    "train_pair_count",
    "dev_d1_pair_count",
    "margin",
    "train_pair_gap_mean",
    "train_pair_gap_p10",
    "train_pair_gap_violation_rate_at_margin",
    "dev_d1_s_gap_control_minus_hidden_mean",
    "dev_d1_s_gap_control_minus_hidden_median",
    "dev_d1_s_gap_violation_rate",
    "dev_d1_g_i3_hidden_mean",
    "dev_d1_g_i3_control_mean",
    "d1_hidden_vs_control_s_auc",
    "d1_pairwise_low_vs_control_s_auc",
    "d1_evidence_positive_vs_control_s_auc",
    "pair_source_available_pair_count",
    "pair_weight_mean",
    "pair_weight_p50",
    "pair_weight_p75",
]


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


RANDOM_PAIR_SOURCES = {"random_low_high_pairs", "random_matched_metric_rubric", "random_matched_metric_rubric_subject"}
UPPER_BOUND_PAIR_SOURCE = "same_question_group_upper_bound"


def within_quality_guard(row: dict[str, Any], base: dict[str, Any]) -> bool:
    base_mae = safe_float(base.get("MAE"))
    base_qwk = safe_float(base.get("QWK"))
    mae = safe_float(row.get("MAE"))
    qwk = safe_float(row.get("QWK"))
    mono = safe_float(row.get("monotonic_violation_rate"))
    return (
        mono == 0.0
        and (math.isnan(base_mae) or mae <= base_mae + 0.02)
        and (math.isnan(base_qwk) or qwk >= base_qwk - 0.02)
    )


def risk_success_for(row: dict[str, Any], base: dict[str, Any]) -> bool:
    base_l2h = safe_float(base.get("low_to_high_rate"))
    l2h = safe_float(row.get("low_to_high_rate"))
    label2 = safe_float(row.get("label2_recall"))
    return within_quality_guard(row, base) and ((not math.isnan(base_l2h) and l2h <= base_l2h - 0.05) or label2 > 0)


def latent_success_for(row: dict[str, Any], pair_row: dict[str, Any], base: dict[str, Any], base_pair: dict[str, Any]) -> bool:
    base_g = safe_float(base.get("mean_g_i3_label2"))
    base_gap = safe_float(base_pair.get("dev_d1_s_gap_control_minus_hidden_mean"))
    g = safe_float(row.get("mean_g_i3_label2"))
    gap = safe_float(pair_row.get("dev_d1_s_gap_control_minus_hidden_mean"))
    auc = safe_float(pair_row.get("d1_hidden_vs_control_s_auc"))
    return within_quality_guard(row, base) and (
        (not math.isnan(base_gap) and not math.isnan(gap) and gap > base_gap)
        or (not math.isnan(auc) and auc >= 0.60)
        or (not math.isnan(base_g) and not math.isnan(g) and g < base_g)
    )


def row_pair_source(row: dict[str, Any]) -> str:
    return str(row.get("pair_source", ""))


def is_main_real_config(row: dict[str, Any]) -> bool:
    source = row_pair_source(row)
    return source not in {"none", UPPER_BOUND_PAIR_SOURCE, *RANDOM_PAIR_SOURCES}


def beats_random_matched(row: dict[str, Any], pair_row: dict[str, Any], random_row: dict[str, Any], random_pair: dict[str, Any]) -> bool:
    l2h = safe_float(row.get("low_to_high_rate"))
    random_l2h = safe_float(random_row.get("low_to_high_rate"))
    gap = safe_float(pair_row.get("dev_d1_s_gap_control_minus_hidden_mean"))
    random_gap = safe_float(random_pair.get("dev_d1_s_gap_control_minus_hidden_mean"))
    auc = safe_float(pair_row.get("d1_hidden_vs_control_s_auc"))
    random_auc = safe_float(random_pair.get("d1_hidden_vs_control_s_auc"))
    return (
        (not math.isnan(l2h) and not math.isnan(random_l2h) and l2h < random_l2h)
        or (not math.isnan(gap) and not math.isnan(random_gap) and gap > random_gap)
        or (not math.isnan(auc) and not math.isnan(random_auc) and auc > random_auc)
    )


def select_best(rows: list[dict[str, Any]], pair: dict[str, dict[str, Any]], base: dict[str, Any], base_pair: dict[str, Any]) -> str:
    candidates = []
    for row in rows:
        name = str(row.get("config_name", ""))
        pair_row = pair.get(name, {})
        risk = risk_success_for(row, base)
        latent = latent_success_for(row, pair_row, base, base_pair)
        label2 = safe_float(row.get("label2_recall"))
        l2h = safe_float(row.get("low_to_high_rate"))
        gap = safe_float(pair_row.get("dev_d1_s_gap_control_minus_hidden_mean"))
        auc = safe_float(pair_row.get("d1_hidden_vs_control_s_auc"))
        mae = safe_float(row.get("MAE"))
        qwk = safe_float(row.get("QWK"))
        candidates.append((not risk, not latent, math.isnan(label2), -label2 if not math.isnan(label2) else 0, l2h, -auc if not math.isnan(auc) else 0, -gap if not math.isnan(gap) else 0, mae, -qwk, name))
    if not candidates:
        return ""
    candidates.sort()
    return candidates[0][-1]


def decision(dev_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    dev = by_config(dev_rows)
    pair = by_config(pair_rows)
    base = dev.get("C0_0_ordinal_continue", {})
    base_pair = pair.get("C0_0_ordinal_continue", {})
    main_rows = [row for row in dev_rows if is_main_real_config(row)]
    upper_rows = [row for row in dev_rows if row_pair_source(row) == UPPER_BOUND_PAIR_SOURCE]
    best_main = select_best(main_rows, pair, base, base_pair)
    best_upper = select_best(upper_rows, pair, base, base_pair)
    best_dev = dev.get(best_main, {})
    best_pair = pair.get(best_main, {})
    base_mae = safe_float(base.get("MAE"))
    base_qwk = safe_float(base.get("QWK"))
    base_l2h = safe_float(base.get("low_to_high_rate"))
    base_g = safe_float(base.get("mean_g_i3_label2"))
    base_gap = safe_float(base_pair.get("dev_d1_s_gap_control_minus_hidden_mean"))
    mae = safe_float(best_dev.get("MAE"))
    qwk = safe_float(best_dev.get("QWK"))
    l2h = safe_float(best_dev.get("low_to_high_rate"))
    label2 = safe_float(best_dev.get("label2_recall"))
    g = safe_float(best_dev.get("mean_g_i3_label2"))
    gap = safe_float(best_pair.get("dev_d1_s_gap_control_minus_hidden_mean"))
    random_matched = dev.get("C0_12_random_matched_metric_rubric_gamma0p05_m0p2", {})
    random_matched_pair = pair.get("C0_12_random_matched_metric_rubric_gamma0p05_m0p2", {})
    same_subject_high = dev.get("C0_9_same_subject_high_weight_gamma0p05_m0p2", {})
    same_question = dev.get("C0_14_same_question_group_upper_bound_gamma0p05_m0p2", {})
    main_latent_success = any(latent_success_for(row, pair.get(str(row.get("config_name")), {}), base, base_pair) for row in main_rows)
    main_risk_success = any(risk_success_for(row, base) for row in main_rows)
    upper_success = any(
        risk_success_for(row, base) or latent_success_for(row, pair.get(str(row.get("config_name")), {}), base, base_pair)
        for row in upper_rows
    )
    best_main_beats_random = bool(best_main) and beats_random_matched(best_dev, best_pair, random_matched, random_matched_pair)
    any_main_beats_random = any(
        beats_random_matched(row, pair.get(str(row.get("config_name")), {}), random_matched, random_matched_pair)
        for row in main_rows
    )
    main_method_success = main_risk_success and best_main_beats_random
    all_pairs_ok = any(
        risk_success_for(row, base) or latent_success_for(row, pair.get(str(row.get("config_name")), {}), base, base_pair)
        for row in dev_rows
        if row.get("pair_source") == "all_a0_pairs"
    )
    strict_ok = any(
        risk_success_for(row, base) or latent_success_for(row, pair.get(str(row.get("config_name")), {}), base, base_pair)
        for row in dev_rows
        if row.get("pair_source") in {"same_subject_only", "high_weight_only_p75", "same_subject_high_weight_p75"}
    )
    same_subject_high_ok = (
        risk_success_for(same_subject_high, base)
        or latent_success_for(same_subject_high, pair.get("C0_9_same_subject_high_weight_gamma0p05_m0p2", {}), base, base_pair)
        if same_subject_high
        else False
    )
    same_question_ok = (
        risk_success_for(same_question, base)
        or latent_success_for(same_question, pair.get("C0_14_same_question_group_upper_bound_gamma0p05_m0p2", {}), base, base_pair)
        if same_question
        else False
    )
    real_ok = any(
        risk_success_for(row, base) or latent_success_for(row, pair.get(str(row.get("config_name")), {}), base, base_pair)
        for row in main_rows
    )
    random_ok = any(
        risk_success_for(row, base) or latent_success_for(row, pair.get(str(row.get("config_name")), {}), base, base_pair)
        for row in dev_rows
        if row.get("pair_source") in RANDOM_PAIR_SOURCES
    )
    if main_method_success:
        category = "C0_success_risk"
    elif same_question_ok and not all_pairs_ok and not strict_ok:
        category = "C0_cross_question_noise_likely"
    elif same_subject_high_ok and not all_pairs_ok:
        category = "C0_pair_noise_likely"
    elif main_latent_success and not main_risk_success:
        category = "C0_latent_success_only"
    elif random_ok and not any_main_beats_random:
        category = "C0_no_specific_pair_signal"
    elif not real_ok and not random_ok:
        category = "C0_failed_method_or_backbone"
    else:
        category = "C0_inconclusive"
    return {
        "final_decision": category,
        "latent_success": bool(main_latent_success),
        "risk_success": bool(main_risk_success),
        "main_method_success": bool(main_method_success),
        "c0_success": bool(main_method_success),
        "best_config": best_main,
        "best_main_config": best_main,
        "best_upper_bound_config": best_upper,
        "enter_c1": bool(main_method_success),
        "enter_b1": False,
        "mae_degradation_vs_c0_0": mae - base_mae if not math.isnan(base_mae) else float("nan"),
        "qwk_degradation_vs_c0_0": base_qwk - qwk if not math.isnan(base_qwk) else float("nan"),
        "low_to_high_delta_vs_c0_0": l2h - base_l2h if not math.isnan(base_l2h) else float("nan"),
        "label2_recall": label2,
        "mean_g_i3_label2_delta_vs_c0_0": g - base_g if not math.isnan(base_g) else float("nan"),
        "dev_d1_s_gap_delta_vs_c0_0": gap - base_gap if not math.isnan(base_gap) else float("nan"),
        "a0_pairs_outperform_random_matched_metric_rubric": bool(any_main_beats_random),
        "upper_bound_diagnostic_success": bool(upper_success),
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
        "| config | pair source | MAE | QWK | low-to-high | label2 recall | mean g_i3 label2 | D1 s gap | D1 s AUC | train pair gap | train pairs |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(dev_rows, key=lambda item: item.get("config_name", "")):
        name = row["config_name"]
        pr = pair.get(name, {})
        lines.append(
            f"| `{name}` | `{row.get('pair_source')}` | {fmt(row.get('MAE'))} | {fmt(row.get('QWK'))} | {fmt(row.get('low_to_high_rate'))} | "
            f"{fmt(row.get('label2_recall'))} | {fmt(row.get('mean_g_i3_label2'))} | "
            f"{fmt(pr.get('dev_d1_s_gap_control_minus_hidden_mean'))} | {fmt(pr.get('d1_hidden_vs_control_s_auc'))} | "
            f"{fmt(pr.get('train_pair_gap_mean'))} | {fmt(pr.get('train_pair_count'), 0)} |"
        )
    compare_pairs = [
        ("all_a0_pairs", "same_subject_only"),
        ("all_a0_pairs", "high_weight_only_p75"),
        ("all_a0_pairs", "same_subject_high_weight_p75"),
        ("all_a0_pairs", "random_low_high_pairs"),
        ("all_a0_pairs", "random_matched_metric_rubric"),
        ("all_a0_pairs", "same_question_group_upper_bound"),
    ]
    lines.extend(["", "## Noise-Control Comparisons", "", "| comparison | left best l2h | right best l2h | left D1 AUC | right D1 AUC |", "|---|---:|---:|---:|---:|"])
    for left, right in compare_pairs:
        left_rows = [row for row in dev_rows if row.get("pair_source") == left]
        right_rows = [row for row in dev_rows if row.get("pair_source") == right]
        if not left_rows or not right_rows:
            continue
        left_best = min(left_rows, key=lambda item: safe_float(item.get("low_to_high_rate")))
        right_best = min(right_rows, key=lambda item: safe_float(item.get("low_to_high_rate")))
        left_pair = pair.get(str(left_best.get("config_name")), {})
        right_pair = pair.get(str(right_best.get("config_name")), {})
        lines.append(
            f"| `{left}` vs `{right}` | {fmt(left_best.get('low_to_high_rate'))} | {fmt(right_best.get('low_to_high_rate'))} | "
            f"{fmt(left_pair.get('d1_hidden_vs_control_s_auc'))} | {fmt(right_pair.get('d1_hidden_vs_control_s_auc'))} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- final_decision: `{dec['final_decision']}`",
            f"- best_main_config: `{dec['best_main_config']}`",
            f"- best_upper_bound_config: `{dec['best_upper_bound_config']}`",
            f"- latent_success: `{dec['latent_success']}`",
            f"- risk_success: `{dec['risk_success']}`",
            f"- main_method_success: `{dec['main_method_success']}`",
            f"- c0_success: `{dec['c0_success']}`",
            f"- enter_c1: `{dec['enter_c1']}`",
            f"- enter_b1: `{dec['enter_b1']}`",
            f"- low_to_high_delta_vs_c0_0: {fmt(dec['low_to_high_delta_vs_c0_0'])}",
            f"- mean_g_i3_label2_delta_vs_c0_0: {fmt(dec['mean_g_i3_label2_delta_vs_c0_0'])}",
            f"- dev_d1_s_gap_delta_vs_c0_0: {fmt(dec['dev_d1_s_gap_delta_vs_c0_0'])}",
            f"- A0 pairs outperform random matched metric/rubric control: `{dec['a0_pairs_outperform_random_matched_metric_rubric']}`",
            f"- upper_bound_diagnostic_success: `{dec['upper_bound_diagnostic_success']}`",
            "",
            "## Decision Questions",
            "",
            f"- Did any main C0 config achieve risk success? `{dec['risk_success']}`",
            f"- Did any main C0 config achieve latent success? `{dec['latent_success']}`",
            f"- Did same_question_group_upper_bound only succeed? `{dec['upper_bound_diagnostic_success'] and not dec['latent_success'] and not dec['risk_success']}`",
            f"- Are A0 pairs better than random matched metric/rubric controls? `{dec['a0_pairs_outperform_random_matched_metric_rubric']}`",
            f"- Should we proceed to C1? `{dec['enter_c1']}`",
            f"- Should we still block B1 suppression? `{not dec['enter_b1']}`",
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
    parser.add_argument("--configs", nargs="+", default=C0_CONFIG_NAMES)
    args = parser.parse_args()

    dev_rows, pair_rows, warnings = collect_runs(args.output_dir, args.configs, int(args.seed))
    write_csv(args.output_dir / "tables" / "exp17_c0_dev_metrics.csv", dev_rows, fieldnames=DEV_FIELDS)
    write_csv(args.output_dir / "tables" / "exp17_c0_pair_eval.csv", pair_rows, fieldnames=PAIR_FIELDS)
    dec = write_report(args.output_dir, dev_rows, pair_rows, warnings)
    print(json.dumps({"decision": dec["final_decision"], "warnings": len(warnings), "out_dir": relpath(args.output_dir)}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
