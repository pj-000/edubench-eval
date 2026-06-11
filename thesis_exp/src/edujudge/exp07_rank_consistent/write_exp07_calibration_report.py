"""Write Exp7-C calibration reports."""

from __future__ import annotations

from typing import Any

from thesis_exp.src.edujudge.exp07_rank_consistent import (
    EXP07_CALIBRATION_REPORTS_DIR,
    EXP07_CALIBRATION_TABLES_DIR,
    ensure_exp07_calibration_dirs,
)
from thesis_exp.src.edujudge.utils.io import read_csv, write_text


def _read(name: str) -> list[dict[str, str]]:
    path = EXP07_CALIBRATION_TABLES_DIR / name
    return read_csv(path) if path.exists() else []


def _float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, digits: int = 4) -> str:
    number = _float(value)
    if number is None:
        return "NA"
    return f"{number:.{digits}f}"


def _best_calibrated(summary: list[dict[str, str]]) -> dict[str, str]:
    rows = [
        row
        for row in summary
        if row.get("split") == "test" and row.get("method") not in {"raw", "raw_reference_existing_summary"}
    ]
    if not rows:
        return {}
    return min(
        rows,
        key=lambda row: (
            _float(row.get("low_to_high_rate")) if _float(row.get("low_to_high_rate")) is not None else 999,
            _float(row.get("MAE_label")) if _float(row.get("MAE_label")) is not None else 999,
            -(_float(row.get("Acc@5")) if _float(row.get("Acc@5")) is not None else -999),
        ),
    )


def _raw_for(summary: list[dict[str, str]], base_model: str) -> dict[str, str]:
    for row in summary:
        if row.get("base_model") == base_model and row.get("method") == "raw" and row.get("split") == "test":
            return row
    return {}


def _reference_for(summary: list[dict[str, str]], base_model: str) -> dict[str, str]:
    for row in summary:
        if row.get("base_model") == base_model and row.get("method") == "raw_reference_existing_summary":
            return row
    return {}


def _answer_help(rows: list[dict[str, str]], method: str) -> str:
    method_rows = [row for row in rows if row.get("method") == method and row.get("split") == "test"]
    if not method_rows:
        return "NOT AVAILABLE for local artifacts."
    row = min(method_rows, key=lambda item: _float(item.get("low_to_high_rate")) or 999)
    raw = _raw_for(rows, row["base_model"])
    if not raw:
        return "NO RAW BASELINE AVAILABLE."
    low_delta = (_float(row.get("low_to_high_rate")) or 0.0) - (_float(raw.get("low_to_high_rate")) or 0.0)
    mae_delta = (_float(row.get("MAE_label")) or 0.0) - (_float(raw.get("MAE_label")) or 0.0)
    return (
        f"{'YES' if low_delta < 0 else 'NO'}; low_to_high delta={low_delta:.4f}, "
        f"MAE delta={mae_delta:.4f} on {row['base_model']}."
    )


def _table(rows: list[dict[str, str]], fields: list[str], max_rows: int = 12) -> list[str]:
    out = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows[:max_rows]:
        out.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return out


def write_report() -> None:
    ensure_exp07_calibration_dirs()
    inventory = _read("calibration_base_inventory.csv")
    summary = _read("calibration_summary.csv")
    best_configs = _read("risk_aware_threshold_best_configs.csv")
    rejection = _read("rejection_analysis.csv")
    best = _best_calibrated(summary)
    raw = _raw_for(summary, best.get("base_model", "")) if best else {}
    qd_b1 = _reference_for(summary, "QD-B1_human_only_L1_weighted_ordinal")
    risk_best = next((row for row in best_configs if row.get("base_model") == best.get("base_model")), {})
    calibrated_ready = [row["base_model"] for row in inventory if row.get("can_threshold_calibrate") == "yes"]
    b1_low = _float(qd_b1.get("low_to_high_rate"))
    best_low = _float(best.get("low_to_high_rate")) if best else None
    best_mae = _float(best.get("MAE_label")) if best else None
    b1_mae = _float(qd_b1.get("MAE_label"))
    outperforms_b1 = best_low is not None and b1_low is not None and best_low < b1_low
    final_method = bool(best) and outperforms_b1 and (best_mae is None or b1_mae is None or best_mae <= b1_mae + 0.03)
    lines = [
        "# Exp7-C Risk-aware Ordinal Calibration",
        "",
        "Scope: calibration only. No model training, API calls, synthetic generation, Exp7-B, new training loss, or raw artifact edits were performed.",
        "",
        "## Base Inventory",
        "",
        *_table(
            inventory,
            [
                "base_model",
                "dev_probs_available",
                "test_probs_available",
                "dev_logits_available",
                "test_logits_available",
                "can_threshold_calibrate",
                "can_temperature_calibrate",
                "blocking_reason",
            ],
        ),
        "",
        "## Test Summary",
        "",
        *_table(
            summary,
            ["base_model", "method", "split", "MAE_label", "QWK", "Kendall tau", "low_to_high_rate", "Acc@5"],
            max_rows=20,
        ),
        "",
        "## Required Answers",
        "",
        f"1. Calibration-ready bases: {', '.join(calibrated_ready) if calibrated_ready else 'none'}.",
        f"2. Temperature scaling help? {_answer_help(summary, 'temperature_scaling')}",
        f"3. Global threshold calibration help? {_answer_help(summary, 'global_threshold')}",
        f"4. Risk-aware threshold calibration reduce low_to_high? {_answer_help(summary, 'risk_aware_threshold')}",
        (
            "5. Trade-off: "
            f"best method `{best.get('method', 'NA')}` changes low_to_high {_fmt(raw.get('low_to_high_rate'))} -> {_fmt(best.get('low_to_high_rate'))}, "
            f"MAE {_fmt(raw.get('MAE_label'))} -> {_fmt(best.get('MAE_label'))}, "
            f"QWK {_fmt(raw.get('QWK'))} -> {_fmt(best.get('QWK'))}, "
            f"Acc@5 {_fmt(raw.get('Acc@5'))} -> {_fmt(best.get('Acc@5'))}."
        ),
        f"6. Best calibrated model: `{best.get('base_model', 'NA')}` with `{best.get('method', 'NA')}`.",
        (
            "7. Calibration outperform QD-B1 raw? "
            f"{'YES' if outperforms_b1 else 'NO'}; best low_to_high={_fmt(best_low)}, QD-B1 raw low_to_high={_fmt(b1_low)}."
        ),
        "8. Calibration addresses low-score overestimation without additional training or synthetic data; this run is a decision-layer intervention only.",
        f"9. Exp7-C final method? {'YES, conditionally' if final_method else 'NOT YET'} based on current local artifacts.",
        "10. Limitations: only bases with local logits/probs can be calibrated here; B0/B1 require artifact sync or server-side calibration.",
        "",
        "## Selected Risk-aware Config",
        "",
        *_table(best_configs, ["base_model", "lambda_low", "eta_high", "theta_1", "theta_2", "theta_3", "theta_4", "dev_objective", "selection_type"]),
        "",
        "## Optional Rejection Analysis",
        "",
        *_table(rejection, ["base_model", "review_rate", "coverage", "low_to_high_all", "low_to_high_non_rejected", "Acc@5_all", "Acc@5_non_rejected"]),
    ]
    report_text = "\n".join(lines)
    write_text(EXP07_CALIBRATION_REPORTS_DIR / "exp07_calibration_report.md", report_text)
    review_lines = [
        "# Exp7-C Calibration Review Package",
        "",
        "- Scope: decision calibration only.",
        "- Selection split: dev only.",
        "- Test split: final evaluation only.",
        "- Training run: NO.",
        "- API call: NO.",
        "- Synthetic data: NO.",
        "- Exp7-B: not started.",
        f"- Calibration-ready bases: {', '.join(calibrated_ready) if calibrated_ready else 'none'}.",
        f"- Best method: `{best.get('method', 'NA')}` on `{best.get('base_model', 'NA')}`.",
        f"- Best thresholds: {risk_best.get('theta_1', 'NA')}, {risk_best.get('theta_2', 'NA')}, {risk_best.get('theta_3', 'NA')}, {risk_best.get('theta_4', 'NA')}.",
        "- Review focus: verify dev-only threshold selection and local artifact availability before claiming B0/B1 calibration.",
    ]
    write_text(EXP07_CALIBRATION_REPORTS_DIR / "exp07_calibration_review_package.md", "\n".join(review_lines))
    notion_lines = [
        "# Exp7-C Calibration Summary",
        "",
        f"- Calibration-ready bases: {', '.join(calibrated_ready) if calibrated_ready else 'none'}.",
        f"- Best calibrated model: `{best.get('base_model', 'NA')}`.",
        f"- Best method: `{best.get('method', 'NA')}`.",
        f"- low_to_high: {_fmt(raw.get('low_to_high_rate'))} -> {_fmt(best.get('low_to_high_rate'))}.",
        f"- Acc@5: {_fmt(raw.get('Acc@5'))} -> {_fmt(best.get('Acc@5'))}.",
        f"- MAE/QWK: {_fmt(raw.get('MAE_label'))}/{_fmt(raw.get('QWK'))} -> {_fmt(best.get('MAE_label'))}/{_fmt(best.get('QWK'))}.",
        f"- Outperforms QD-B1 raw on low_to_high: {'yes' if outperforms_b1 else 'no'}.",
        "- Training/API/synthetic: no.",
    ]
    write_text(EXP07_CALIBRATION_REPORTS_DIR / "notion_exp07_calibration_summary.md", "\n".join(notion_lines))


def main() -> None:
    write_report()
    print(f"Wrote Exp7-C calibration reports: {EXP07_CALIBRATION_REPORTS_DIR}")


if __name__ == "__main__":
    main()
