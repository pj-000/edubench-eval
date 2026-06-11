"""Write Exp7-C2 unified server-side calibration reports."""

from __future__ import annotations

from typing import Any

from thesis_exp.src.edujudge.exp07_rank_consistent import (
    EXP07_CALIBRATION_REPORTS_DIR,
    EXP07_CALIBRATION_TABLES_DIR,
    QD_B1_RUN_ID,
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
    return "NA" if number is None else f"{number:.{digits}f}"


def _test_rows(summary: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in summary if row.get("split") == "test"]


def _raw_for(summary: list[dict[str, str]], base_model: str) -> dict[str, str]:
    return next(
        (
            row
            for row in summary
            if row.get("base_model") == base_model and row.get("method") == "raw" and row.get("split") == "test"
        ),
        {},
    )


def _best_for_base(summary: list[dict[str, str]], base_model: str) -> dict[str, str]:
    rows = [
        row
        for row in _test_rows(summary)
        if row.get("base_model") == base_model and row.get("method") != "raw"
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


def _best_overall(summary: list[dict[str, str]]) -> dict[str, str]:
    rows = [row for row in _test_rows(summary) if row.get("method") != "raw"]
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


def _table(rows: list[dict[str, str]], fields: list[str], max_rows: int = 30) -> list[str]:
    out = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows[:max_rows]:
        out.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return out


def _config_summary(row: dict[str, str]) -> str:
    method = row.get("best_method", "")
    if method == "temperature_scaling":
        return f"temperature={row.get('temperature', 'NA')}; no decision-threshold change"
    if "threshold" in method:
        return f"theta={row.get('theta_1')}, {row.get('theta_2')}, {row.get('theta_3')}, {row.get('theta_4')}"
    return "no calibration parameters"


def _method_help(summary: list[dict[str, str]], base_model: str, method: str) -> str:
    raw = _raw_for(summary, base_model)
    row = next(
        (
            item
            for item in _test_rows(summary)
            if item.get("base_model") == base_model and item.get("method") == method
        ),
        {},
    )
    if not raw or not row:
        return "NA"
    low_delta = (_float(row.get("low_to_high_rate")) or 0.0) - (_float(raw.get("low_to_high_rate")) or 0.0)
    mae_delta = (_float(row.get("MAE_label")) or 0.0) - (_float(raw.get("MAE_label")) or 0.0)
    return f"low_to_high delta={low_delta:.4f}, MAE delta={mae_delta:.4f}"


def write_c2_report() -> None:
    ensure_exp07_calibration_dirs()
    inventory = _read("server_calibration_artifact_inventory.csv")
    summary = _read("unified_calibration_summary.csv")
    best_selection = _read("best_calibrated_model_selection.csv")
    rejection = _read("unified_rejection_analysis.csv")
    ready = [row["base_model"] for row in inventory if row.get("calibration_ready") == "yes"]
    best = _best_overall(summary)
    qd_b1_raw = _raw_for(summary, QD_B1_RUN_ID)
    qd_b1_low = _float(qd_b1_raw.get("low_to_high_rate"))
    best_low = _float(best.get("low_to_high_rate"))
    beats_qd_b1 = best_low is not None and qd_b1_low is not None and best_low < qd_b1_low
    lines = [
        "# Exp7-C2 Unified Calibration After Server Artifact Sync",
        "",
        "Scope: local decision calibration after syncing server artifacts. No model training, API calls, synthetic generation, Exp7-B, new training loss, or raw artifact edits were performed.",
        "",
        "## Server Artifact Inventory",
        "",
        *_table(
            inventory,
            [
                "base_model",
                "dev_logits_available",
                "test_logits_available",
                "dev_probs_available",
                "test_probs_available",
                "dev_labels_available",
                "test_labels_available",
                "dev_predictions_available",
                "test_predictions_available",
                "calibration_ready",
            ],
        ),
        "",
        "## Unified Test Summary",
        "",
        *_table(
            _test_rows(summary),
            ["base_model", "method", "MAE_label", "QWK", "Kendall tau", "low_to_high_rate", "Acc@5"],
        ),
        "",
        "## Required Answers",
        "",
        f"1. Calibrated base models: {', '.join(ready) if ready else 'none'}.",
        "2. Best method per base:",
    ]
    for row in best_selection:
        lines.append(
            f"   - {row.get('base_model')}: {row.get('best_method')} "
            f"({_config_summary(row)})."
        )
    lines.extend(
        [
            f"3. Best overall: `{best.get('base_model', 'NA')}` with `{best.get('method', 'NA')}`.",
            (
                "4. Beats QD-B1 raw low_to_high=0.4516? "
                f"{'YES' if beats_qd_b1 else 'NO'}; best low_to_high={_fmt(best_low)}."
            ),
            "5. Trade-off in MAE/QWK/Acc@5:",
        ]
    )
    for row in best_selection:
        base = row.get("base_model", "")
        raw = _raw_for(summary, base)
        lines.append(
            f"   - {base}: low_to_high {_fmt(raw.get('low_to_high_rate'))} -> {_fmt(row.get('low_to_high_rate'))}; "
            f"MAE {_fmt(raw.get('MAE_label'))} -> {_fmt(row.get('MAE_label'))}; "
            f"QWK {_fmt(raw.get('QWK'))} -> {_fmt(row.get('QWK'))}; "
            f"Acc@5 {_fmt(raw.get('Acc@5'))} -> {_fmt(row.get('Acc@5'))}."
        )
    lines.extend(
        [
            "6. Optional rejection analysis:",
            *_table(rejection, ["base_model", "review_rate", "coverage", "low_to_high_all", "low_to_high_non_rejected", "Acc@5_all", "Acc@5_non_rejected"]),
            f"7. Exp7-C thesis final method? {'YES, conditionally' if beats_qd_b1 else 'NOT YET'} under this unified synced-artifact run.",
            "8. Exp7-B should remain blocked until calibration trade-offs are accepted and documented.",
            "",
            "## Method Notes",
            "",
        ]
    )
    for base in ready:
        lines.append(f"- {base} temperature scaling: {_method_help(summary, base, 'temperature_scaling')}.")
        lines.append(f"- {base} global threshold: {_method_help(summary, base, 'global_threshold')}.")
        lines.append(f"- {base} risk-aware threshold: {_method_help(summary, base, 'risk_aware_threshold')}.")
    write_text(EXP07_CALIBRATION_REPORTS_DIR / "exp07_c2_unified_calibration_report.md", "\n".join(lines))

    review_lines = [
        "# Exp7-C2 Review Package",
        "",
        "- Scope: local calibration after syncing server artifacts.",
        "- Selection split: dev only.",
        "- Test split: final evaluation only.",
        "- Training run: NO.",
        "- API call: NO.",
        "- Synthetic data: NO.",
        "- Raw artifacts modified: NO.",
        "- Exp7-B: remains blocked.",
        f"- Best overall: `{best.get('base_model', 'NA')}` / `{best.get('method', 'NA')}`.",
        f"- Beats QD-B1 raw low_to_high: {'yes' if beats_qd_b1 else 'no'}.",
    ]
    write_text(EXP07_CALIBRATION_REPORTS_DIR / "exp07_c2_review_package.md", "\n".join(review_lines))

    notion_lines = [
        "# Exp7-C2 Unified Calibration Summary",
        "",
        f"- Calibrated bases: {', '.join(ready) if ready else 'none'}.",
        f"- Best overall: `{best.get('base_model', 'NA')}` / `{best.get('method', 'NA')}`.",
        f"- Best low_to_high: `{_fmt(best_low)}`.",
        f"- Beats QD-B1 raw low_to_high=0.4516: {'yes' if beats_qd_b1 else 'no'}.",
        "- Training/API/synthetic: no.",
        "- Exp7-B: remains blocked.",
    ]
    write_text(EXP07_CALIBRATION_REPORTS_DIR / "notion_exp07_c2_unified_calibration_summary.md", "\n".join(notion_lines))


def main() -> None:
    write_c2_report()
    print(f"Wrote Exp7-C2 reports: {EXP07_CALIBRATION_REPORTS_DIR}")


if __name__ == "__main__":
    main()
