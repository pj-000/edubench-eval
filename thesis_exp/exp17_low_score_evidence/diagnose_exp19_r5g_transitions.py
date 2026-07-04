"""Diagnose sample-level transitions for Exp19-R5G frontier runs.

This script compares existing dev raw predictions for R2c/R4b, R5F2 real-only,
R5G A3, and R5G B1. It does not train, does not read test, and writes only
lightweight CSV/MD diagnostics. Raw prediction files remain local/server-side.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence import collect_exp19_sft_first_round_dev_results as sft_collect  # noqa: E402
from thesis_exp.src.edujudge.utils.io import write_csv, write_json, write_text  # noqa: E402


DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_r5g_transition_diagnostics")
DEFAULT_REFERENCE = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp19_sft_dpo_datasets_seed42/tables/exp19_dev_reference.csv"
)
DEFAULT_D1_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/d1_hidden_failure_audit_seed42_dev/"
    "summary_human_rationale_recovered"
)
DEFAULT_SFT_PREDICTION_ROOT = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_sft_second_round/dev_predictions")
DEFAULT_R5F2_PREDICTION_ROOT = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_r5f2_dpo_scout/dev_predictions")
DEFAULT_R5G_PREDICTION_ROOT = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_r5g_dpo_scout/dev_predictions")

RUN_SPECS = {
    "r2c": ("r2c_clean_reason_score_balanced", DEFAULT_SFT_PREDICTION_ROOT),
    "r4b": ("r4b_shuffled_reason_balanced", DEFAULT_SFT_PREDICTION_ROOT),
    "r5f2_real": ("r5f2_real_only_from_r2c", DEFAULT_R5F2_PREDICTION_ROOT),
    "r5g_a3": ("r5g_a3_real_only_s50_b0p05_lr5em6", DEFAULT_R5G_PREDICTION_ROOT),
    "r5g_b1": ("r5g_b1_ratio70_30_s100_b0p03_lr5em6", DEFAULT_R5G_PREDICTION_ROOT),
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except Exception:
        return None


def resolve_roots(args: argparse.Namespace) -> dict[str, tuple[str, Path]]:
    return {
        "r2c": ("r2c_clean_reason_score_balanced", args.sft_prediction_root),
        "r4b": ("r4b_shuffled_reason_balanced", args.sft_prediction_root),
        "r5f2_real": ("r5f2_real_only_from_r2c", args.r5f2_prediction_root),
        "r5g_a3": ("r5g_a3_real_only_s50_b0p05_lr5em6", args.r5g_prediction_root),
        "r5g_b1": ("r5g_b1_ratio70_30_s100_b0p03_lr5em6", args.r5g_prediction_root),
    }


def load_run_predictions(
    reference: list[dict[str, str]],
    run_key: str,
    run_name: str,
    root: Path,
    allow_missing: bool,
) -> tuple[dict[str, dict[str, Any]], str | None]:
    run_dir = root / run_name
    try:
        prediction_file = sft_collect.find_prediction_file(run_dir)
    except FileNotFoundError:
        if allow_missing:
            return {}, run_name
        raise
    records = sft_collect.load_prediction_records(prediction_file)
    aligned = sft_collect.align_predictions(reference, records, run_key, run_name)
    return {str(row["sample_id"]): row for row in aligned}, None


def load_d1_hidden_ids(d1_dir: Path) -> set[str]:
    annotations, _pairs, _controls = sft_collect.load_d1_annotations(d1_dir)
    return set(annotations)


def pred(row_by_id: dict[str, dict[str, Any]], sample_id: str) -> int | None:
    row = row_by_id.get(sample_id)
    if not row:
        return None
    return safe_int(row.get("pred_label"))


def is_l2h(gold: int, value: int | None) -> bool:
    return gold <= 2 and value is not None and value >= 4


def is_h2l(gold: int, value: int | None) -> bool:
    return gold >= 4 and value is not None and value <= 2


def low_transition_rows(
    reference: list[dict[str, str]],
    predictions: dict[str, dict[str, dict[str, Any]]],
    d1_ids: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ref in reference:
        gold = int(ref["gold_label"])
        if gold > 2:
            continue
        sid = ref["sample_id"]
        p_r2c = pred(predictions.get("r2c", {}), sid)
        p_r4b = pred(predictions.get("r4b", {}), sid)
        p_f2 = pred(predictions.get("r5f2_real", {}), sid)
        p_a3 = pred(predictions.get("r5g_a3", {}), sid)
        p_b1 = pred(predictions.get("r5g_b1", {}), sid)
        before = is_l2h(gold, p_r2c)
        rows.append(
            {
                "sample_id": sid,
                "gold_label": gold,
                "metric": ref.get("metric", ""),
                "language": ref.get("language", ""),
                "is_d1_hidden": sid in d1_ids,
                "pred_r2c": p_r2c,
                "pred_r4b": p_r4b,
                "pred_r5f2_real": p_f2,
                "pred_r5g_a3": p_a3,
                "pred_r5g_b1": p_b1,
                "fixed_by_r5f2_real": before and not is_l2h(gold, p_f2),
                "fixed_by_a3": before and not is_l2h(gold, p_a3),
                "fixed_by_b1": before and not is_l2h(gold, p_b1),
                "worsened_by_a3": (not before) and is_l2h(gold, p_a3),
            }
        )
    return rows


def high_audit_rows(
    reference: list[dict[str, str]],
    predictions: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ref in reference:
        gold = int(ref["gold_label"])
        if gold < 4:
            continue
        sid = ref["sample_id"]
        p_r2c = pred(predictions.get("r2c", {}), sid)
        p_f2 = pred(predictions.get("r5f2_real", {}), sid)
        p_a3 = pred(predictions.get("r5g_a3", {}), sid)
        rows.append(
            {
                "sample_id": sid,
                "gold_label": gold,
                "metric": ref.get("metric", ""),
                "language": ref.get("language", ""),
                "pred_r2c": p_r2c,
                "pred_r5f2_real": p_f2,
                "pred_r5g_a3": p_a3,
                "high_to_low_r5f2_real": is_h2l(gold, p_f2),
                "high_to_low_a3": is_h2l(gold, p_a3),
            }
        )
    return rows


def count_true(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if bool(row.get(key)))


def cluster_counts(rows: list[dict[str, Any]], flag_key: str) -> list[dict[str, Any]]:
    counter: Counter[tuple[str, str]] = Counter()
    for row in rows:
        if bool(row.get(flag_key)):
            counter.update([(str(row.get("metric", "")), str(row.get("language", "")))])
    return [
        {"flag": flag_key, "metric": metric, "language": language, "n": n}
        for (metric, language), n in counter.most_common()
    ]


def write_report(
    out_dir: Path,
    low_rows: list[dict[str, Any]],
    high_rows: list[dict[str, Any]],
    missing: list[str],
) -> None:
    fixed_f2 = count_true(low_rows, "fixed_by_r5f2_real")
    fixed_a3 = count_true(low_rows, "fixed_by_a3")
    fixed_b1 = count_true(low_rows, "fixed_by_b1")
    d1_fixed_a3 = sum(1 for row in low_rows if row["is_d1_hidden"] and row["fixed_by_a3"])
    h2l_f2 = count_true(high_rows, "high_to_low_r5f2_real")
    h2l_a3 = count_true(high_rows, "high_to_low_a3")
    cluster_lines = []
    for row in (cluster_counts(high_rows, "high_to_low_r5f2_real") + cluster_counts(high_rows, "high_to_low_a3"))[:20]:
        cluster_lines.append(f"- {row['flag']} / {row['metric']} / {row['language']}: {row['n']}")
    if not cluster_lines:
        cluster_lines.append("- none")
    lines = [
        "# Exp19-R5G Transition Diagnostic Report",
        "",
        "This report compares dev predictions across R2c, R4b, R5F2 real-only, R5G A3, and R5G B1.",
        "",
        "## Low-Score Fixes",
        "",
        f"- low samples analyzed: {len(low_rows)}",
        f"- fixed_by_r5f2_real: {fixed_f2}",
        f"- fixed_by_a3: {fixed_a3}",
        f"- fixed_by_b1: {fixed_b1}",
        f"- fixed_by_a3 among D1 hidden: {d1_fixed_a3}",
        "",
        "## High-Score Harm",
        "",
        f"- high samples analyzed: {len(high_rows)}",
        f"- high_to_low_r5f2_real: {h2l_f2}",
        f"- high_to_low_a3: {h2l_a3}",
        "",
        "## Metric/Language Clusters",
        "",
        *cluster_lines,
        "",
        "## Interpretation",
        "",
        "- R5F2 real-only shows how many R2c low-to-high cases can be fixed by strong low-risk DPO.",
        "- R5G A3 shows which fixes survive a less over-conservative risk-calibrated setting.",
        "- High-to-low rows show where strong low-risk DPO harms true high-score samples.",
        "",
        "## Missing Prediction Runs",
        "",
        f"- {', '.join(missing) or 'none'}",
        "",
        "## Guardrails",
        "",
        "- This script reads dev predictions only.",
        "- It does not read test.",
        "- It does not train or write raw predictions.",
    ]
    write_text(out_dir / "reports" / "r5g_transition_diagnostic_report.md", "\n".join(lines))


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    reference = read_csv_rows(args.reference_csv)
    d1_ids = load_d1_hidden_ids(args.d1_dir)
    predictions: dict[str, dict[str, dict[str, Any]]] = {}
    missing: list[str] = []
    for key, (run_name, root) in resolve_roots(args).items():
        loaded, missing_run = load_run_predictions(reference, key, run_name, root, args.allow_missing_predictions)
        predictions[key] = loaded
        if missing_run:
            missing.append(missing_run)
    low_rows = low_transition_rows(reference, predictions, d1_ids)
    high_rows = high_audit_rows(reference, predictions)
    write_csv(
        args.out_dir / "tables" / "r5g_low_transition_analysis.csv",
        low_rows,
        [
            "sample_id",
            "gold_label",
            "metric",
            "language",
            "is_d1_hidden",
            "pred_r2c",
            "pred_r4b",
            "pred_r5f2_real",
            "pred_r5g_a3",
            "pred_r5g_b1",
            "fixed_by_r5f2_real",
            "fixed_by_a3",
            "fixed_by_b1",
            "worsened_by_a3",
        ],
    )
    write_csv(
        args.out_dir / "tables" / "r5g_high_to_low_audit.csv",
        high_rows,
        [
            "sample_id",
            "gold_label",
            "metric",
            "language",
            "pred_r2c",
            "pred_r5f2_real",
            "pred_r5g_a3",
            "high_to_low_r5f2_real",
            "high_to_low_a3",
        ],
    )
    write_report(args.out_dir, low_rows, high_rows, missing)
    decision = {
        "low_rows": len(low_rows),
        "high_rows": len(high_rows),
        "missing_prediction_runs": missing,
        "guardrails": {"no_test_read": True, "dev_only": True, "no_raw_predictions_written": True},
    }
    write_json(args.out_dir / "decision" / "r5g_transition_diagnostic_decision.json", decision)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Exp19-R5G sample-level transitions.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--d1-dir", type=Path, default=DEFAULT_D1_DIR)
    parser.add_argument("--sft-prediction-root", type=Path, default=DEFAULT_SFT_PREDICTION_ROOT)
    parser.add_argument("--r5f2-prediction-root", type=Path, default=DEFAULT_R5F2_PREDICTION_ROOT)
    parser.add_argument("--r5g-prediction-root", type=Path, default=DEFAULT_R5G_PREDICTION_ROOT)
    parser.add_argument("--allow-missing-predictions", action="store_true")
    args = parser.parse_args()
    print(json.dumps(diagnose(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
