"""Diagnose why Exp24 score-channel ORC-DPO did not reduce low-to-high.

This diagnostic reads existing dev prediction files only. It does not train,
does not read test, and writes lightweight CSV/MD/JSON outputs without raw
generations, checkpoints, logs, answers, or human rationales.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence import collect_exp19_sft_first_round_dev_results as sft_collect  # noqa: E402
from thesis_exp.src.edujudge.utils.io import write_csv, write_json, write_text  # noqa: E402


DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp24r_orc_failure_diagnosis_seed42")
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


RUN_SPECS = [
    ("r2c", "r2c_clean_reason_score_balanced", "baseline_sft", DEFAULT_SFT_PREDICTION_ROOT),
    ("r4b", "r4b_shuffled_reason_balanced", "baseline_sft", DEFAULT_SFT_PREDICTION_ROOT),
    ("r7d", "r7d_reason_real_s100_b0p03_lr5em6", "ordinary_reason_dpo", DEFAULT_EXP23_PREDICTION_ROOT),
    ("r7e", "r7e_matched_score_only_s100_b0p03_lr5em6", "ordinary_score_dpo", DEFAULT_EXP23_PREDICTION_ROOT),
    ("r7f", "r7f_score_reason_consistency_s100_b0p03_lr5em6", "ordinary_src_dpo", DEFAULT_EXP23_PREDICTION_ROOT),
    ("dpo0", "exp24_dpo0_r2c", "same_trainer_dpo0", DEFAULT_EXP24_PREDICTION_ROOT),
    ("orc_a", "exp24_orc_a_r2c", "orc", DEFAULT_EXP24_PREDICTION_ROOT),
    ("orc_b", "exp24_orc_b_r2c", "orc", DEFAULT_EXP24_PREDICTION_ROOT),
    ("orc_b_noreason", "exp24_orc_b_noreason_r2c", "orc_no_reason", DEFAULT_EXP24_PREDICTION_ROOT),
    ("orc_c", "exp24_orc_c_r2c", "orc", DEFAULT_EXP24_PREDICTION_ROOT),
]

FOCUS_RUNS = ["r2c", "r4b", "r7d", "r7e", "dpo0", "orc_b", "orc_b_noreason"]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        out = int(float(value))
        return out if 1 <= out <= 5 else None
    except Exception:
        return None


def safe_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def fmt(value: Any, digits: int = 4) -> str:
    val = safe_float(value)
    if math.isnan(val):
        return "nan"
    return f"{val:.{digits}f}"


def safe_rate(num: int, den: int) -> float:
    return num / den if den else float("nan")


def prediction_root_for(default_root: Path, args: argparse.Namespace) -> Path:
    if default_root == DEFAULT_SFT_PREDICTION_ROOT:
        return args.sft_prediction_root
    if default_root == DEFAULT_EXP23_PREDICTION_ROOT:
        return args.exp23_prediction_root
    if default_root == DEFAULT_EXP24_PREDICTION_ROOT:
        return args.exp24_prediction_root
    return default_root


def load_predictions(
    reference: list[dict[str, str]],
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, dict[str, Any]]], list[str]]:
    predictions: dict[str, dict[str, dict[str, Any]]] = {}
    missing: list[str] = []
    for short_name, run_name, _family, default_root in RUN_SPECS:
        root = prediction_root_for(default_root, args)
        run_dir = root / run_name
        try:
            prediction_file = sft_collect.find_prediction_file(run_dir)
        except FileNotFoundError:
            missing.append(run_name)
            if args.allow_missing_predictions:
                predictions[short_name] = {}
                continue
            raise
        records = sft_collect.load_prediction_records(prediction_file)
        aligned = sft_collect.align_predictions(reference, records, short_name, run_name)
        predictions[short_name] = {str(row["sample_id"]): row for row in aligned}
    return predictions, missing


def pred(predictions: dict[str, dict[str, dict[str, Any]]], run: str, sample_id: str) -> int | None:
    row = predictions.get(run, {}).get(sample_id)
    if not row:
        return None
    return safe_int(row.get("pred_label"))


def is_low_to_high(gold: int, p: int | None) -> bool:
    return gold <= 2 and p is not None and p >= 4


def is_high_to_low(gold: int, p: int | None) -> bool:
    return gold >= 4 and p is not None and p <= 2


def load_d1_ids(d1_dir: Path) -> tuple[set[str], set[str]]:
    annotations, _pairs, controls = sft_collect.load_d1_annotations(d1_dir)
    return set(annotations), set(controls)


def label_distribution_rows(
    reference: list[dict[str, str]],
    predictions: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for short_name, run_name, family, _root in RUN_SPECS:
        pred_by_id = predictions.get(short_name, {})
        if not pred_by_id:
            continue
        for gold in range(1, 6):
            refs = [row for row in reference if int(row["gold_label"]) == gold]
            preds = [pred(predictions, short_name, row["sample_id"]) for row in refs]
            valid = [p for p in preds if p is not None]
            row: dict[str, Any] = {
                "run_key": short_name,
                "run_name": run_name,
                "run_family": family,
                "gold_label": gold,
                "n": len(refs),
                "valid_n": len(valid),
                "mean_pred": sum(valid) / len(valid) if valid else float("nan"),
                "exact_rate": safe_rate(sum(1 for p in preds if p == gold), len(refs)),
                "pred_ge4_rate": safe_rate(sum(1 for p in preds if p is not None and p >= 4), len(refs)),
                "pred_le2_rate": safe_rate(sum(1 for p in preds if p is not None and p <= 2), len(refs)),
            }
            for p in range(1, 6):
                row[f"pred_{p}_rate"] = safe_rate(sum(1 for value in preds if value == p), len(refs))
            out.append(row)
    return out


def low_transition_rows(
    reference: list[dict[str, str]],
    predictions: dict[str, dict[str, dict[str, Any]]],
    d1_hidden_ids: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ref in reference:
        gold = int(ref["gold_label"])
        if gold > 2:
            continue
        sid = ref["sample_id"]
        row: dict[str, Any] = {
            "sample_id": sid,
            "question_key": ref.get("question_key", ""),
            "question_group_id": ref.get("question_group_id", ""),
            "metric": ref.get("metric", ""),
            "language": ref.get("language", ""),
            "subject": ref.get("subject", ""),
            "gold_label": gold,
            "is_d1_hidden": sid in d1_hidden_ids,
        }
        for run in FOCUS_RUNS:
            row[f"pred_{run}"] = pred(predictions, run, sid)
            row[f"low_to_high_{run}"] = is_low_to_high(gold, row[f"pred_{run}"])
        row["dpo0_changed_vs_r2c"] = row.get("pred_dpo0") != row.get("pred_r2c")
        row["orc_b_changed_vs_dpo0"] = row.get("pred_orc_b") != row.get("pred_dpo0")
        row["r7d_fixed_r2c_l2h"] = bool(row.get("low_to_high_r2c")) and not bool(row.get("low_to_high_r7d"))
        row["dpo0_fixed_r2c_l2h"] = bool(row.get("low_to_high_r2c")) and not bool(row.get("low_to_high_dpo0"))
        row["orc_b_fixed_dpo0_l2h"] = bool(row.get("low_to_high_dpo0")) and not bool(row.get("low_to_high_orc_b"))
        row["orc_b_worsened_vs_dpo0"] = (not bool(row.get("low_to_high_dpo0"))) and bool(row.get("low_to_high_orc_b"))
        rows.append(row)
    return rows


def d1_transition_rows(
    reference: list[dict[str, str]],
    predictions: dict[str, dict[str, dict[str, Any]]],
    d1_hidden_ids: set[str],
) -> list[dict[str, Any]]:
    ref_by_id = {row["sample_id"]: row for row in reference}
    rows: list[dict[str, Any]] = []
    for sid in sorted(d1_hidden_ids):
        ref = ref_by_id.get(sid)
        if not ref:
            continue
        gold = int(ref["gold_label"])
        row: dict[str, Any] = {
            "sample_id": sid,
            "question_key": ref.get("question_key", ""),
            "question_group_id": ref.get("question_group_id", ""),
            "metric": ref.get("metric", ""),
            "language": ref.get("language", ""),
            "subject": ref.get("subject", ""),
            "gold_label": gold,
        }
        for run in FOCUS_RUNS:
            row[f"pred_{run}"] = pred(predictions, run, sid)
            row[f"pred_ge4_{run}"] = row[f"pred_{run}"] is not None and int(row[f"pred_{run}"]) >= 4
        row["r7d_reduced_vs_r2c"] = (
            row.get("pred_r7d") is not None
            and row.get("pred_r2c") is not None
            and int(row["pred_r7d"]) < int(row["pred_r2c"])
        )
        row["orc_b_reduced_vs_dpo0"] = (
            row.get("pred_orc_b") is not None
            and row.get("pred_dpo0") is not None
            and int(row["pred_orc_b"]) < int(row["pred_dpo0"])
        )
        rows.append(row)
    return rows


def pair_effect_rows(
    reference: list[dict[str, str]],
    predictions: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    comparisons = [
        ("dpo0_vs_r2c", "r2c", "dpo0"),
        ("orc_b_vs_dpo0", "dpo0", "orc_b"),
        ("orc_b_noreason_vs_dpo0", "dpo0", "orc_b_noreason"),
        ("orc_b_vs_r7e", "r7e", "orc_b"),
        ("r7d_vs_r7e", "r7e", "r7d"),
    ]
    rows: list[dict[str, Any]] = []
    for name, base, target in comparisons:
        n = changed = low_fixed = low_worse = high_harm_reduced = high_harm_added = 0
        total_abs_delta = 0
        for ref in reference:
            gold = int(ref["gold_label"])
            sid = ref["sample_id"]
            p0 = pred(predictions, base, sid)
            p1 = pred(predictions, target, sid)
            if p0 is None or p1 is None:
                continue
            n += 1
            changed += int(p0 != p1)
            total_abs_delta += abs(p1 - p0)
            low_fixed += int(is_low_to_high(gold, p0) and not is_low_to_high(gold, p1))
            low_worse += int((not is_low_to_high(gold, p0)) and is_low_to_high(gold, p1))
            high_harm_reduced += int(is_high_to_low(gold, p0) and not is_high_to_low(gold, p1))
            high_harm_added += int((not is_high_to_low(gold, p0)) and is_high_to_low(gold, p1))
        rows.append(
            {
                "comparison": name,
                "base_run": base,
                "target_run": target,
                "n": n,
                "changed_count": changed,
                "changed_rate": safe_rate(changed, n),
                "mean_abs_pred_delta": safe_rate(total_abs_delta, n),
                "low_to_high_fixed_count": low_fixed,
                "low_to_high_worsened_count": low_worse,
                "high_to_low_reduced_count": high_harm_reduced,
                "high_to_low_added_count": high_harm_added,
            }
        )
    return rows


def metric_language_rows(
    low_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in low_rows:
        groups[(str(row.get("metric", "")), str(row.get("language", "")))].append(row)
    out: list[dict[str, Any]] = []
    for (metric, language), rows in sorted(groups.items()):
        item: dict[str, Any] = {"metric": metric, "language": language, "low_n": len(rows)}
        for run in ["r2c", "r4b", "r7d", "r7e", "dpo0", "orc_b"]:
            item[f"low_to_high_{run}"] = safe_rate(sum(1 for row in rows if row.get(f"low_to_high_{run}")), len(rows))
        out.append(item)
    return out


def write_report(
    out_dir: Path,
    label_rows: list[dict[str, Any]],
    low_rows: list[dict[str, Any]],
    d1_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    missing: list[str],
) -> None:
    by_label = {(row["run_key"], int(row["gold_label"])): row for row in label_rows}
    def label_metric(run: str, label: int, field: str) -> Any:
        return by_label.get((run, label), {}).get(field, "")

    pair_by_name = {row["comparison"]: row for row in pair_rows}
    dpo0_vs_r2c = pair_by_name.get("dpo0_vs_r2c", {})
    orc_vs_dpo0 = pair_by_name.get("orc_b_vs_dpo0", {})
    r7d_vs_r7e = pair_by_name.get("r7d_vs_r7e", {})
    d1_orc_reduced = sum(1 for row in d1_rows if row.get("orc_b_reduced_vs_dpo0"))
    d1_r7d_reduced = sum(1 for row in d1_rows if row.get("r7d_reduced_vs_r2c"))
    lines = [
        "# Exp24R ORC-DPO Failure Diagnosis",
        "",
        "This diagnosis explains why Exp24 score-channel ORC-DPO did not pass the dev success rule.",
        "It uses existing dev predictions only and writes no raw generations.",
        "",
        "## Main Findings",
        "",
        f"- Low dev samples analyzed: {len(low_rows)}.",
        f"- D1 hidden cases analyzed: {len(d1_rows)}.",
        f"- DPO0 vs R2C changed predictions for {dpo0_vs_r2c.get('changed_count', '')} samples "
        f"({fmt(dpo0_vs_r2c.get('changed_rate'))}).",
        f"- ORC-B vs DPO0 changed predictions for {orc_vs_dpo0.get('changed_count', '')} samples "
        f"({fmt(orc_vs_dpo0.get('changed_rate'))}).",
        f"- ORC-B fixed {orc_vs_dpo0.get('low_to_high_fixed_count', '')} DPO0 low-to-high cases and worsened "
        f"{orc_vs_dpo0.get('low_to_high_worsened_count', '')}.",
        f"- R7D vs R7E fixed {r7d_vs_r7e.get('low_to_high_fixed_count', '')} low-to-high cases.",
        f"- ORC-B reduced D1 hidden predictions vs DPO0 for {d1_orc_reduced}/{len(d1_rows)} cases.",
        f"- R7D reduced D1 hidden predictions vs R2C for {d1_r7d_reduced}/{len(d1_rows)} cases.",
        "",
        "## Label-2 Distribution",
        "",
        "| run | n | mean pred | pred>=4 | exact label2 recall | pred 5 rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for run in ["r2c", "r4b", "r7d", "r7e", "dpo0", "orc_b", "orc_b_noreason"]:
        lines.append(
            f"| `{run}` | {label_metric(run, 2, 'n')} | {fmt(label_metric(run, 2, 'mean_pred'))} | "
            f"{fmt(label_metric(run, 2, 'pred_ge4_rate'))} | {fmt(label_metric(run, 2, 'exact_rate'))} | "
            f"{fmt(label_metric(run, 2, 'pred_5_rate'))} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Same-trainer DPO0 moved the model toward high-score outputs and increased low-to-high relative to R2C.",
            "- ORC-B's weights/margins changed too few predictions relative to DPO0 to repair low-to-high.",
            "- Human-reason ordinary DPO (R7D) is the only compared run that materially lowers D1 hidden predictions, but it hurts MAE/QWK.",
            "- Therefore the bottleneck is not parse or trainer failure; it is the score-only preference signal being too weak for hidden low-score evidence.",
            "",
            "## Recommended Next Step",
            "",
            "Do not run test or multi-seed Exp24 yet. Use this diagnosis to design SRC-DPO or hidden-failure data expansion:",
            "",
            "1. SRC-DPO: contrast reason-score consistency negatives, not only score-only negatives.",
            "2. Train-only hidden-failure expansion: add more low-to-high/D1-like cases and high-score protection controls.",
            "3. Optionally add an explicit low-tail term penalizing P(score>=4 | gold<=2) if using token probabilities.",
            "",
            "## Missing Runs",
            "",
            f"- {', '.join(missing) or 'none'}",
            "",
            "## Guardrails",
            "",
            "- No test split is read.",
            "- Dev labels are used for diagnosis only.",
            "- Raw predictions are read but not written to the output directory.",
            "- Output files contain IDs and scalar predictions only, not answers, raw generations, or human rationales.",
        ]
    )
    write_text(out_dir / "reports" / "exp24r_orc_failure_diagnosis_report.md", "\n".join(lines))


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    reference = read_csv_rows(args.reference_csv)
    d1_hidden_ids, _control_ids = load_d1_ids(args.d1_dir)
    predictions, missing = load_predictions(reference, args)
    label_rows = label_distribution_rows(reference, predictions)
    low_rows = low_transition_rows(reference, predictions, d1_hidden_ids)
    d1_rows = d1_transition_rows(reference, predictions, d1_hidden_ids)
    pair_rows = pair_effect_rows(reference, predictions)
    group_rows = metric_language_rows(low_rows)

    write_csv(args.out_dir / "tables" / "exp24r_label_prediction_distribution.csv", label_rows)
    write_csv(args.out_dir / "tables" / "exp24r_low_score_transitions.csv", low_rows)
    write_csv(args.out_dir / "tables" / "exp24r_d1_hidden_transitions.csv", d1_rows)
    write_csv(args.out_dir / "tables" / "exp24r_pair_effect_summary.csv", pair_rows)
    write_csv(args.out_dir / "tables" / "exp24r_metric_language_breakdown.csv", group_rows)
    write_report(args.out_dir, label_rows, low_rows, d1_rows, pair_rows, missing)

    decision = {
        "recommendation": "do_not_rerun_exp24_as_is",
        "next_step": "src_dpo_or_train_only_hidden_failure_expansion",
        "reason": "ORC-B changes too few DPO0 low-score predictions and does not reduce low-to-high.",
        "missing_prediction_runs": missing,
        "dev_rows_read": len(reference),
        "d1_hidden_cases": len(d1_rows),
        "low_score_cases": len(low_rows),
        "test_read": False,
        "raw_predictions_written": False,
        "raw_answers_or_reasons_written": False,
    }
    write_json(args.out_dir / "decision" / "exp24r_orc_failure_diagnosis_decision.json", decision)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Exp24 ORC-DPO failure modes.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--d1-dir", type=Path, default=DEFAULT_D1_DIR)
    parser.add_argument("--sft-prediction-root", type=Path, default=DEFAULT_SFT_PREDICTION_ROOT)
    parser.add_argument("--exp23-prediction-root", type=Path, default=DEFAULT_EXP23_PREDICTION_ROOT)
    parser.add_argument("--exp24-prediction-root", type=Path, default=DEFAULT_EXP24_PREDICTION_ROOT)
    parser.add_argument("--allow-missing-predictions", action="store_true")
    args = parser.parse_args()
    print(json.dumps(diagnose(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
