"""Diagnose Exp17-C0 outcome transitions and boundary drift."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.src.edujudge.utils.io import read_jsonl, relpath, write_csv, write_json, write_text


DEFAULT_C0_OUTPUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp17_c0_pairwise_separation_seed42")
DEFAULT_DEV_JSONL = Path("thesis_exp/data/splits/question_seed42/dev.jsonl")
DEFAULT_D1_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/d1_hidden_failure_audit_seed42_dev/summary_human_rationale_recovered")
DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp17_c0_d1_diagnostics_seed42")

CONFIGS = {
    "c0_0": "C0_0_ordinal_continue",
    "c0_2": "C0_2_all_pairs_gamma0p05_m0p2",
    "c0_5": "C0_5_evidence_positive_plus_pairwise_low_gamma0p05_m0p2",
    "c0_6": "C0_6_random_pair_control_gamma0p05_m0p2",
    "c0_12": "C0_12_random_matched_metric_rubric_gamma0p05_m0p2",
    "c0_14": "C0_14_same_question_group_upper_bound_gamma0p05_m0p2",
}

TRANSITION_FIELDS = [
    "sample_id",
    "gold_label",
    "metric",
    "question_key",
    "pred_exp16a_init",
    "pred_c0_0",
    "pred_c0_2",
    "pred_c0_5",
    "pred_c0_6",
    "pred_c0_12",
    "pred_c0_14",
    "s_exp16a_init",
    "s_c0_0",
    "s_c0_2",
    "tau3_exp16a_init",
    "tau3_c0_0",
    "tau3_c0_2",
    "alpha_exp16a_init",
    "alpha_c0_0",
    "alpha_c0_2",
    "g_i3_exp16a_init",
    "g_i3_c0_0",
    "g_i3_c0_2",
    "is_low_to_high_exp16a_init",
    "is_low_to_high_c0_0",
    "is_low_to_high_c0_2",
    "transition_type",
    "is_d1_hidden",
    "d1_failure_mode",
]

DRIFT_FIELDS = [
    "config",
    "gold_label",
    "n",
    "mean_s",
    "median_s",
    "mean_tau2",
    "mean_tau3",
    "mean_tau4",
    "mean_alpha",
    "mean_margin_tau3",
    "median_margin_tau3",
    "mean_g_i3",
    "median_g_i3",
    "rate_g_i3_gt0",
    "pred_ge4_rate",
    "recall",
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


def safe_int(value: Any) -> int | None:
    try:
        if value in {"", None}:
            return None
        return int(float(value))
    except Exception:
        return None


def stable_hash(value: Any) -> str:
    return hashlib.sha1(str(value or "").strip().encode("utf-8")).hexdigest()


def q(values: list[float], pct: float) -> float:
    values = [value for value in values if not math.isnan(value)]
    return float(np.percentile(values, pct)) if values else float("nan")


def mean(values: list[float]) -> float:
    values = [value for value in values if not math.isnan(value)]
    return float(np.mean(values)) if values else float("nan")


def load_dev_meta(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        sample_id = str(row.get("record_id") or row.get("sample_id") or row.get("id") or "")
        rubric = row.get("rubric_text") or row.get("rubric") or row.get("rubric_canonical") or ""
        out[sample_id] = {
            "sample_id": sample_id,
            "question_key": row.get("question_key") or "",
            "metric": row.get("metric_canonical") or row.get("metric") or row.get("metric_abbr") or "",
            "rubric_hash": stable_hash(rubric),
            "gold_label": int(row.get("label_5", row.get("label", 0))),
        }
    return out


def normalize_prediction_row(row: dict[str, Any], meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sample_id = str(row.get("sample_id") or row.get("record_id") or "")
    gold = safe_int(row.get("gold_label") or row.get("label") or row.get("label_5"))
    pred = safe_int(row.get("pred_label"))
    s = safe_float(row.get("quality_score_s"))
    tau3 = safe_float(row.get("tau3"))
    alpha = safe_float(row.get("scale_alpha"))
    g_i3 = safe_float(row.get("g_i3"))
    if math.isnan(g_i3) and not math.isnan(s) and not math.isnan(tau3) and not math.isnan(alpha):
        g_i3 = alpha * (s - tau3)
    metadata = meta.get(sample_id, {})
    return {
        "sample_id": sample_id,
        "gold_label": gold if gold is not None else metadata.get("gold_label", ""),
        "pred_label": pred if pred is not None else "",
        "quality_score_s": s,
        "tau1": safe_float(row.get("tau1")),
        "tau2": safe_float(row.get("tau2")),
        "tau3": tau3,
        "tau4": safe_float(row.get("tau4")),
        "scale_alpha": alpha,
        "g_i3": g_i3,
        "margin_tau3": safe_float(row.get("margin_tau3")),
        "question_key": row.get("question_key") or metadata.get("question_key", ""),
        "metric": row.get("metric") or metadata.get("metric", ""),
        "rubric_hash": row.get("rubric_hash") or metadata.get("rubric_hash", ""),
        "boundary_key": row.get("boundary_key", ""),
    }


def load_predictions(path: Path | None, meta: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], str]:
    if path is None or not path.exists():
        return {}, f"missing predictions: {path}" if path else "missing predictions path"
    rows: list[dict[str, Any]]
    if path.suffix == ".jsonl":
        rows = read_jsonl(path)
    else:
        rows = read_csv_rows(path)
    out = {row["sample_id"]: row for row in (normalize_prediction_row(row, meta) for row in rows) if row.get("sample_id")}
    return out, ""


def diagnostic_path(c0_output_dir: Path, config: str, seed: int) -> Path:
    return c0_output_dir / "runs" / config / f"seed_{seed}" / "diagnostic_predictions_dev.csv"


def d1_root(path: Path) -> Path:
    return path.parent if path.name.startswith("summary") else path


def load_d1_annotations(path: Path) -> dict[str, dict[str, str]]:
    root = d1_root(path)
    for name in [
        "d1_hidden_failure_annotation_template_filled_human_rationale_recovered.csv",
        "d1_hidden_failure_annotation_template_filled_codex_independent.csv",
        "d1_hidden_failure_annotation_template_filled.csv",
    ]:
        candidate = root / name
        if candidate.exists():
            return {row.get("sample_id", ""): row for row in read_csv_rows(candidate)}
    return {}


def load_d1_pairs(path: Path) -> list[tuple[str, str]]:
    root = d1_root(path)
    candidate = root / "d1_matched_case_control_review.csv"
    if not candidate.exists():
        return []
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in read_csv_rows(candidate):
        pair = (row.get("case_sample_id", ""), row.get("control_sample_id", ""))
        if all(pair) and pair not in seen:
            pairs.append(pair)
            seen.add(pair)
    return pairs


def is_low_to_high(row: dict[str, Any]) -> bool:
    gold = safe_int(row.get("gold_label"))
    pred = safe_int(row.get("pred_label"))
    return bool(gold is not None and pred is not None and gold <= 2 and pred >= 4)


def transition_type(init: dict[str, Any], c0_0: dict[str, Any], c0_2: dict[str, Any]) -> str:
    del init
    before = is_low_to_high(c0_0)
    after = is_low_to_high(c0_2)
    gold = safe_int(c0_2.get("gold_label") or c0_0.get("gold_label"))
    pred = safe_int(c0_2.get("pred_label"))
    if before and not after:
        return "fixed_low_to_high"
    if not before and after:
        return "worsened_to_low_to_high"
    if before and after:
        return "unchanged_low_to_high"
    if gold is not None and pred == gold:
        return "unchanged_correct"
    return "other"


def build_transition_rows(preds: dict[str, dict[str, dict[str, Any]]], meta: dict[str, dict[str, Any]], d1: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sample_ids = sorted(set(meta) | set().union(*(set(items) for items in preds.values())) if preds else set(meta))
    for sample_id in sample_ids:
        base = meta.get(sample_id, {})
        exp16 = preds.get("exp16a_init", {}).get(sample_id, {})
        c0_0 = preds.get("c0_0", {}).get(sample_id, {})
        c0_2 = preds.get("c0_2", {}).get(sample_id, {})
        annotation = d1.get(sample_id, {})
        row = {
            "sample_id": sample_id,
            "gold_label": base.get("gold_label") or c0_2.get("gold_label") or c0_0.get("gold_label") or exp16.get("gold_label"),
            "metric": base.get("metric") or c0_2.get("metric") or c0_0.get("metric") or exp16.get("metric"),
            "question_key": base.get("question_key") or c0_2.get("question_key") or c0_0.get("question_key") or exp16.get("question_key"),
            "transition_type": transition_type(exp16, c0_0, c0_2) if c0_0 and c0_2 else "missing_predictions",
            "is_d1_hidden": bool(annotation),
            "d1_failure_mode": annotation.get("primary_failure_mode_manual", ""),
        }
        for name in ["exp16a_init", "c0_0", "c0_2", "c0_5", "c0_6", "c0_12", "c0_14"]:
            pred = preds.get(name, {}).get(sample_id, {})
            suffix = name
            row[f"pred_{suffix}"] = pred.get("pred_label", "")
            if name in {"exp16a_init", "c0_0", "c0_2"}:
                row[f"s_{suffix}"] = pred.get("quality_score_s", "")
                row[f"tau3_{suffix}"] = pred.get("tau3", "")
                row[f"alpha_{suffix}"] = pred.get("scale_alpha", "")
                row[f"g_i3_{suffix}"] = pred.get("g_i3", "")
                row[f"is_low_to_high_{suffix}"] = is_low_to_high(pred) if pred else ""
        rows.append(row)
    return rows


def boundary_drift_rows(preds: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config, by_id in preds.items():
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in by_id.values():
            gold = safe_int(row.get("gold_label"))
            if gold is not None:
                grouped[gold].append(row)
        for label, items in sorted(grouped.items()):
            g_values = [safe_float(row.get("g_i3")) for row in items]
            pred_labels = [safe_int(row.get("pred_label")) for row in items]
            rows.append(
                {
                    "config": config,
                    "gold_label": label,
                    "n": len(items),
                    "mean_s": mean([safe_float(row.get("quality_score_s")) for row in items]),
                    "median_s": q([safe_float(row.get("quality_score_s")) for row in items], 50),
                    "mean_tau2": mean([safe_float(row.get("tau2")) for row in items]),
                    "mean_tau3": mean([safe_float(row.get("tau3")) for row in items]),
                    "mean_tau4": mean([safe_float(row.get("tau4")) for row in items]),
                    "mean_alpha": mean([safe_float(row.get("scale_alpha")) for row in items]),
                    "mean_margin_tau3": mean([safe_float(row.get("margin_tau3")) for row in items]),
                    "median_margin_tau3": q([safe_float(row.get("margin_tau3")) for row in items], 50),
                    "mean_g_i3": mean(g_values),
                    "median_g_i3": q(g_values, 50),
                    "rate_g_i3_gt0": sum(1 for value in g_values if not math.isnan(value) and value > 0) / len(g_values) if g_values else float("nan"),
                    "pred_ge4_rate": sum(1 for value in pred_labels if value is not None and value >= 4) / len(pred_labels) if pred_labels else float("nan"),
                    "recall": sum(1 for value in pred_labels if value == label) / len(pred_labels) if pred_labels else float("nan"),
                }
            )
    return rows


def bootstrap_delta(preds: dict[str, dict[str, dict[str, Any]]], left: str, right: str, n_bootstrap: int, seed: int) -> dict[str, Any]:
    left_rows = preds.get(left, {})
    right_rows = preds.get(right, {})
    sample_ids = [sid for sid, row in right_rows.items() if safe_int(row.get("gold_label")) in {1, 2} and sid in left_rows]
    if not sample_ids:
        return {"comparison": f"{right} vs {left}", "mean_delta": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "n_bootstrap": n_bootstrap}
    deltas = []
    rng = random.Random(seed)
    for _ in range(n_bootstrap):
        draw = [rng.choice(sample_ids) for _ in sample_ids]
        left_rate = sum(1 for sid in draw if is_low_to_high(left_rows[sid])) / len(draw)
        right_rate = sum(1 for sid in draw if is_low_to_high(right_rows[sid])) / len(draw)
        deltas.append(right_rate - left_rate)
    return {
        "comparison": f"{right} vs {left}",
        "mean_delta": mean(deltas),
        "ci95_low": q(deltas, 2.5),
        "ci95_high": q(deltas, 97.5),
        "n_bootstrap": n_bootstrap,
    }


def auc_control_gt_hidden(hidden: list[float], control: list[float]) -> float:
    if not hidden or not control:
        return float("nan")
    wins = 0.0
    total = 0
    for h in hidden:
        for c in control:
            total += 1
            if c > h:
                wins += 1.0
            elif c == h:
                wins += 0.5
    return wins / total if total else float("nan")


def d1_gap_rows(preds: dict[str, dict[str, dict[str, Any]]], pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for config, by_id in preds.items():
        gaps = []
        hidden_s = []
        control_s = []
        hidden_g = []
        control_g = []
        for hidden_id, control_id in pairs:
            if hidden_id not in by_id or control_id not in by_id:
                continue
            h = by_id[hidden_id]
            c = by_id[control_id]
            hs = safe_float(h.get("quality_score_s"))
            cs = safe_float(c.get("quality_score_s"))
            gaps.append(cs - hs)
            hidden_s.append(hs)
            control_s.append(cs)
            hidden_g.append(safe_float(h.get("g_i3")))
            control_g.append(safe_float(c.get("g_i3")))
        rows.append(
            {
                "config": config,
                "n_pairs": len(gaps),
                "mean_s_gap_control_minus_hidden": mean(gaps),
                "median_s_gap_control_minus_hidden": q(gaps, 50),
                "auc_hidden_vs_control_s": auc_control_gt_hidden(hidden_s, control_s),
                "mean_g_i3_hidden": mean(hidden_g),
                "mean_g_i3_control": mean(control_g),
                "violation_rate_control_s_le_hidden_s": sum(1 for gap in gaps if gap <= 0) / len(gaps) if gaps else float("nan"),
            }
        )
    return rows


def write_report(out_dir: Path, warnings: list[str], transition_rows: list[dict[str, Any]], bootstrap_rows: list[dict[str, Any]], d1_rows: list[dict[str, Any]]) -> None:
    fixed = [row for row in transition_rows if row.get("transition_type") == "fixed_low_to_high"]
    fixed_labels = {str(row.get("gold_label")): 0 for row in fixed}
    for row in fixed:
        fixed_labels[str(row.get("gold_label"))] = fixed_labels.get(str(row.get("gold_label")), 0) + 1
    fixed_ids = ", ".join(str(row["sample_id"]) for row in fixed[:20]) if fixed else "none"
    c0_2_boot = next((row for row in bootstrap_rows if row["comparison"] == "c0_2 vs c0_0"), {})
    c0_2_d1 = next((row for row in d1_rows if row["config"] == "c0_2"), {})
    lines = [
        "# Exp17-C0 D1 Diagnostic Report",
        "",
        "Default decision: `C0_weak_risk_signal_needs_c0b`.",
        "",
        "## Warnings",
        "",
    ]
    lines.extend([f"- {warning}" for warning in warnings] or ["- none"])
    lines.extend(
        [
            "",
            "## Transition Summary",
            "",
            f"- fixed_low_to_high_count_c0_2_vs_c0_0: {len(fixed)}",
            f"- fixed sample ids: {fixed_ids}",
            f"- fixed labels: {json.dumps(fixed_labels, ensure_ascii=False, sort_keys=True)}",
            f"- fixed D1 hidden count: {sum(1 for row in fixed if str(row.get('is_d1_hidden')).lower() == 'true')}",
            "",
            "## Bootstrap",
            "",
            f"- C0_2 vs C0_0 mean_delta: {c0_2_boot.get('mean_delta', 'NA')}",
            f"- C0_2 vs C0_0 95% CI: [{c0_2_boot.get('ci95_low', 'NA')}, {c0_2_boot.get('ci95_high', 'NA')}]",
            "",
            "## D1 Gap",
            "",
            f"- C0_2 mean_s_gap_control_minus_hidden: {c0_2_d1.get('mean_s_gap_control_minus_hidden', 'NA')}",
            f"- C0_2 auc_hidden_vs_control_s: {c0_2_d1.get('auc_hidden_vs_control_s', 'NA')}",
            "",
            "## Answers",
            "",
            "- Which exact low samples were fixed by C0_2? See `c0_transition_analysis.csv` rows with `transition_type=fixed_low_to_high`.",
            "- Are fixed samples label1 or label2? See the fixed labels summary above.",
            "- Did they move to 3 or 2? See `pred_c0_2` in `c0_transition_analysis.csv`.",
            "- Are they D1 hidden cases? See `is_d1_hidden` and `d1_failure_mode`.",
            "- Did random controls fix the same samples? Compare `pred_c0_6` and `pred_c0_12` for the fixed rows.",
            "- Did tau3 or alpha drift explain g_i3 changes? See `c0_boundary_drift_by_label.csv`.",
            "- Is C0_2 improvement significant under bootstrap? Use `c0_low_sample_bootstrap.csv`; if CI crosses 0, treat as weak.",
            "- Does C0_2 improve relative to original Exp16A init? Only if Exp16A init predictions were supplied.",
            "- Should C1 remain blocked? `yes`, until C0b resolves the latent diagnostics.",
            "- Should B1 remain blocked? `yes`.",
        ]
    )
    write_text(out_dir / "exp17_c0_d1_diagnostic_report.md", "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Exp17-C0 weak risk signal and latent separation.")
    parser.add_argument("--c0-output-dir", type=Path, default=DEFAULT_C0_OUTPUT_DIR)
    parser.add_argument("--exp16a-init-predictions", type=Path, default=None)
    parser.add_argument("--dev-jsonl", type=Path, default=DEFAULT_DEV_JSONL)
    parser.add_argument("--d1-dir", type=Path, default=DEFAULT_D1_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()

    meta = load_dev_meta(args.dev_jsonl)
    d1 = load_d1_annotations(args.d1_dir)
    pairs = load_d1_pairs(args.d1_dir)
    warnings: list[str] = []
    preds: dict[str, dict[str, dict[str, Any]]] = {}
    if args.exp16a_init_predictions:
        init_preds, warning = load_predictions(args.exp16a_init_predictions, meta)
        if warning:
            warnings.append(warning)
        else:
            preds["exp16a_init"] = init_preds
    else:
        warnings.append("Exp16A init predictions not supplied; Exp16A-relative rows will be blank.")
    for short_name, config in CONFIGS.items():
        path = diagnostic_path(args.c0_output_dir, config, int(args.seed))
        loaded, warning = load_predictions(path, meta)
        if warning:
            warnings.append(f"{short_name}: {warning}; rerun/export lightweight diagnostic predictions for full C0-D1 analysis.")
        else:
            preds[short_name] = loaded

    args.out_dir.mkdir(parents=True, exist_ok=True)
    transition_rows = build_transition_rows(preds, meta, d1)
    drift_rows = boundary_drift_rows(preds)
    bootstrap_rows = [
        bootstrap_delta(preds, "c0_0", "c0_2", int(args.n_bootstrap), int(args.seed)),
        bootstrap_delta(preds, "exp16a_init", "c0_2", int(args.n_bootstrap), int(args.seed)),
        bootstrap_delta(preds, "c0_6", "c0_2", int(args.n_bootstrap), int(args.seed)),
        bootstrap_delta(preds, "c0_12", "c0_2", int(args.n_bootstrap), int(args.seed)),
    ]
    d1_rows = d1_gap_rows(preds, pairs)

    write_csv(args.out_dir / "c0_transition_analysis.csv", transition_rows, fieldnames=TRANSITION_FIELDS)
    write_csv(args.out_dir / "c0_boundary_drift_by_label.csv", drift_rows, fieldnames=DRIFT_FIELDS)
    write_csv(args.out_dir / "c0_low_sample_bootstrap.csv", bootstrap_rows)
    write_csv(args.out_dir / "c0_d1_gap_diagnostics.csv", d1_rows)
    write_json(args.out_dir / "c0_d1_diagnostic_decision.json", {"decision": "C0_weak_risk_signal_needs_c0b", "warnings": warnings})
    write_report(args.out_dir, warnings, transition_rows, bootstrap_rows, d1_rows)
    print(json.dumps({"status": "COMPLETED", "out_dir": relpath(args.out_dir), "warnings": len(warnings)}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
