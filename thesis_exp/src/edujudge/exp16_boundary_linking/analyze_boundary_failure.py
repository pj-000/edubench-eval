"""RQ1 failure diagnosis for Exp16A boundary-linking predictions."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from thesis_exp.src.edujudge.exp16_boundary_linking import EXP16_OUTPUT_DIR
from thesis_exp.src.edujudge.utils.io import read_jsonl, relpath, write_csv, write_text


DEFAULT_VARIANTS = ("qmr", "metric_rubric")
DEFAULT_SPLITS = ("dev", "test")


def warn(message: str) -> None:
    print(f"WARNING: {message}")


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def mean(values: list[float]) -> float:
    clean = [value for value in values if not math.isnan(value)]
    return float(np.mean(clean)) if clean else float("nan")


def std(values: list[float]) -> float:
    clean = [value for value in values if not math.isnan(value)]
    return float(np.std(clean)) if clean else float("nan")


def med(values: list[float]) -> float:
    clean = [value for value in values if not math.isnan(value)]
    return float(median(clean)) if clean else float("nan")


def rate(num: int, den: int) -> float:
    return float(num / den) if den else float("nan")


def fmt(value: Any, digits: int = 4) -> str:
    try:
        if math.isnan(float(value)):
            return "nan"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def prediction_path(input_root: Path, variant: str, split: str, seed: int) -> Path | None:
    candidates = [
        input_root / f"scout_seed{seed}" / variant / f"predictions_{split}.jsonl",
        input_root / "runs" / variant / f"seed_{seed}" / f"predictions_{split}.jsonl",
        input_root / variant / f"seed_{seed}" / f"predictions_{split}.jsonl",
        input_root / variant / f"predictions_{split}.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return path
    warn(f"missing predictions for variant={variant} split={split}; checked: {', '.join(relpath(p) for p in candidates)}")
    return None


def load_predictions(input_root: Path, variants: list[str], splits: list[str], seed: int) -> dict[tuple[str, str], list[dict[str, Any]]]:
    out: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for variant in variants:
        for split in splits:
            path = prediction_path(input_root, variant, split, seed)
            if path is None:
                continue
            out[(variant, split)] = read_jsonl(path)
    return out


def summarize_by_label(rows: list[dict[str, Any]], variant: str, split: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for label in [1, 2, 3, 4, 5]:
        items = [row for row in rows if int(row.get("gold_label", 0)) == label]
        pred_ge4 = sum(1 for row in items if int(row.get("pred_label", 0)) >= 4)
        recall_hits = sum(1 for row in items if int(row.get("pred_label", 0)) == label)
        row = {
            "variant": variant,
            "split": split,
            "gold_label": label,
            "n": len(items),
            "mean_s": mean([safe_float(item.get("quality_score_s")) for item in items]),
            "median_s": med([safe_float(item.get("quality_score_s")) for item in items]),
            "std_s": std([safe_float(item.get("quality_score_s")) for item in items]),
            "mean_tau1": mean([safe_float(item.get("tau1")) for item in items]),
            "mean_tau2": mean([safe_float(item.get("tau2")) for item in items]),
            "mean_tau3": mean([safe_float(item.get("tau3")) for item in items]),
            "mean_tau4": mean([safe_float(item.get("tau4")) for item in items]),
            "mean_margin_tau2": mean([safe_float(item.get("margin_tau2")) for item in items]),
            "median_margin_tau2": med([safe_float(item.get("margin_tau2")) for item in items]),
            "mean_margin_tau3": mean([safe_float(item.get("margin_tau3")) for item in items]),
            "median_margin_tau3": med([safe_float(item.get("margin_tau3")) for item in items]),
            "pred_ge4_count": pred_ge4,
            "pred_ge4_rate": rate(pred_ge4, len(items)),
            "recall": rate(recall_hits, len(items)),
        }
        out.append(row)
    return out


def low_to_high_rows(rows: list[dict[str, Any]], variant: str, split: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    fields = [
        "variant",
        "split",
        "sample_id",
        "question_key",
        "metric",
        "gold_label",
        "pred_label",
        "quality_score_s",
        "tau1",
        "tau2",
        "tau3",
        "tau4",
        "margin_tau2",
        "margin_tau3",
        "probs",
        "boundary_key",
    ]
    for row in rows:
        if int(row.get("gold_label", 0)) <= 2 and int(row.get("pred_label", 0)) >= 4:
            item = {"variant": variant, "split": split}
            for field in fields:
                if field in {"variant", "split"}:
                    continue
                value = row.get(field, "")
                item[field] = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
            out.append(item)
    return out


def metric_failure_summary(rows: list[dict[str, Any]], variant: str, split: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("metric") or "")].append(row)
    out: list[dict[str, Any]] = []
    for metric, items in sorted(groups.items()):
        low = [row for row in items if int(row.get("gold_label", 0)) <= 2]
        label2 = [row for row in items if int(row.get("gold_label", 0)) == 2]
        l2h = [row for row in low if int(row.get("pred_label", 0)) >= 4]
        label2_ge4 = [row for row in label2 if int(row.get("pred_label", 0)) >= 4]
        out.append(
            {
                "variant": variant,
                "split": split,
                "metric": metric,
                "n": len(items),
                "true_low_n": len(low),
                "low_to_high_count": len(l2h),
                "low_to_high_rate": rate(len(l2h), len(low)),
                "label2_n": len(label2),
                "label2_pred_ge4_count": len(label2_ge4),
                "label2_pred_ge4_rate": rate(len(label2_ge4), len(label2)),
                "mean_s_for_low": mean([safe_float(row.get("quality_score_s")) for row in low]),
                "mean_tau3": mean([safe_float(row.get("tau3")) for row in items]),
                "mean_tau4": mean([safe_float(row.get("tau4")) for row in items]),
                "mean_margin_tau3_for_low": mean([safe_float(row.get("margin_tau3")) for row in low]),
            }
        )
    return out


def boundary_stability(rows: list[dict[str, Any]], variant: str, split: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = str(row.get("boundary_key") or "")
        if key:
            groups[key].append(row)
    out: list[dict[str, Any]] = []
    for key, items in sorted(groups.items()):
        tau_values = [[safe_float(row.get(f"tau{idx}")) for row in items] for idx in range(1, 5)]
        max_abs_diff = 0.0
        for values in tau_values:
            clean = [value for value in values if not math.isnan(value)]
            if clean:
                max_abs_diff = max(max_abs_diff, max(clean) - min(clean))
        out.append(
            {
                "variant": variant,
                "split": split,
                "boundary_key": key,
                "n": len(items),
                "question_key_count": len({str(row.get("question_key") or "") for row in items}),
                "metric_count": len({str(row.get("metric") or "") for row in items}),
                "std_tau1": std(tau_values[0]),
                "std_tau2": std(tau_values[1]),
                "std_tau3": std(tau_values[2]),
                "std_tau4": std(tau_values[3]),
                "max_abs_tau_diff": max_abs_diff,
            }
        )
    return out


def pred_distribution(rows: list[dict[str, Any]], variant: str, split: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for gold in [1, 2, 3, 4, 5]:
        items = [row for row in rows if int(row.get("gold_label", 0)) == gold]
        counts = Counter(int(row.get("pred_label", 0)) for row in items)
        for pred in [1, 2, 3, 4, 5]:
            count = counts.get(pred, 0)
            out.append(
                {
                    "variant": variant,
                    "split": split,
                    "gold_label": gold,
                    "pred_label": pred,
                    "count": count,
                    "rate_within_gold": rate(count, len(items)),
                }
            )
    return out


def diagnose_mode(rows: list[dict[str, Any]]) -> tuple[str, dict[str, float]]:
    low = [row for row in rows if int(row.get("gold_label", 0)) <= 2]
    l2h = [row for row in low if int(row.get("pred_label", 0)) >= 4]
    non_l2h = [row for row in low if int(row.get("pred_label", 0)) < 4]
    if not l2h or not non_l2h:
        return "insufficient low/non-low-to-high contrast", {}
    s_gap = mean([safe_float(row.get("quality_score_s")) for row in l2h]) - mean(
        [safe_float(row.get("quality_score_s")) for row in non_l2h]
    )
    tau3_gap = mean([safe_float(row.get("tau3")) for row in l2h]) - mean([safe_float(row.get("tau3")) for row in non_l2h])
    margin_gap = mean([safe_float(row.get("margin_tau3")) for row in l2h]) - mean(
        [safe_float(row.get("margin_tau3")) for row in non_l2h]
    )
    s_high = s_gap > 0.1
    tau_low = tau3_gap < -0.1
    if s_high and tau_low:
        label = "both"
    elif s_high:
        label = "s-too-high"
    elif tau_low:
        label = "tau-too-low"
    else:
        label = "weak separation / both small"
    return label, {"s_gap_l2h_minus_other_low": s_gap, "tau3_gap_l2h_minus_other_low": tau3_gap, "margin_gap": margin_gap}


def generate_report(
    output_dir: Path,
    data: dict[tuple[str, str], list[dict[str, Any]]],
    margins: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    stability_rows: list[dict[str, Any]],
) -> None:
    lines: list[str] = [
        "# Exp16A-Diag Boundary Failure Diagnosis",
        "",
        "This diagnostic uses saved prediction files only. It does not load checkpoints or train models.",
        "",
        "## Analyzed Runs",
        "",
    ]
    for (variant, split), rows in sorted(data.items()):
        lines.append(f"- `{variant}` `{split}`: `{len(rows)}` rows")
    lines += ["", "## Per-Label Margin Summary", ""]
    lines.append("| variant | split | gold | n | mean s | mean tau3 | mean tau4 | mean margin tau3 | pred>=4 | recall |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in margins:
        lines.append(
            f"| `{row['variant']}` | {row['split']} | {row['gold_label']} | {row['n']} | {fmt(row['mean_s'])} | "
            f"{fmt(row['mean_tau3'])} | {fmt(row['mean_tau4'])} | {fmt(row['mean_margin_tau3'])} | "
            f"{row['pred_ge4_count']}/{row['n']} ({fmt(row['pred_ge4_rate'])}) | {fmt(row['recall'])} |"
        )
    lines += ["", "## Label-2 Failure Pattern", ""]
    for (variant, split), rows in sorted(data.items()):
        label2 = [row for row in rows if int(row.get("gold_label", 0)) == 2]
        dist = Counter(int(row.get("pred_label", 0)) for row in label2)
        ge4 = sum(count for pred, count in dist.items() if pred >= 4)
        pieces = ", ".join(f"pred={pred}: {dist.get(pred, 0)}" for pred in [1, 2, 3, 4, 5])
        lines.append(f"- `{variant}` `{split}` label2 n=`{len(label2)}`, pred>=4=`{ge4}` ({fmt(rate(ge4, len(label2)))})；{pieces}.")
    lines += ["", "## Low-To-High Diagnosis", ""]
    for (variant, split), rows in sorted(data.items()):
        l2h = [row for row in rows if int(row.get("gold_label", 0)) <= 2 and int(row.get("pred_label", 0)) >= 4]
        label, details = diagnose_mode(rows)
        lines.append(
            f"- `{variant}` `{split}`: low-to-high `{len(l2h)}`. "
            f"mean s `{fmt(mean([safe_float(r.get('quality_score_s')) for r in l2h]))}`, "
            f"mean tau3 `{fmt(mean([safe_float(r.get('tau3')) for r in l2h]))}`, "
            f"mean tau4 `{fmt(mean([safe_float(r.get('tau4')) for r in l2h]))}`, "
            f"mean margin_tau3 `{fmt(mean([safe_float(r.get('margin_tau3')) for r in l2h]))}`. "
            f"Diagnosis: `{label}` { {k: fmt(v) for k, v in details.items()} }."
        )
    lines += ["", "## Metric Concentration", ""]
    top_metric_rows = sorted(
        [row for row in metric_rows if int(row["true_low_n"]) > 0],
        key=lambda row: (str(row["variant"]), str(row["split"]), -safe_float(row["low_to_high_rate"]), -int(row["true_low_n"])),
    )
    lines.append("| variant | split | metric | true low | low-to-high | label2 pred>=4 | mean low margin tau3 |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for row in top_metric_rows[:30]:
        lines.append(
            f"| `{row['variant']}` | {row['split']} | {row['metric']} | {row['true_low_n']} | "
            f"{row['low_to_high_count']} ({fmt(row['low_to_high_rate'])}) | "
            f"{row['label2_pred_ge4_count']}/{row['label2_n']} ({fmt(row['label2_pred_ge4_rate'])}) | "
            f"{fmt(row['mean_margin_tau3_for_low'])} |"
        )
    unstable = [row for row in stability_rows if safe_float(row["max_abs_tau_diff"]) > 1e-5]
    lines += ["", "## Boundary-Key Stability", ""]
    if unstable:
        lines.append(
            f"WARNING: `{len(unstable)}` boundary_key groups have non-negligible tau variation "
            "(`max_abs_tau_diff > 1e-5`). This is not a monotonicity violation; it means the same "
            "saved boundary context can still produce slightly different tau values across prediction rows."
        )
    else:
        lines.append("All observed boundary_key groups have near-zero tau variation within tolerance.")
    lines += [
        "",
        "## qmr vs metric_rubric",
        "",
        "- `qmr` includes question text in the boundary tower; `metric_rubric` only uses metric and rubric.",
        "- Compare dev/test low-to-high and label-2 prediction distributions above to decide whether question-specific boundaries improve stability.",
        "",
        "## Concise Conclusion",
        "",
        "Exp16A enforces ordered thresholds with zero monotonic violation, but the same boundary_key can still produce non-identical tau values in saved predictions. Treat this boundary-key warning as part of the RQ1 diagnosis, not as a solved stability result. Label-2 recall and low-to-high failures remain the key weakness. The failure diagnosis should be read as evidence for whether the next step should calibrate tau thresholds or improve the quality score separation before moving to risk-aware RQ2 methods.",
    ]
    write_text(output_dir / "boundary_failure_diagnosis.md", "\n".join(lines))


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_predictions(args.input_root, args.variants, args.splits, args.seed)
    margins: list[dict[str, Any]] = []
    l2h_cases: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    dist_rows: list[dict[str, Any]] = []
    for (variant, split), rows in sorted(data.items()):
        margins.extend(summarize_by_label(rows, variant, split))
        l2h_cases.extend(low_to_high_rows(rows, variant, split))
        metric_rows.extend(metric_failure_summary(rows, variant, split))
        stability_rows.extend(boundary_stability(rows, variant, split))
        dist_rows.extend(pred_distribution(rows, variant, split))
    write_csv(output_dir / "margins_by_label.csv", margins)
    write_csv(output_dir / "low_to_high_cases.csv", l2h_cases)
    write_csv(output_dir / "metric_failure_summary.csv", metric_rows)
    write_csv(output_dir / "boundary_key_stability.csv", stability_rows)
    write_csv(output_dir / "pred_distribution_by_label.csv", dist_rows)
    generate_report(output_dir, data, margins, metric_rows, stability_rows)
    return {"analyzed": len(data), "output_dir": relpath(output_dir)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose Exp16A boundary failures from saved predictions.")
    parser.add_argument("--input_root", type=Path, default=EXP16_OUTPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=EXP16_OUTPUT_DIR / "rq1_diagnosis")
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
