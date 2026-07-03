"""Collect Exp19 first-round SFT dev predictions from LLaMA-Factory outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence.run_exp19_r0a_qwen4b_direct_baseline import (  # noqa: E402
    LABELS,
    kendall_tau_b,
    mean,
    parse_score,
    quadratic_weighted_kappa,
    safe_rate,
)
from thesis_exp.src.edujudge.utils.io import write_csv, write_json, write_text  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_sft_first_round")
DEFAULT_REFERENCE = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp19_sft_dpo_datasets_seed42/tables/exp19_dev_reference.csv"
)

RUNS = {
    "r1_score_only_natural": "R1 score-only natural",
    "r2_reason_score_balanced": "R2 reason-score balanced",
    "r4_shuffled_reason_control": "R4 shuffled reason control",
}

METRIC_FIELDS = [
    "run_name",
    "run_label",
    "split",
    "n",
    "valid_n",
    "MAE",
    "QWK",
    "Signed_Bias",
    "Exact_Match",
    "Kendall_tau",
    "low_to_high_count",
    "low_to_high_rate",
    "high_to_low_count",
    "high_to_low_rate",
    "label1_recall",
    "label2_recall",
    "label2_pred_ge4_rate",
    "label5_recall",
    "parse_success_rate",
    "invalid_score_rate",
]

LABEL_FIELDS = [
    "run_name",
    "run_label",
    "split",
    "gold_label",
    "n",
    "exact_accuracy",
    "mean_pred",
    "pred_1_rate",
    "pred_2_rate",
    "pred_3_rate",
    "pred_4_rate",
    "pred_5_rate",
]

PARSE_FIELDS = [
    "run_name",
    "run_label",
    "split",
    "n",
    "parse_success_count",
    "parse_success_rate",
    "json_parse_count",
    "regex_fallback_count",
    "failed_count",
    "prediction_file",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        out = int(float(value))
        return out if out in LABELS else None
    except Exception:
        return None


def fmt(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except Exception:
        return str(value)
    if math.isnan(val):
        return "nan"
    return f"{val:.{digits}f}"


def first_text_value(record: Any) -> str:
    if isinstance(record, str):
        return record
    if not isinstance(record, dict):
        return ""
    for key in (
        "predict",
        "prediction",
        "generated_text",
        "generated",
        "response",
        "output",
        "text",
        "content",
    ):
        value = record.get(key)
        if isinstance(value, str):
            return value
    for key in ("predictions", "outputs"):
        value = record.get(key)
        if isinstance(value, list) and value:
            return first_text_value(value[0])
    return ""


def load_prediction_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [{"raw_record": item, "raw_output": first_text_value(item)} for item in data]
        if isinstance(data, dict):
            rows = data.get("predictions") or data.get("outputs") or data.get("results")
            if isinstance(rows, list):
                return [{"raw_record": item, "raw_output": first_text_value(item)} for item in rows]
            return [{"raw_record": data, "raw_output": first_text_value(data)}]

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                rows.append({"raw_record": record, "raw_output": first_text_value(record)})
            except json.JSONDecodeError:
                rows.append({"raw_record": line.rstrip("\n"), "raw_output": line.rstrip("\n")})
    return rows


def find_prediction_file(run_dir: Path) -> Path:
    preferred = [
        run_dir / "generated_predictions.jsonl",
        run_dir / "generated_predictions.json",
        run_dir / "predictions.jsonl",
        run_dir / "predict_results.json",
    ]
    for path in preferred:
        if path.exists():
            return path
    candidates = sorted(
        path
        for path in run_dir.rglob("*")
        if path.is_file()
        and re.search(r"(generated|predict|prediction).*\.(jsonl|json)$", path.name, flags=re.IGNORECASE)
    )
    if not candidates:
        raise FileNotFoundError(f"No LLaMA-Factory prediction file found under {run_dir}")
    return candidates[0]


def align_predictions(
    reference: list[dict[str, str]],
    prediction_records: list[dict[str, Any]],
    run_name: str,
    run_label: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    n = min(len(reference), len(prediction_records))
    for idx in range(n):
        ref = reference[idx]
        raw_output = prediction_records[idx].get("raw_output", "")
        parsed = parse_score(str(raw_output))
        pred = parsed.get("pred_label")
        out.append(
            {
                "run_name": run_name,
                "run_label": run_label,
                "eval_index": int(ref["eval_index"]),
                "split": ref.get("split", "dev"),
                "sample_id": ref.get("sample_id", ""),
                "record_id": ref.get("record_id", ""),
                "question_key": ref.get("question_key", ""),
                "question_group_id": ref.get("question_group_id", ""),
                "metric": ref.get("metric", ""),
                "metric_id": ref.get("metric_id", ""),
                "language": ref.get("language", ""),
                "subject": ref.get("subject", ""),
                "gold_label": int(ref["gold_label"]),
                "pred_label": pred,
                "parse_success": bool(parsed.get("parse_success")),
                "parse_method": parsed.get("parse_method", "failed"),
                "raw_output_truncated": str(raw_output)[:280],
            }
        )
    if len(reference) != len(prediction_records):
        print(
            f"WARNING {run_name}: reference rows={len(reference)} prediction rows={len(prediction_records)}; aligned={n}",
            file=sys.stderr,
        )
    return out


def metric_summary(rows: list[dict[str, Any]], run_name: str, run_label: str, split: str) -> dict[str, Any]:
    valid = [row for row in rows if row.get("pred_label") is not None and bool(row.get("parse_success"))]
    gold = [int(row["gold_label"]) for row in valid]
    pred = [int(row["pred_label"]) for row in valid]
    low = [row for row in rows if int(row["gold_label"]) <= 2]
    high = [row for row in rows if int(row["gold_label"]) >= 4]
    label1 = [row for row in rows if int(row["gold_label"]) == 1]
    label2 = [row for row in rows if int(row["gold_label"]) == 2]
    label5 = [row for row in rows if int(row["gold_label"]) == 5]
    low_to_high = sum(1 for row in low if safe_int(row.get("pred_label")) is not None and int(row["pred_label"]) >= 4)
    high_to_low = sum(1 for row in high if safe_int(row.get("pred_label")) is not None and int(row["pred_label"]) <= 2)
    exact = sum(1 for row in rows if safe_int(row.get("pred_label")) == int(row["gold_label"]))
    label1_hits = sum(1 for row in label1 if safe_int(row.get("pred_label")) == 1)
    label2_hits = sum(1 for row in label2 if safe_int(row.get("pred_label")) == 2)
    label2_ge4 = sum(1 for row in label2 if safe_int(row.get("pred_label")) is not None and int(row["pred_label"]) >= 4)
    label5_hits = sum(1 for row in label5 if safe_int(row.get("pred_label")) == 5)
    return {
        "run_name": run_name,
        "run_label": run_label,
        "split": split,
        "n": len(rows),
        "valid_n": len(valid),
        "MAE": mean([abs(g - p) for g, p in zip(gold, pred)]),
        "QWK": quadratic_weighted_kappa(gold, pred),
        "Signed_Bias": mean([p - g for g, p in zip(gold, pred)]),
        "Exact_Match": safe_rate(exact, len(rows)),
        "Kendall_tau": kendall_tau_b(gold, pred),
        "low_to_high_count": low_to_high,
        "low_to_high_rate": safe_rate(low_to_high, len(low)),
        "high_to_low_count": high_to_low,
        "high_to_low_rate": safe_rate(high_to_low, len(high)),
        "label1_recall": safe_rate(label1_hits, len(label1)),
        "label2_recall": safe_rate(label2_hits, len(label2)),
        "label2_pred_ge4_rate": safe_rate(label2_ge4, len(label2)),
        "label5_recall": safe_rate(label5_hits, len(label5)),
        "parse_success_rate": safe_rate(len(valid), len(rows)),
        "invalid_score_rate": safe_rate(len(rows) - len(valid), len(rows)),
    }


def label_summary(rows: list[dict[str, Any]], run_name: str, run_label: str, split: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for label in LABELS:
        items = [row for row in rows if int(row["gold_label"]) == label]
        preds = [safe_int(row.get("pred_label")) for row in items]
        valid = [int(value) for value in preds if value is not None]
        row = {
            "run_name": run_name,
            "run_label": run_label,
            "split": split,
            "gold_label": label,
            "n": len(items),
            "exact_accuracy": safe_rate(sum(1 for value in preds if value == label), len(items)),
            "mean_pred": mean(valid),
        }
        for pred_label in LABELS:
            row[f"pred_{pred_label}_rate"] = safe_rate(sum(1 for value in preds if value == pred_label), len(items))
        out.append(row)
    return out


def grouped_metric_rows(rows: list[dict[str, Any]], group_key: str, run_name: str, run_label: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(group_key) or "unknown")].append(row)
    out: list[dict[str, Any]] = []
    for key, items in sorted(groups.items()):
        if not items:
            continue
        summary = metric_summary(items, run_name, run_label, str(items[0].get("split", "dev")))
        summary[group_key] = key
        out.append(summary)
    return out


def parse_summary(
    rows: list[dict[str, Any]],
    run_name: str,
    run_label: str,
    split: str,
    prediction_file: Path,
) -> dict[str, Any]:
    methods = Counter(str(row.get("parse_method", "failed")) for row in rows)
    success = sum(1 for row in rows if bool(row.get("parse_success")))
    return {
        "run_name": run_name,
        "run_label": run_label,
        "split": split,
        "n": len(rows),
        "parse_success_count": success,
        "parse_success_rate": safe_rate(success, len(rows)),
        "json_parse_count": methods.get("json", 0),
        "regex_fallback_count": methods.get("regex_fallback", 0),
        "failed_count": methods.get("failed", 0),
        "prediction_file": str(prediction_file),
    }


def write_report(out_dir: Path, metric_rows: list[dict[str, Any]], parse_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Exp19 First-Round SFT Dev Evaluation",
        "",
        "This report summarizes LLaMA-Factory `do_predict` outputs for R1/R2/R4 on the original dev split.",
        "Raw generated predictions remain in gitignored `dev_predictions/` directories.",
        "",
        "| run | n | parse | MAE | QWK | bias | exact | low-to-high | label2 recall | label5 recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            f"| {row['run_label']} | {row['n']} | {fmt(row['parse_success_rate'])} | "
            f"{fmt(row['MAE'])} | {fmt(row['QWK'])} | {fmt(row['Signed_Bias'])} | "
            f"{fmt(row['Exact_Match'])} | {row['low_to_high_count']} ({fmt(row['low_to_high_rate'])}) | "
            f"{fmt(row['label2_recall'])} | {fmt(row['label5_recall'])} |"
        )
    lines.extend(["", "## Parse Summary", ""])
    for row in parse_rows:
        lines.append(
            f"- {row['run_label']}: success={row['parse_success_count']}/{row['n']} "
            f"({fmt(row['parse_success_rate'])}), json={row['json_parse_count']}, "
            f"regex={row['regex_fallback_count']}, failed={row['failed_count']}"
        )
    lines.extend(
        [
            "",
            "## Evaluation Guardrails",
            "",
            "- Evaluation uses the original question-disjoint dev split, not a balanced training distribution.",
            "- No test split is read by this collection script.",
            "- The collector parses only generated assistant text and gold labels from the dev reference table.",
        ]
    )
    write_text(out_dir / "reports" / "exp19_sft_first_round_dev_report.md", "\n".join(lines))


def collect(args: argparse.Namespace) -> dict[str, Any]:
    reference = read_csv_rows(args.reference_csv)
    metric_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    parse_rows: list[dict[str, Any]] = []
    metric_group_rows: list[dict[str, Any]] = []
    language_group_rows: list[dict[str, Any]] = []
    parsed_rows_all: list[dict[str, Any]] = []
    prediction_files: dict[str, str] = {}

    for run_name, run_label in RUNS.items():
        run_dir = args.prediction_root / run_name
        prediction_file = find_prediction_file(run_dir)
        prediction_files[run_name] = str(prediction_file)
        prediction_records = load_prediction_records(prediction_file)
        parsed_rows = align_predictions(reference, prediction_records, run_name, run_label)
        split = parsed_rows[0].get("split", "dev") if parsed_rows else "dev"
        metric_rows.append(metric_summary(parsed_rows, run_name, run_label, str(split)))
        label_rows.extend(label_summary(parsed_rows, run_name, run_label, str(split)))
        parse_rows.append(parse_summary(parsed_rows, run_name, run_label, str(split), prediction_file))
        metric_group_rows.extend(grouped_metric_rows(parsed_rows, "metric", run_name, run_label))
        language_group_rows.extend(grouped_metric_rows(parsed_rows, "language", run_name, run_label))
        parsed_rows_all.extend(parsed_rows)

    tables_dir = args.out_dir / "tables"
    write_csv(tables_dir / "exp19_sft_first_round_dev_metrics.csv", metric_rows, METRIC_FIELDS)
    write_csv(tables_dir / "exp19_sft_first_round_dev_label_metrics.csv", label_rows, LABEL_FIELDS)
    write_csv(tables_dir / "exp19_sft_first_round_dev_parse_summary.csv", parse_rows, PARSE_FIELDS)
    write_csv(tables_dir / "exp19_sft_first_round_dev_by_metric.csv", metric_group_rows)
    write_csv(tables_dir / "exp19_sft_first_round_dev_by_language.csv", language_group_rows)
    if args.write_parsed_csv:
        write_csv(args.out_dir / "diagnostics" / "exp19_sft_first_round_dev_parsed_predictions.csv", parsed_rows_all)
    write_json(args.out_dir / "reports" / "exp19_sft_first_round_dev_prediction_files.json", prediction_files)
    write_report(args.out_dir, metric_rows, parse_rows)
    return {"runs": len(metric_rows), "prediction_files": prediction_files}


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Exp19 first-round SFT dev predictions.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_OUTPUT_DIR / "dev_predictions")
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--write-parsed-csv", action="store_true")
    args = parser.parse_args()
    summary = collect(args)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
