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
    clean_text,
    json_metric,
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
DEFAULT_D1_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/d1_hidden_failure_audit_seed42_dev/"
    "summary_human_rationale_recovered"
)

RUNS = {
    "r1_score_only_natural": "R1 score-only natural",
    "r2_reason_score_balanced": "R2 reason-score balanced",
    "r4_shuffled_reason_control": "R4 shuffled reason control",
}
FAILURE_EVAL_TYPES = [
    "missing_key_point",
    "factual_or_rubric_mismatch",
    "answer_key_or_reference_mismatch",
    "surface_fluent_but_hidden_defect",
    "insufficient_evidence",
    "task_constraint_violation",
    "format_violation",
    "possible_label_conflict",
]

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
D1_FIELDS = [
    "run_name",
    "run_label",
    "n_d1_cases",
    "mean_pred_d1_hidden",
    "pred_ge4_rate_d1_hidden",
    "pred_5_rate_d1_hidden",
    "label2_recall_d1",
    "matched_control_mean_pred",
    "hidden_control_score_gap",
]
FAILURE_TYPE_FIELDS = [
    "run_name",
    "run_label",
    "failure_type_micro_f1",
    "failure_type_macro_f1",
    "missing_key_point_recall",
    "factual_or_rubric_mismatch_recall",
    "insufficient_evidence_recall",
    "task_constraint_violation_recall",
    "no_major_failure_rate_on_controls",
    "major_failure_nonempty_rate_on_d1_hidden",
    "major_failure_nonempty_rate_on_high_controls",
]
STRUCTURED_PARSE_FIELDS = [
    "run_name",
    "run_label",
    "n",
    "score_json_parse_rate",
    "major_failures_parse_rate",
    "score_cap_parse_rate",
    "rubric_satisfied_parse_rate",
    "valid_full_schema_rate",
    "no_major_failure_rate",
    "score_cap_nonnull_rate",
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


def safe_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def fmt(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except Exception:
        return str(value)
    if math.isnan(val):
        return "nan"
    return f"{val:.{digits}f}"


def truthy(value: Any, blank_is_true: bool = False) -> bool:
    text = clean_text(value).lower()
    if not text:
        return bool(blank_is_true)
    return text in {"1", "true", "yes", "y"}


def normalize_failure_name(value: Any) -> str | None:
    text = clean_text(value).lower().strip()
    text = re.sub(r"[^a-z0-9_]+", "_", text).strip("_")
    aliases = {
        "factual_mismatch": "factual_or_rubric_mismatch",
        "rubric_mismatch": "factual_or_rubric_mismatch",
        "factual_or_rubric_mismatch": "factual_or_rubric_mismatch",
        "missing_key_point": "missing_key_point",
        "missing_required_content": "missing_key_point",
        "insufficient_evidence": "insufficient_evidence",
        "task_constraint_violation": "task_constraint_violation",
        "format_violation": "format_violation",
        "surface_fluent_but_hidden_defect": "surface_fluent_but_hidden_defect",
        "answer_key_or_reference_mismatch": "answer_key_or_reference_mismatch",
        "possible_label_conflict": "possible_label_conflict",
        "no_major_failure": "no_major_failure",
        "none": "no_major_failure",
        "unclear": "unclear",
    }
    return aliases.get(text)


def normalize_failure_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
            if isinstance(loaded, list):
                value = loaded
            else:
                value = [value]
        except Exception:
            value = [part for part in re.split(r"[;,|]", value) if part.strip()]
    if not isinstance(value, list):
        value = [value]
    out: list[str] = []
    for item in value:
        name = normalize_failure_name(item)
        if name and name not in out:
            out.append(name)
    return out


def first_json_object(text: str) -> dict[str, Any] | None:
    text = clean_text(text)
    if not text:
        return None
    candidates = [text]
    start = text.find("{")
    if start >= 0:
        decoder = json.JSONDecoder()
        try:
            obj, _end = decoder.raw_decode(text[start:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.insert(0, match.group(0))
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def parsed_payload(raw_output: str) -> dict[str, Any]:
    data = first_json_object(raw_output) or {}
    score = safe_int(data.get("score")) if data else None
    failures_raw = data.get("major_failures") if data else None
    failures = normalize_failure_list(failures_raw)
    score_cap_present = data and "score_cap" in data
    score_cap = None
    score_cap_parse_ok = False
    if score_cap_present:
        if data.get("score_cap") is None:
            score_cap_parse_ok = True
        else:
            score_cap = safe_int(data.get("score_cap"))
            score_cap_parse_ok = score_cap is not None
    rubric_present = data and "rubric_satisfied" in data
    rubric_satisfied: bool | None = None
    rubric_parse_ok = False
    if rubric_present:
        value = data.get("rubric_satisfied")
        if isinstance(value, bool):
            rubric_satisfied = value
            rubric_parse_ok = True
        elif isinstance(value, str) and value.lower().strip() in {"true", "false"}:
            rubric_satisfied = value.lower().strip() == "true"
            rubric_parse_ok = True
    return {
        "payload": data,
        "score_json_parse_ok": score is not None,
        "major_failures": failures,
        "major_failures_parse_ok": isinstance(failures_raw, list) or (isinstance(failures_raw, str) and bool(failures)),
        "score_cap": score_cap,
        "score_cap_present": bool(score_cap_present),
        "score_cap_parse_ok": score_cap_parse_ok,
        "rubric_satisfied": rubric_satisfied,
        "rubric_satisfied_present": bool(rubric_present),
        "rubric_satisfied_parse_ok": rubric_parse_ok,
        "brief_reason": clean_text(data.get("brief_reason") if data else ""),
        "valid_full_schema": bool(
            score is not None and failures and score_cap_present and score_cap_parse_ok and rubric_parse_ok
        ),
    }


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
        structured = parsed_payload(str(raw_output))
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
                "score_json_parse_ok": bool(structured["score_json_parse_ok"]),
                "major_failures": structured["major_failures"],
                "major_failures_parse_ok": bool(structured["major_failures_parse_ok"]),
                "score_cap": structured["score_cap"],
                "score_cap_present": bool(structured["score_cap_present"]),
                "score_cap_parse_ok": bool(structured["score_cap_parse_ok"]),
                "rubric_satisfied": structured["rubric_satisfied"],
                "rubric_satisfied_present": bool(structured["rubric_satisfied_present"]),
                "rubric_satisfied_parse_ok": bool(structured["rubric_satisfied_parse_ok"]),
                "brief_reason": structured["brief_reason"],
                "valid_full_schema": bool(structured["valid_full_schema"]),
                "raw_output_truncated": str(raw_output)[:280],
            }
        )
    if len(reference) != len(prediction_records):
        print(
            f"WARNING {run_name}: reference rows={len(reference)} prediction rows={len(prediction_records)}; aligned={n}",
            file=sys.stderr,
        )
    return out


def has_substantive_failure(row: dict[str, Any]) -> bool:
    failures = normalize_failure_list(row.get("major_failures"))
    return any(name not in {"no_major_failure", "unclear"} for name in failures)


def has_no_major_failure(row: dict[str, Any]) -> bool:
    failures = normalize_failure_list(row.get("major_failures"))
    return "no_major_failure" in failures


def resolve_d1_base(path: Path) -> Path:
    path = Path(path)
    if (path / "d1_hidden_failure_annotation_template_filled_human_rationale_recovered.csv").exists():
        return path
    if path.name.startswith("summary") and (
        path.parent / "d1_hidden_failure_annotation_template_filled_human_rationale_recovered.csv"
    ).exists():
        return path.parent
    return path


def load_d1_annotations(d1_dir: Path) -> tuple[dict[str, dict[str, str]], list[tuple[str, str]], set[str]]:
    base = resolve_d1_base(d1_dir)
    annotation_path = base / "d1_hidden_failure_annotation_template_filled_human_rationale_recovered.csv"
    annotations: dict[str, dict[str, str]] = {}
    if annotation_path.exists():
        with annotation_path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                sid = clean_text(row.get("sample_id"))
                if not sid:
                    continue
                hidden = truthy(row.get("is_hidden_failure_manual"), blank_is_true=True)
                conflict = truthy(row.get("possible_label_conflict_manual"))
                if hidden and not conflict:
                    annotations[sid] = row
    pair_path = base / "d1_matched_case_control_review.csv"
    pairs: list[tuple[str, str]] = []
    controls: set[str] = set()
    if pair_path.exists():
        with pair_path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                case_id = clean_text(row.get("case_sample_id"))
                control_id = clean_text(row.get("control_sample_id"))
                if case_id and control_id:
                    pairs.append((case_id, control_id))
                    controls.add(control_id)
    return annotations, pairs, controls


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


def structured_parse_summary(rows: list[dict[str, Any]], run_name: str, run_label: str) -> dict[str, Any]:
    n = len(rows)
    return {
        "run_name": run_name,
        "run_label": run_label,
        "n": n,
        "score_json_parse_rate": safe_rate(sum(1 for row in rows if row.get("score_json_parse_ok")), n),
        "major_failures_parse_rate": safe_rate(sum(1 for row in rows if row.get("major_failures_parse_ok")), n),
        "score_cap_parse_rate": safe_rate(sum(1 for row in rows if row.get("score_cap_parse_ok")), n),
        "rubric_satisfied_parse_rate": safe_rate(sum(1 for row in rows if row.get("rubric_satisfied_parse_ok")), n),
        "valid_full_schema_rate": safe_rate(sum(1 for row in rows if row.get("valid_full_schema")), n),
        "no_major_failure_rate": safe_rate(sum(1 for row in rows if has_no_major_failure(row)), n),
        "score_cap_nonnull_rate": safe_rate(sum(1 for row in rows if row.get("score_cap") is not None), n),
    }


def d1_eval_row(rows: list[dict[str, Any]], d1_dir: Path, run_name: str, run_label: str) -> dict[str, Any]:
    annotations, pairs, control_ids = load_d1_annotations(d1_dir)
    pred_by_id = {row["sample_id"]: row for row in rows}
    cases = [pred_by_id[sid] for sid in sorted(annotations) if sid in pred_by_id]
    controls = [pred_by_id[sid] for sid in sorted(control_ids) if sid in pred_by_id]
    case_preds = [int(row["pred_label"]) for row in cases if safe_int(row.get("pred_label")) is not None]
    control_preds = [int(row["pred_label"]) for row in controls if safe_int(row.get("pred_label")) is not None]
    label2 = [row for row in cases if int(row["gold_label"]) == 2]
    paired_gaps = []
    for case_id, control_id in pairs:
        if case_id in pred_by_id and control_id in pred_by_id:
            case_pred = safe_int(pred_by_id[case_id].get("pred_label"))
            control_pred = safe_int(pred_by_id[control_id].get("pred_label"))
            if case_pred is not None and control_pred is not None:
                paired_gaps.append(control_pred - case_pred)
    return {
        "run_name": run_name,
        "run_label": run_label,
        "n_d1_cases": len(cases),
        "mean_pred_d1_hidden": mean(case_preds),
        "pred_ge4_rate_d1_hidden": safe_rate(sum(1 for value in case_preds if value >= 4), len(cases)),
        "pred_5_rate_d1_hidden": safe_rate(sum(1 for value in case_preds if value == 5), len(cases)),
        "label2_recall_d1": safe_rate(sum(1 for row in label2 if safe_int(row.get("pred_label")) == 2), len(label2)),
        "matched_control_mean_pred": mean(control_preds),
        "hidden_control_score_gap": mean(paired_gaps),
    }


def f1_score(tp: int, fp: int, fn: int) -> float:
    den = (2 * tp) + fp + fn
    return safe_rate(2 * tp, den)


def failure_type_eval_row(rows: list[dict[str, Any]], d1_dir: Path, run_name: str, run_label: str) -> dict[str, Any]:
    annotations, _pairs, control_ids = load_d1_annotations(d1_dir)
    pred_by_id = {row["sample_id"]: row for row in rows}
    counts: dict[str, Counter[str]] = {name: Counter() for name in FAILURE_EVAL_TYPES}
    micro = Counter()
    for sid, ann in annotations.items():
        if sid not in pred_by_id:
            continue
        gold = normalize_failure_name(ann.get("primary_failure_mode_manual")) or "unclear"
        pred_set = set(normalize_failure_list(pred_by_id[sid].get("major_failures")))
        pred_eval = {name for name in pred_set if name in FAILURE_EVAL_TYPES}
        if gold in FAILURE_EVAL_TYPES and gold in pred_eval:
            counts[gold]["tp"] += 1
            micro["tp"] += 1
        elif gold in FAILURE_EVAL_TYPES:
            counts[gold]["fn"] += 1
            micro["fn"] += 1
        for pred_name in pred_eval:
            if pred_name != gold:
                counts[pred_name]["fp"] += 1
                micro["fp"] += 1
    macro_values = []
    for name in FAILURE_EVAL_TYPES:
        if sum(counts[name].values()) > 0:
            macro_values.append(f1_score(counts[name]["tp"], counts[name]["fp"], counts[name]["fn"]))
    controls = [pred_by_id[sid] for sid in sorted(control_ids) if sid in pred_by_id]
    d1_hidden = [pred_by_id[sid] for sid in sorted(annotations) if sid in pred_by_id]
    high_controls = [row for row in rows if int(row["gold_label"]) >= 4]
    return {
        "run_name": run_name,
        "run_label": run_label,
        "failure_type_micro_f1": f1_score(micro["tp"], micro["fp"], micro["fn"]),
        "failure_type_macro_f1": mean(macro_values),
        "missing_key_point_recall": safe_rate(
            counts["missing_key_point"]["tp"],
            counts["missing_key_point"]["tp"] + counts["missing_key_point"]["fn"],
        ),
        "factual_or_rubric_mismatch_recall": safe_rate(
            counts["factual_or_rubric_mismatch"]["tp"],
            counts["factual_or_rubric_mismatch"]["tp"] + counts["factual_or_rubric_mismatch"]["fn"],
        ),
        "insufficient_evidence_recall": safe_rate(
            counts["insufficient_evidence"]["tp"],
            counts["insufficient_evidence"]["tp"] + counts["insufficient_evidence"]["fn"],
        ),
        "task_constraint_violation_recall": safe_rate(
            counts["task_constraint_violation"]["tp"],
            counts["task_constraint_violation"]["tp"] + counts["task_constraint_violation"]["fn"],
        ),
        "no_major_failure_rate_on_controls": safe_rate(sum(1 for row in controls if has_no_major_failure(row)), len(controls)),
        "major_failure_nonempty_rate_on_d1_hidden": safe_rate(
            sum(1 for row in d1_hidden if has_substantive_failure(row)), len(d1_hidden)
        ),
        "major_failure_nonempty_rate_on_high_controls": safe_rate(
            sum(1 for row in high_controls if has_substantive_failure(row)), len(high_controls)
        ),
    }


def run_by_name(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next((row for row in rows if row.get("run_name") == name), {})


def first_round_decision_json(metric_rows: list[dict[str, Any]], d1_rows: list[dict[str, Any]]) -> dict[str, Any]:
    r1 = run_by_name(metric_rows, "r1_score_only_natural")
    r2 = run_by_name(metric_rows, "r2_reason_score_balanced")
    r4 = run_by_name(metric_rows, "r4_shuffled_reason_control")
    r2_d1 = run_by_name(d1_rows, "r2_reason_score_balanced")
    r4_d1 = run_by_name(d1_rows, "r4_shuffled_reason_control")

    def value(row: dict[str, Any], key: str) -> float:
        return safe_float(row.get(key))

    runs_sorted_mae = sorted(metric_rows, key=lambda row: value(row, "MAE"))
    runs_sorted_l2h = sorted(metric_rows, key=lambda row: value(row, "low_to_high_rate"))
    best_overall = runs_sorted_mae[0]["run_name"] if runs_sorted_mae else ""
    best_low_risk = runs_sorted_l2h[0]["run_name"] if runs_sorted_l2h else ""

    r2_mae_guard_r1 = value(r2, "MAE") <= value(r1, "MAE") + 0.10
    r2_h2l_guard_r1 = value(r2, "high_to_low_rate") <= value(r1, "high_to_low_rate") + 0.10
    r2_beats_r1 = bool(
        value(r2, "low_to_high_rate") < value(r1, "low_to_high_rate")
        and value(r2, "label2_recall") >= value(r1, "label2_recall")
        and r2_mae_guard_r1
        and r2_h2l_guard_r1
    )
    r2_preserves_vs_r4 = bool(
        value(r2, "MAE") <= value(r4, "MAE") + 0.02 and value(r2, "QWK") >= value(r4, "QWK") - 0.02
    )
    d1_comparable = not math.isnan(value(r2_d1, "pred_ge4_rate_d1_hidden")) and not math.isnan(
        value(r4_d1, "pred_ge4_rate_d1_hidden")
    )
    r2_risk_better_than_r4 = value(r2, "low_to_high_rate") < value(r4, "low_to_high_rate") or (
        d1_comparable and value(r2_d1, "pred_ge4_rate_d1_hidden") < value(r4_d1, "pred_ge4_rate_d1_hidden")
    )
    r2_beats_r4 = bool(r2_risk_better_than_r4 and r2_preserves_vs_r4)
    proceed_to_r3 = bool(r2_beats_r1 and r2_beats_r4)
    non_control_structured_usable = bool(r2_beats_r1 or r2_beats_r4)
    proceed_to_second_round_ablation = bool(not r2_beats_r4)
    proceed_to_dpo = bool(non_control_structured_usable and not proceed_to_r3 and not proceed_to_second_round_ablation)
    reason = (
        "R2 beats both R1 and shuffled R4; proceed to rationale-bearing R3."
        if proceed_to_r3
        else (
            "R2 shows partial non-control structured signal but not enough for R3; DPO can be considered after ablation."
            if proceed_to_dpo
            else "R4 shuffled control is competitive with or better than R2; do second-round SFT ablation before R3/DPO."
        )
    )
    return {
        "best_overall_run": best_overall,
        "best_low_risk_run": best_low_risk,
        "r2_beats_r1": r2_beats_r1,
        "r2_beats_r4": r2_beats_r4,
        "proceed_to_r3": proceed_to_r3,
        "proceed_to_dpo": proceed_to_dpo,
        "proceed_to_second_round_ablation": proceed_to_second_round_ablation,
        "reason": reason,
        "metrics_used": {
            "r1_low_to_high": json_metric(r1.get("low_to_high_rate")),
            "r2_low_to_high": json_metric(r2.get("low_to_high_rate")),
            "r4_low_to_high": json_metric(r4.get("low_to_high_rate")),
            "r1_label2_recall": json_metric(r1.get("label2_recall")),
            "r2_label2_recall": json_metric(r2.get("label2_recall")),
            "r4_label2_recall": json_metric(r4.get("label2_recall")),
            "r2_d1_pred_ge4": json_metric(r2_d1.get("pred_ge4_rate_d1_hidden")),
            "r4_d1_pred_ge4": json_metric(r4_d1.get("pred_ge4_rate_d1_hidden")),
        },
    }


def second_round_decision_json(metric_rows: list[dict[str, Any]], d1_rows: list[dict[str, Any]]) -> dict[str, Any]:
    r1b = run_by_name(metric_rows, "r1b_score_only_balanced")
    r2n = run_by_name(metric_rows, "r2n_reason_score_natural")
    r2c = run_by_name(metric_rows, "r2c_clean_reason_score_balanced")
    r4b = run_by_name(metric_rows, "r4b_shuffled_reason_balanced")
    r2c_d1 = run_by_name(d1_rows, "r2c_clean_reason_score_balanced")
    r4b_d1 = run_by_name(d1_rows, "r4b_shuffled_reason_balanced")

    def value(row: dict[str, Any], key: str) -> float:
        return safe_float(row.get(key))

    runs_sorted_mae = sorted(metric_rows, key=lambda row: value(row, "MAE"))
    runs_sorted_l2h = sorted(metric_rows, key=lambda row: value(row, "low_to_high_rate"))
    best_overall = runs_sorted_mae[0]["run_name"] if runs_sorted_mae else ""
    best_low_risk = runs_sorted_l2h[0]["run_name"] if runs_sorted_l2h else ""

    r2c_vs_r1b_quality_guard = bool(
        value(r2c, "MAE") <= value(r1b, "MAE") + 0.10
        and value(r2c, "high_to_low_rate") <= value(r1b, "high_to_low_rate") + 0.10
    )
    r2c_beats_r1b = bool(
        value(r2c, "low_to_high_rate") < value(r1b, "low_to_high_rate")
        and value(r2c, "label2_recall") >= value(r1b, "label2_recall")
        and r2c_vs_r1b_quality_guard
    )
    r2c_preserves_vs_r4b = bool(
        value(r2c, "MAE") <= value(r4b, "MAE") + 0.02 and value(r2c, "QWK") >= value(r4b, "QWK") - 0.02
    )
    d1_comparable = not math.isnan(value(r2c_d1, "pred_ge4_rate_d1_hidden")) and not math.isnan(
        value(r4b_d1, "pred_ge4_rate_d1_hidden")
    )
    r2c_risk_better_than_r4b = value(r2c, "low_to_high_rate") < value(r4b, "low_to_high_rate") or (
        d1_comparable and value(r2c_d1, "pred_ge4_rate_d1_hidden") < value(r4b_d1, "pred_ge4_rate_d1_hidden")
    )
    r2c_beats_r4b = bool(r2c_risk_better_than_r4b and r2c_preserves_vs_r4b)
    proceed_to_r3 = bool(r2c_beats_r1b and r2c_beats_r4b)
    proceed_to_dpo = bool(not proceed_to_r3 and best_low_risk in {"r1b_score_only_balanced", "r4b_shuffled_reason_balanced"})
    reason = (
        "R2c beats R1b and the fair shuffled R4b control; clean reason supervision is promising, so R3 can be considered."
        if proceed_to_r3
        else (
            "Score-only balanced or shuffled-reason control is stronger on low-risk metrics; prefer risk-balanced DPO or target-schema revision before R3."
            if proceed_to_dpo
            else "Second-round ablation is inconclusive; inspect schema quality and per-label behavior before moving to R3/DPO."
        )
    )
    return {
        "best_overall_run": best_overall,
        "best_low_risk_run": best_low_risk,
        "r2c_beats_r1b": r2c_beats_r1b,
        "r2c_beats_r4b": r2c_beats_r4b,
        "proceed_to_r3": proceed_to_r3,
        "proceed_to_dpo": proceed_to_dpo,
        "reason": reason,
        "metrics_used": {
            "r1b_low_to_high": json_metric(r1b.get("low_to_high_rate")),
            "r2n_low_to_high": json_metric(r2n.get("low_to_high_rate")),
            "r2c_low_to_high": json_metric(r2c.get("low_to_high_rate")),
            "r4b_low_to_high": json_metric(r4b.get("low_to_high_rate")),
            "r1b_label2_recall": json_metric(r1b.get("label2_recall")),
            "r2c_label2_recall": json_metric(r2c.get("label2_recall")),
            "r4b_label2_recall": json_metric(r4b.get("label2_recall")),
            "r2c_d1_pred_ge4": json_metric(r2c_d1.get("pred_ge4_rate_d1_hidden")),
            "r4b_d1_pred_ge4": json_metric(r4b_d1.get("pred_ge4_rate_d1_hidden")),
        },
    }


def decision_json(metric_rows: list[dict[str, Any]], d1_rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    if mode == "none":
        return {}
    if mode == "second_round":
        return second_round_decision_json(metric_rows, d1_rows)
    return first_round_decision_json(metric_rows, d1_rows)


def load_runs(args: argparse.Namespace) -> dict[str, str]:
    if args.run_manifest:
        data = json.loads(args.run_manifest.read_text(encoding="utf-8"))
    elif args.run_manifest_json:
        data = json.loads(args.run_manifest_json)
    else:
        return dict(RUNS)
    if isinstance(data, dict):
        return {str(key): str(value) for key, value in data.items()}
    if isinstance(data, list):
        runs: dict[str, str] = {}
        for item in data:
            if not isinstance(item, dict):
                raise ValueError("run manifest list items must be objects")
            name = clean_text(item.get("run_name") or item.get("name"))
            label = clean_text(item.get("run_label") or item.get("label") or name)
            if not name:
                raise ValueError("run manifest item missing run_name")
            runs[name] = label
        return runs
    raise ValueError("run manifest must be a mapping or a list of run objects")


def write_report(
    out_dir: Path,
    metric_rows: list[dict[str, Any]],
    parse_rows: list[dict[str, Any]],
    d1_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    structured_rows: list[dict[str, Any]],
    decision: dict[str, Any],
    report_title: str,
    report_description: str,
    report_path: Path,
) -> None:
    lines = [
        f"# {report_title}",
        "",
        report_description,
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
    if decision:
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                f"- best overall run by MAE: `{decision.get('best_overall_run', '')}`.",
                f"- best low-risk run by low-to-high: `{decision.get('best_low_risk_run', '')}`.",
                f"- proceed to R3: `{decision.get('proceed_to_r3')}`.",
                f"- proceed to DPO: `{decision.get('proceed_to_dpo')}`.",
                f"- recommendation: {decision.get('reason')}",
            ]
        )
    lines.extend(
        [
            "",
            "## D1 Hidden Evaluation",
            "",
            "| run | n | mean pred | pred>=4 | pred=5 | label2 recall | control mean | control-case gap |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in d1_rows:
        lines.append(
            f"| {row['run_label']} | {row['n_d1_cases']} | {fmt(row['mean_pred_d1_hidden'])} | "
            f"{fmt(row['pred_ge4_rate_d1_hidden'])} | {fmt(row['pred_5_rate_d1_hidden'])} | "
            f"{fmt(row['label2_recall_d1'])} | {fmt(row['matched_control_mean_pred'])} | "
            f"{fmt(row['hidden_control_score_gap'])} |"
        )
    lines.extend(["", "## Structured Field Quality", ""])
    for row in structured_rows:
        lines.append(
            f"- {row['run_label']}: full_schema={fmt(row['valid_full_schema_rate'])}, "
            f"major_failures={fmt(row['major_failures_parse_rate'])}, "
            f"score_cap={fmt(row['score_cap_parse_rate'])}, "
            f"rubric_satisfied={fmt(row['rubric_satisfied_parse_rate'])}."
        )
    lines.extend(["", "## Failure Type Evaluation", ""])
    for row in failure_rows:
        lines.append(
            f"- {row['run_label']}: micro-F1={fmt(row['failure_type_micro_f1'])}, "
            f"macro-F1={fmt(row['failure_type_macro_f1'])}, "
            f"D1 nonempty failure={fmt(row['major_failure_nonempty_rate_on_d1_hidden'])}, "
            f"high-control nonempty failure={fmt(row['major_failure_nonempty_rate_on_high_controls'])}."
        )
    lines.extend(
        [
            "",
            "## Evaluation Guardrails",
            "",
            "- Evaluation uses the original question-disjoint dev split, not a balanced training distribution.",
            "- No test split is read by this collection script.",
            "- The collector parses only generated assistant text and gold labels from the dev reference table.",
            "- D1 annotations are used only as evaluation references, not as model prompts or training labels.",
        ]
    )
    write_text(report_path, "\n".join(lines))


def collect(args: argparse.Namespace) -> dict[str, Any]:
    reference = read_csv_rows(args.reference_csv)
    runs = load_runs(args)
    metric_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    parse_rows: list[dict[str, Any]] = []
    metric_group_rows: list[dict[str, Any]] = []
    language_group_rows: list[dict[str, Any]] = []
    d1_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    structured_rows: list[dict[str, Any]] = []
    parsed_rows_all: list[dict[str, Any]] = []
    prediction_files: dict[str, str] = {}

    for run_name, run_label in runs.items():
        run_dir = args.prediction_root / run_name
        prediction_file = find_prediction_file(run_dir)
        prediction_files[run_name] = str(prediction_file)
        prediction_records = load_prediction_records(prediction_file)
        parsed_rows = align_predictions(reference, prediction_records, run_name, run_label)
        split = parsed_rows[0].get("split", "dev") if parsed_rows else "dev"
        metric_rows.append(metric_summary(parsed_rows, run_name, run_label, str(split)))
        label_rows.extend(label_summary(parsed_rows, run_name, run_label, str(split)))
        parse_rows.append(parse_summary(parsed_rows, run_name, run_label, str(split), prediction_file))
        if args.write_structured_eval:
            structured_rows.append(structured_parse_summary(parsed_rows, run_name, run_label))
            if resolve_d1_base(args.d1_dir).exists():
                d1_rows.append(d1_eval_row(parsed_rows, args.d1_dir, run_name, run_label))
                failure_rows.append(failure_type_eval_row(parsed_rows, args.d1_dir, run_name, run_label))
        metric_group_rows.extend(grouped_metric_rows(parsed_rows, "metric", run_name, run_label))
        language_group_rows.extend(grouped_metric_rows(parsed_rows, "language", run_name, run_label))
        parsed_rows_all.extend(parsed_rows)

    tables_dir = args.out_dir / "tables"
    prefix = args.file_prefix
    write_csv(tables_dir / f"{prefix}_metrics.csv", metric_rows, METRIC_FIELDS)
    write_csv(tables_dir / f"{prefix}_label_metrics.csv", label_rows, LABEL_FIELDS)
    write_csv(tables_dir / f"{prefix}_parse_summary.csv", parse_rows, PARSE_FIELDS)
    write_csv(tables_dir / f"{prefix}_by_metric.csv", metric_group_rows)
    write_csv(tables_dir / f"{prefix}_by_language.csv", language_group_rows)
    decision = decision_json(metric_rows, d1_rows, args.decision_mode) if args.write_structured_eval else {}
    if args.write_structured_eval:
        write_csv(tables_dir / f"{prefix}_d1_hidden_eval.csv", d1_rows, D1_FIELDS)
        write_csv(tables_dir / f"{prefix}_failure_type_eval.csv", failure_rows, FAILURE_TYPE_FIELDS)
        write_csv(tables_dir / f"{prefix}_structured_parse.csv", structured_rows, STRUCTURED_PARSE_FIELDS)
        write_json(args.out_dir / "decision" / f"{prefix}_decision.json", decision)
    if args.write_parsed_csv:
        write_csv(args.out_dir / "diagnostics" / f"{prefix}_parsed_predictions.csv", parsed_rows_all)
    write_json(args.out_dir / "reports" / f"{prefix}_prediction_files.json", prediction_files)
    report_title = args.report_title or "Exp19 First-Round SFT Dev Evaluation"
    report_description = args.report_description or (
        "This report summarizes LLaMA-Factory `do_predict` outputs on the original dev split."
    )
    write_report(
        args.out_dir,
        metric_rows,
        parse_rows,
        d1_rows,
        failure_rows,
        structured_rows,
        decision,
        report_title,
        report_description,
        args.out_dir / "reports" / f"{prefix}_report.md",
    )
    return {"runs": len(metric_rows), "prediction_files": prediction_files, "decision": decision}


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Exp19 first-round SFT dev predictions.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_OUTPUT_DIR / "dev_predictions")
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--d1-dir", type=Path, default=DEFAULT_D1_DIR)
    parser.add_argument("--run-manifest", type=Path, default=None)
    parser.add_argument("--run-manifest-json", default="")
    parser.add_argument("--file-prefix", default="exp19_sft_first_round_dev")
    parser.add_argument("--report-title", default="")
    parser.add_argument("--report-description", default="")
    parser.add_argument("--decision-mode", choices=["first_round", "second_round", "none"], default="first_round")
    parser.add_argument("--write-parsed-csv", action="store_true")
    parser.add_argument("--write-structured-eval", action="store_true")
    args = parser.parse_args()
    summary = collect(args)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
