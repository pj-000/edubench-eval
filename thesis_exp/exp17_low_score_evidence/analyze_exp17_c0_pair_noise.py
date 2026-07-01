"""Audit Exp17-C0 pair noise before pairwise training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.src.edujudge.utils.io import read_jsonl, relpath, write_csv, write_text


DEFAULT_PAIRS = Path("thesis_exp/exp17_low_score_evidence/outputs/exp17_a0_train_hidden_failure_signals_seed42/train_hidden_failure_pairs.csv")
DEFAULT_TRAIN = Path("thesis_exp/data/splits/question_seed42/train.jsonl")
DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp17_c0_pair_noise_audit_seed42")

PAIR_SOURCES = [
    "all_a0_pairs",
    "same_subject_only",
    "high_weight_only_p75",
    "same_subject_high_weight_p75",
    "exclude_format_auxiliary",
    "exclude_answer_key_dependent",
    "weak_evidence_only",
    "missing_key_point_only",
    "factual_or_rubric_mismatch_only",
    "insufficient_evidence_only",
    "random_low_high_pairs",
    "random_matched_metric_rubric",
    "random_matched_metric_rubric_subject",
    "same_question_group_upper_bound",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except Exception:
        return default


def rate(count: int, total: int) -> float:
    return count / total if total else float("nan")


def pct(values: list[float], q: float) -> float:
    values = [value for value in values if not math.isnan(value)]
    return float(np.percentile(values, q)) if values else float("nan")


def text_hash(value: Any) -> str:
    return hashlib.sha1(str(value or "").strip().encode("utf-8")).hexdigest()


def norm_sample(row: dict[str, Any]) -> dict[str, Any]:
    metric = str(row.get("metric_canonical") or row.get("metric") or row.get("metric_abbr") or "").strip()
    subject = str(row.get("subject_canonical") or row.get("subject") or "").strip()
    language = str(row.get("language") or "").strip()
    rubric = str(row.get("rubric_text") or row.get("rubric") or row.get("rubric_canonical") or "").strip()
    sample_id = str(row.get("record_id") or row.get("id") or row.get("sample_id") or "").strip()
    question_key = str(row.get("question_key") or row.get("source_question_key") or "").strip()
    return {
        "sample_id": sample_id,
        "metric": metric,
        "subject": subject,
        "language": language,
        "rubric_hash": text_hash(rubric),
        "question_group": str(row.get("question_group_id") or question_key).strip(),
        "label": int(row.get("label_5", row.get("label", 0))),
    }


def status_for_count(n: int) -> str:
    if n < 10:
        return "unusable"
    if n < 30:
        return "too_small_for_training_but_ok_for_diagnostic"
    return "ok"


def candidate_counts(rows: list[dict[str, str]], train_samples: list[dict[str, Any]]) -> dict[str, int]:
    valid_rows = rows
    weights = [safe_float(row.get("pair_weight")) for row in valid_rows]
    p75 = pct(weights, 75)
    counts = {
        "all_a0_pairs": len(valid_rows),
        "same_subject_only": sum(1 for row in valid_rows if truthy(row.get("same_subject"))),
        "high_weight_only_p75": sum(1 for row in valid_rows if safe_float(row.get("pair_weight"), 0.0) >= p75),
        "same_subject_high_weight_p75": sum(
            1 for row in valid_rows if truthy(row.get("same_subject")) and safe_float(row.get("pair_weight"), 0.0) >= p75
        ),
        "exclude_format_auxiliary": sum(
            1
            for row in valid_rows
            if row.get("low_candidate_type") != "format_auxiliary" and row.get("low_failure_mode_auto") != "format_violation"
        ),
        "exclude_answer_key_dependent": sum(
            1
            for row in valid_rows
            if row.get("low_candidate_type") != "answer_key_dependent"
            and row.get("low_failure_mode_auto") != "answer_key_or_reference_mismatch"
        ),
        "weak_evidence_only": sum(1 for row in valid_rows if row.get("low_candidate_type") in {"weak_evidence_positive", "strong_evidence_positive"}),
        "missing_key_point_only": sum(1 for row in valid_rows if "missing_key_point" in row.get("low_failure_mode_auto", "")),
        "factual_or_rubric_mismatch_only": sum(
            1
            for row in valid_rows
            if "factual_or_rubric_mismatch" in row.get("low_failure_mode_auto", "")
            or "answer_key_or_reference_mismatch" in row.get("low_failure_mode_auto", "")
        ),
        "insufficient_evidence_only": sum(1 for row in valid_rows if "insufficient_evidence" in row.get("low_failure_mode_auto", "")),
    }
    lows = [sample for sample in train_samples if sample["label"] <= 2]
    highs = [sample for sample in train_samples if sample["label"] >= 4]
    counts["random_low_high_pairs"] = len(valid_rows) if lows and highs else 0
    for source, include_subject in [
        ("random_matched_metric_rubric", False),
        ("random_matched_metric_rubric_subject", True),
    ]:
        high_keys = {
            (
                high["metric"],
                high["language"],
                high["rubric_hash"],
                high["subject"] if include_subject else "",
            )
            for high in highs
        }
        matched_low = [
            low
            for low in lows
            if (
                low["metric"],
                low["language"],
                low["rubric_hash"],
                low["subject"] if include_subject else "",
            )
            in high_keys
        ]
        target = counts["same_subject_only"] if include_subject else len(valid_rows)
        counts[source] = target if matched_low else 0
    grouped: dict[tuple[str, str, str], dict[str, int]] = defaultdict(lambda: {"low": 0, "high": 0})
    for sample in train_samples:
        key = (sample["question_group"], sample["metric"], sample["rubric_hash"])
        if sample["label"] <= 2:
            grouped[key]["low"] += 1
        elif sample["label"] >= 4:
            grouped[key]["high"] += 1
    counts["same_question_group_upper_bound"] = sum(min(bucket["low"] * bucket["high"], bucket["low"] * 5) for bucket in grouped.values())
    return counts


def field_distribution(rows: list[dict[str, str]], train_samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sample_by_id = {sample["sample_id"]: sample for sample in train_samples}
    total = len(rows)
    outputs: list[dict[str, Any]] = []
    for field in ["low_candidate_type", "recommended_pair_use", "low_failure_mode_auto"]:
        for value, n in Counter(row.get(field, "") for row in rows).most_common():
            outputs.append({"field_name": field, "value": value, "n": n, "rate": rate(n, total)})
    for field in ["metric", "subject", "language"]:
        counter: Counter[str] = Counter()
        for row in rows:
            low = sample_by_id.get(row.get("low_sample_id", ""))
            if low:
                counter[str(low.get(field, ""))] += 1
        for value, n in counter.most_common():
            outputs.append({"field_name": field, "value": value, "n": n, "rate": rate(n, total)})
    return outputs


def build_report(summary: dict[str, Any], counts: list[dict[str, Any]]) -> str:
    strict_ok = [row["pair_source"] for row in counts if row["status"] == "ok" and row["pair_source"] not in {"random_low_high_pairs"}]
    too_small = [row["pair_source"] for row in counts if row["status"] != "ok"]
    direct = "no; run all pairs only as a preliminary baseline with noise-control ablations"
    return "\n".join(
        [
            "# Exp17-C0 Pair Noise Audit",
            "",
            "This audit reads train-side pair metadata only. It does not train a model and does not read test.",
            "",
            "## Summary",
            "",
            f"- n_pairs: {summary['n_pairs']}",
            f"- same_metric_rate: {summary['same_metric_rate']:.4f}",
            f"- same_language_rate: {summary['same_language_rate']:.4f}",
            f"- same_rubric_hash_rate: {summary['same_rubric_hash_rate']:.4f}",
            f"- same_subject_rate: {summary['same_subject_rate']:.4f}",
            f"- same_question_group_rate: {summary['same_question_group_rate']:.4f}",
            f"- same_boundary_key_rate: {summary['same_boundary_key_rate']:.4f}",
            f"- pair_weight_p50: {summary['pair_weight_p50']:.4f}",
            f"- pair_weight_p75: {summary['pair_weight_p75']:.4f}",
            f"- pair_weight_p90: {summary['pair_weight_p90']:.4f}",
            "",
            "## Interpretation",
            "",
            "- Current pairs are mainly cross-question pairs if same_question_group_rate is near 0.",
            "- same_subject controls whether cross-subject noise is likely to enter the pairwise signal.",
            "- high_weight_only_p75 tests whether A0 pair confidence helps filter noisy preferences.",
            f"- strict filters with enough pairs: {', '.join(strict_ok) if strict_ok else 'none'}",
            f"- unavailable or too small pair sources: {', '.join(too_small) if too_small else 'none'}",
            f"- recommended_direct_all_pairs_only: {direct}",
            "",
            "## Recommended C0 Scout Configs",
            "",
            "- C0_0_ordinal_continue",
            "- C0_1/C0_2/C0_3 all_a0_pairs",
            "- C0_7 same_subject_only",
            "- C0_8 high_weight_only_p75",
            "- C0_9 same_subject_high_weight_p75",
            "- C0_10 exclude_format_auxiliary",
            "- C0_11 exclude_answer_key_dependent",
            "- C0_12 random_matched_metric_rubric",
            "- C0_13 random_matched_metric_rubric_subject",
            "- C0_14 same_question_group_upper_bound only as an upper-bound diagnostic if available",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Exp17-C0 pair noise and candidate pair-source sizes.")
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    del args.seed

    rows = read_csv_rows(args.pairs)
    train_samples = [norm_sample(row) for row in read_jsonl(args.train_jsonl)] if args.train_jsonl.exists() else []
    weights = [safe_float(row.get("pair_weight")) for row in rows]
    summary = {
        "n_pairs": len(rows),
        "same_metric_rate": rate(sum(1 for row in rows if truthy(row.get("same_metric"))), len(rows)),
        "same_language_rate": rate(sum(1 for row in rows if truthy(row.get("same_language"))), len(rows)),
        "same_rubric_hash_rate": rate(sum(1 for row in rows if truthy(row.get("same_rubric_hash"))), len(rows)),
        "same_subject_rate": rate(sum(1 for row in rows if truthy(row.get("same_subject"))), len(rows)),
        "same_question_group_rate": rate(sum(1 for row in rows if truthy(row.get("same_question_group"))), len(rows)),
        "same_boundary_key_rate": rate(sum(1 for row in rows if truthy(row.get("same_boundary_key"))), len(rows)),
        "pair_weight_mean": float(np.nanmean(weights)) if weights else float("nan"),
        "pair_weight_p25": pct(weights, 25),
        "pair_weight_p50": pct(weights, 50),
        "pair_weight_p75": pct(weights, 75),
        "pair_weight_p90": pct(weights, 90),
        "pair_weight_max": max(weights) if weights else float("nan"),
    }
    counts_map = candidate_counts(rows, train_samples)
    counts = [{"pair_source": source, "available_pair_count": counts_map.get(source, 0), "status": status_for_count(counts_map.get(source, 0))} for source in PAIR_SOURCES]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "pair_noise_summary.csv", [summary])
    write_csv(args.out_dir / "pair_source_candidate_counts.csv", counts)
    write_csv(args.out_dir / "pair_field_distribution.csv", field_distribution(rows, train_samples))
    (args.out_dir / "recommended_c0_configs.json").write_text(
        json.dumps({"recommended_configs": [row["pair_source"] for row in counts if row["status"] != "unusable"]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_text(args.out_dir / "exp17_c0_pair_noise_audit_report.md", build_report(summary, counts))
    print(json.dumps({"status": "COMPLETED", "out_dir": relpath(args.out_dir), "n_pairs": len(rows)}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
