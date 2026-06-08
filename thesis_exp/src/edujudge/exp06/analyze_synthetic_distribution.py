"""Analyze normalized synthetic candidate distributions for Exp6."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from thesis_exp.src.edujudge.exp06 import EXP06_TABLES_DIR, ensure_exp06_dirs
from thesis_exp.src.edujudge.exp06.common import read_csv_rows, write_rows


def label_value(row: dict[str, str]) -> str:
    return str(row.get("target_label_5") or "missing")


def is_low_label(value: str) -> bool:
    return value in {"1", "2"}


def grouped_counts(rows: list[dict[str, str]], keys: list[str]) -> list[dict[str, Any]]:
    counter: Counter[tuple[str, ...]] = Counter()
    for row in rows:
        values = []
        for key in keys:
            if key == "target_label_5":
                values.append(label_value(row))
            else:
                values.append(str(row.get(key) or "missing"))
        counter[tuple(values)] += 1
    total = sum(counter.values())
    out = []
    for values, count in sorted(counter.items(), key=lambda item: item[0]):
        item = {key: values[idx] for idx, key in enumerate(keys)}
        item["count"] = count
        item["pct"] = round(count / total, 6) if total else 0
        out.append(item)
    return out


def build_recommendations(
    candidates: list[dict[str, str]],
    inventory: list[dict[str, str]],
    leakage: list[dict[str, str]],
) -> list[dict[str, Any]]:
    inv_by_path = {str(row.get("source_path") or ""): row for row in inventory}
    leakage_by_source = {str(row.get("source_file") or ""): row for row in leakage}
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        by_source[str(row.get("source_file") or "")].append(row)

    rows: list[dict[str, Any]] = []
    for source, group in sorted(by_source.items(), key=lambda item: str(item[0] or "")):
        inv = inv_by_path.get(source, {})
        leak = leakage_by_source.get(source, {})
        labels = [label_value(row) for row in group]
        label_rows = sum(1 for value in labels if value != "missing")
        low_rows = sum(1 for value in labels if is_low_label(value))
        error_rows = sum(1 for row in group if row.get("error_type"))
        statuses = Counter(str(row.get("normalization_status") or "") for row in group)
        role = inv.get("likely_role", "")
        dev_test_leak = str(leak.get("any_dev_test_leakage", "")).lower() == "true"
        blocked_reasons = []
        if dev_test_leak:
            blocked_reasons.append("exact dev/test overlap detected")
        if role == "model_judge_output" or "model_judge_output" in statuses:
            blocked_reasons.append("model/judge output, not human label")
        if "groupby_metric_" in source or source.endswith("merge_model_metric.jsonl") or source.endswith("deepseek-r1_merged.jsonl"):
            blocked_reasons.append("required blocked judge/model source")
        if source.endswith("human_sampled_eval_sft_criteria_test.json"):
            blocked_reasons.append("test-style human-sampled SFT file")
        if source.endswith("sampled_merge_50_new.json") or source.endswith("sampled_merge_50_new_swift.json"):
            blocked_reasons.append("required HIGH risk sampled_merge source")
        if label_rows == 0:
            blocked_reasons.append("no target_label_5")
        if low_rows == 0:
            blocked_reasons.append("no low-score target_label_5 rows")

        if blocked_reasons and ("required HIGH risk sampled_merge source" not in blocked_reasons):
            recommendation = "BLOCKED_OR_REVIEW_ONLY"
        elif blocked_reasons:
            recommendation = "HIGH_RISK_REVIEW_ONLY"
        elif label_rows and low_rows:
            recommendation = "POSSIBLE_FILTERED_TRAIN_ONLY_AFTER_MANUAL_CONFIRMATION"
        else:
            recommendation = "REVIEW_ONLY"

        rows.append(
            {
                "source_file": source,
                "likely_role": role or "unknown",
                "candidate_rows": len(group),
                "target_label_5_rows": label_rows,
                "low_score_rows": low_rows,
                "error_type_rows": error_rows,
                "normalization_status_counts": dict(statuses),
                "dev_test_leakage": dev_test_leak,
                "leakage_risk": leak.get("leakage_risk", "unknown"),
                "label_reliability_status": "unverified_human_or_model_generated",
                "recommended_use": recommendation,
                "allowed_split": "train_only_if_approved",
                "blocked_reasons": "; ".join(blocked_reasons),
            }
        )
    return rows


def main() -> None:
    ensure_exp06_dirs()
    candidates = read_csv_rows(EXP06_TABLES_DIR / "synthetic_candidate_rows.csv")
    inventory = read_csv_rows(EXP06_TABLES_DIR / "synthetic_source_inventory.csv")
    leakage = read_csv_rows(EXP06_TABLES_DIR / "synthetic_leakage_summary.csv")

    write_rows(
        EXP06_TABLES_DIR / "synthetic_score_distribution.csv",
        grouped_counts(candidates, ["source_file", "target_label_5"]),
        ["source_file", "target_label_5", "count", "pct"],
    )
    write_rows(
        EXP06_TABLES_DIR / "synthetic_metric_distribution.csv",
        grouped_counts(candidates, ["source_file", "metric_canonical", "target_label_5"]),
        ["source_file", "metric_canonical", "target_label_5", "count", "pct"],
    )
    write_rows(
        EXP06_TABLES_DIR / "synthetic_language_distribution.csv",
        grouped_counts(candidates, ["source_file", "language", "target_label_5"]),
        ["source_file", "language", "target_label_5", "count", "pct"],
    )
    write_rows(
        EXP06_TABLES_DIR / "synthetic_error_type_distribution.csv",
        grouped_counts(candidates, ["source_file", "error_type", "target_label_5"]),
        ["source_file", "error_type", "target_label_5", "count", "pct"],
    )
    recommendations = build_recommendations(candidates, inventory, leakage)
    write_rows(
        EXP06_TABLES_DIR / "synthetic_filter_recommendation.csv",
        recommendations,
        [
            "source_file",
            "likely_role",
            "candidate_rows",
            "target_label_5_rows",
            "low_score_rows",
            "error_type_rows",
            "normalization_status_counts",
            "dev_test_leakage",
            "leakage_risk",
            "label_reliability_status",
            "recommended_use",
            "allowed_split",
            "blocked_reasons",
        ],
    )
    print(f"Wrote synthetic distribution tables to {EXP06_TABLES_DIR}")


if __name__ == "__main__":
    main()
