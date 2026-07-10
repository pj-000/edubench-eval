"""Analyze Exp27J blind reviews and final adjudications."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.src.edujudge.utils.io import write_csv, write_json, write_text  # noqa: E402


DEFAULT_OUT_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27j_independent_audit_seed42"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_table(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def as_score(value: Any) -> int | None:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return score if 1 <= score <= 5 else None


def safe_div(num: float, den: float) -> float | None:
    return num / den if den else None


def fmt(value: float | None, digits: int = 6) -> str:
    return "" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def ranges_overlap(a: list[int], b: list[int]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def needs_adjudication(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return any(
        [
            a.get("most_plausible_score") != b.get("most_plausible_score"),
            not ranges_overlap(a.get("score_range", [1, 5]), b.get("score_range", [1, 5])),
            a.get("failure_bucket") != b.get("failure_bucket"),
            bool(a.get("needs_adjudication")),
            bool(b.get("needs_adjudication")),
        ]
    )


def quadratic_weighted_kappa(a: list[int], b: list[int]) -> float | None:
    if len(a) != len(b) or not a:
        return None
    n = len(a)
    observed = [[0.0 for _ in range(5)] for _ in range(5)]
    hist_a = [0.0] * 5
    hist_b = [0.0] * 5
    for left, right in zip(a, b):
        observed[left - 1][right - 1] += 1.0
        hist_a[left - 1] += 1.0
        hist_b[right - 1] += 1.0
    obs = 0.0
    exp = 0.0
    for i in range(5):
        for j in range(5):
            weight = ((i - j) / 4.0) ** 2
            obs += weight * observed[i][j] / n
            exp += weight * (hist_a[i] * hist_b[j]) / (n * n)
    return 1.0 - obs / exp if exp else (1.0 if obs == 0 else None)


def ordinal_alpha(a: list[int], b: list[int]) -> float | None:
    if len(a) != len(b) or not a:
        return None
    observed = statistics.fmean((left - right) ** 2 for left, right in zip(a, b))
    ratings = a + b
    if len(ratings) < 2:
        return None
    expected = sum(
        (left - right) ** 2
        for left_index, left in enumerate(ratings)
        for right_index, right in enumerate(ratings)
        if left_index != right_index
    ) / (len(ratings) * (len(ratings) - 1))
    return 1.0 - observed / expected if expected else (1.0 if observed == 0 else None)


def pair_metrics(a: list[int], b: list[int]) -> dict[str, float | None]:
    if not a:
        return {"mae": None, "exact": None, "within_one": None, "qwk": None, "signed_bias": None}
    return {
        "mae": statistics.fmean(abs(left - right) for left, right in zip(a, b)),
        "exact": statistics.fmean(left == right for left, right in zip(a, b)),
        "within_one": statistics.fmean(abs(left - right) <= 1 for left, right in zip(a, b)),
        "qwk": quadratic_weighted_kappa(a, b),
        "signed_bias": statistics.fmean(left - right for left, right in zip(a, b)),
    }


def source_metric_row(
    source: str,
    view: str,
    predictions: list[int],
    gold: list[int],
    expected_n: int,
) -> dict[str, Any]:
    metrics = pair_metrics(predictions, gold)
    low_indices = [idx for idx, value in enumerate(gold) if value <= 2]
    high_indices = [idx for idx, value in enumerate(gold) if value >= 4]
    row: dict[str, Any] = {
        "source": source,
        "view": view,
        "n": len(gold),
        "expected_n": expected_n,
        "coverage_rate": fmt(safe_div(len(gold), expected_n)),
        **{key: fmt(value) for key, value in metrics.items()},
        "low_to_high_count": sum(predictions[idx] >= 4 for idx in low_indices),
        "low_to_high_rate": fmt(safe_div(sum(predictions[idx] >= 4 for idx in low_indices), len(low_indices))),
        "high_to_low_count": sum(predictions[idx] <= 2 for idx in high_indices),
        "high_to_low_rate": fmt(safe_div(sum(predictions[idx] <= 2 for idx in high_indices), len(high_indices))),
    }
    for label in [1, 2, 5]:
        indices = [idx for idx, value in enumerate(gold) if value == label]
        row[f"label{label}_recall"] = fmt(
            safe_div(sum(predictions[idx] == label for idx in indices), len(indices))
        )
    return row


def human_aggregate(row: dict[str, Any], method: str) -> int | None:
    values = [
        float(value)
        for value in [row.get("human_1"), row.get("human_2"), row.get("human_3")]
        if value is not None
    ]
    if not values:
        return None
    aggregate = statistics.fmean(values) if method == "mean" else statistics.median(values)
    return as_score(aggregate)


def naive_three_way_median(row: dict[str, Any]) -> int | None:
    values = [row.get("original_label_5"), row.get("qwen_score"), row.get("deepseek_score")]
    if any(value is None for value in values):
        return None
    return as_score(statistics.median(float(value) for value in values))


def final_reference(
    packets: dict[str, dict[str, Any]],
    reviewer_a: dict[str, dict[str, Any]],
    reviewer_b: dict[str, dict[str, Any]],
    adjudicated: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    final: dict[str, dict[str, Any]] = {}
    templates: list[dict[str, Any]] = []
    for sid, packet in packets.items():
        a = reviewer_a.get(sid)
        b = reviewer_b.get(sid)
        if not a or not b:
            continue
        requires = needs_adjudication(a, b)
        if requires:
            if sid in adjudicated:
                row = adjudicated[sid]
                final[sid] = {
                    "score": int(row["final_score"]),
                    "score_range": row["final_score_range"],
                    "failure_bucket": row["final_failure_bucket"],
                    "major_failures": row["final_major_failures"],
                    "confidence": row["final_confidence"],
                    "source": "final_adjudication",
                }
            else:
                templates.append(
                    {
                        "sample_id": sid,
                        "review_packet_hash": packet["review_packet_hash"],
                        "blind_packet": packet,
                        "reviewer_a": a,
                        "reviewer_b": b,
                        "final_score_range": None,
                        "final_score": None,
                        "final_failure_bucket": None,
                        "final_major_failures": None,
                        "final_rubric_evidence": None,
                        "final_output_evidence": None,
                        "final_confidence": None,
                        "ambiguity_type": None,
                        "adjudication_reason": None,
                    }
                )
        else:
            final[sid] = {
                "score": int(a["most_plausible_score"]),
                "score_range": [
                    max(a["score_range"][0], b["score_range"][0]),
                    min(a["score_range"][1], b["score_range"][1]),
                ],
                "failure_bucket": a["failure_bucket"],
                "major_failures": sorted(set(a.get("major_failures", []) + b.get("major_failures", []))),
                "confidence": "high" if a.get("confidence") == b.get("confidence") == "high" else "medium",
                "source": "blind_reviewer_consensus",
            }
    return final, templates


def weighted_rate(rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> float | None:
    weights = [float(row.get("design_weight") or 0.0) for row in rows]
    den = sum(weights)
    return sum(weight * bool(predicate(row)) for row, weight in zip(rows, weights)) / den if den else None


def cluster_bootstrap_ci(
    rows: list[dict[str, Any]],
    statistic: Callable[[list[dict[str, Any]]], float | None],
    seed: int,
    resamples: int,
) -> tuple[float | None, float | None]:
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[str(row["question_key"])].append(row)
    keys = sorted(clusters)
    if not keys:
        return None, None
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(resamples):
        sampled: list[dict[str, Any]] = []
        for key in rng.choices(keys, k=len(keys)):
            sampled.extend(clusters[key])
        value = statistic(sampled)
        if value is not None and math.isfinite(value):
            estimates.append(value)
    if not estimates:
        return None, None
    estimates.sort()
    low = estimates[int(0.025 * (len(estimates) - 1))]
    high = estimates[int(0.975 * (len(estimates) - 1))]
    return low, high


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out_dir
    packet_rows = read_jsonl(out_dir / "packets" / "exp27j_all_blind_packets.jsonl")
    packets = {str(row["sample_id"]): row for row in packet_rows}
    private_rows = read_jsonl(out_dir / "private" / "exp27j_original_teacher_calibration_reference.jsonl")
    private = {str(row["sample_id"]): row for row in private_rows}
    reviewer_a_rows = read_jsonl(args.reviewer_a)
    reviewer_b_rows = read_jsonl(args.reviewer_b)
    adjudication_rows = read_jsonl(args.adjudication)
    provenance = (
        json.loads(args.review_provenance.read_text(encoding="utf-8"))
        if args.review_provenance.exists()
        else {"reference_status": "unspecified"}
    )
    reviewer_a = {str(row["sample_id"]): row for row in reviewer_a_rows}
    reviewer_b = {str(row["sample_id"]): row for row in reviewer_b_rows}
    adjudicated = {str(row["sample_id"]): row for row in adjudication_rows}

    completion_rows = [
        {"artifact": "blind_packets", "completed_rows": len(packet_rows), "expected_rows": 180},
        {"artifact": "reviewer_a", "completed_rows": len(reviewer_a_rows), "expected_rows": 180},
        {"artifact": "reviewer_b", "completed_rows": len(reviewer_b_rows), "expected_rows": 180},
        {
            "artifact": "final_adjudication",
            "completed_rows": len(adjudication_rows),
            "expected_rows": "computed_after_dual_review",
        },
    ]
    write_csv(out_dir / "tables" / "exp27j_review_completion.csv", completion_rows)

    final, adjudication_templates = final_reference(packets, reviewer_a, reviewer_b, adjudicated)
    if reviewer_a and reviewer_b:
        write_jsonl(
            out_dir / "annotation" / "exp27j_final_adjudication_template.jsonl",
            adjudication_templates,
        )

    manifest_rows: list[dict[str, Any]] = []
    for sid, ref in sorted(final.items()):
        meta = private.get(sid, {})
        a = reviewer_a.get(sid, {})
        b = reviewer_b.get(sid, {})
        manifest_rows.append(
            {
                "sample_id": sid,
                "question_key_hash": hashlib.sha1(
                    str(packets[sid]["question_key"]).encode("utf-8")
                ).hexdigest(),
                "view": meta.get("view"),
                "design_weight": meta.get("design_weight"),
                "original_score": meta.get("original_label_5"),
                "qwen_score": meta.get("qwen_score"),
                "deepseek_score": meta.get("deepseek_score"),
                "exp27i_calibrated_score": meta.get("exp27i_calibrated_score"),
                "exp27i_training_use": meta.get("exp27i_recommended_training_use"),
                "reviewer_a_score": a.get("most_plausible_score"),
                "reviewer_b_score": b.get("most_plausible_score"),
                "required_adjudication": needs_adjudication(a, b) if a and b else "",
                "final_score": ref.get("score"),
                "final_score_range": json.dumps(ref.get("score_range"), ensure_ascii=False),
                "final_failure_bucket": ref.get("failure_bucket"),
                "final_confidence": ref.get("confidence"),
                "final_reference_source": ref.get("source"),
            }
        )
    write_table(
        out_dir / "tables" / "exp27j_adjudicated_manifest.csv",
        manifest_rows,
        [
            "sample_id",
            "question_key_hash",
            "view",
            "design_weight",
            "original_score",
            "qwen_score",
            "deepseek_score",
            "exp27i_calibrated_score",
            "exp27i_training_use",
            "reviewer_a_score",
            "reviewer_b_score",
            "required_adjudication",
            "final_score",
            "final_score_range",
            "final_failure_bucket",
            "final_confidence",
            "final_reference_source",
        ],
    )
    distribution_counts: Counter[tuple[str, int, str, str, str]] = Counter(
        (
            str(private.get(sid, {}).get("view") or "unknown"),
            int(ref["score"]),
            str(ref.get("failure_bucket") or "unknown"),
            str(ref.get("confidence") or "unknown"),
            str(ref.get("source") or "unknown"),
        )
        for sid, ref in final.items()
    )
    write_table(
        out_dir / "tables" / "exp27j_final_reference_distribution.csv",
        [
            {
                "view": key[0],
                "final_score": key[1],
                "failure_bucket": key[2],
                "final_confidence": key[3],
                "final_reference_source": key[4],
                "count": value,
            }
            for key, value in sorted(distribution_counts.items())
        ],
        ["view", "final_score", "failure_bucket", "final_confidence", "final_reference_source", "count"],
    )

    agreement_rows: list[dict[str, Any]] = []
    common = sorted(set(reviewer_a) & set(reviewer_b))
    reviewer_qwk: float | None = None
    reviewer_alpha: float | None = None
    if common:
        for view in ["all", "representative", "risk_enriched"]:
            ids_for_view = [
                sid for sid in common if view == "all" or private.get(sid, {}).get("view") == view
            ]
            a_scores = [int(reviewer_a[sid]["most_plausible_score"]) for sid in ids_for_view]
            b_scores = [int(reviewer_b[sid]["most_plausible_score"]) for sid in ids_for_view]
            qwk = quadratic_weighted_kappa(a_scores, b_scores)
            alpha = ordinal_alpha(a_scores, b_scores)
            metrics = pair_metrics(a_scores, b_scores)
            if view == "all":
                reviewer_qwk = qwk
                reviewer_alpha = alpha
            agreement_rows.append(
                {
                    "reviewer_pair": "a_vs_b",
                    "view": view,
                    "n": len(ids_for_view),
                    "qwk": fmt(qwk),
                    "krippendorff_ordinal_alpha": fmt(alpha),
                    "exact_agreement": fmt(metrics["exact"]),
                    "within_one_agreement": fmt(metrics["within_one"]),
                    "mae": fmt(metrics["mae"]),
                    "score_range_overlap": fmt(
                        statistics.fmean(
                            ranges_overlap(reviewer_a[sid]["score_range"], reviewer_b[sid]["score_range"])
                            for sid in ids_for_view
                        )
                    ),
                    "adjudication_required_rate": fmt(
                        statistics.fmean(
                            needs_adjudication(reviewer_a[sid], reviewer_b[sid]) for sid in ids_for_view
                        )
                    ),
                }
            )
    write_table(
        out_dir / "tables" / "exp27j_inter_reviewer_agreement.csv",
        agreement_rows,
        [
            "reviewer_pair",
            "view",
            "n",
            "qwk",
            "krippendorff_ordinal_alpha",
            "exact_agreement",
            "within_one_agreement",
            "mae",
            "score_range_overlap",
            "adjudication_required_rate",
        ],
    )

    source_rows: list[dict[str, Any]] = []
    if final:
        source_getters: dict[str, Callable[[dict[str, Any]], int | None]] = {
            "original_label_5": lambda row: as_score(row.get("original_label_5")),
            "human_mean": lambda row: human_aggregate(row, "mean"),
            "human_median": lambda row: human_aggregate(row, "median"),
            "qwen": lambda row: as_score(row.get("qwen_score")),
            "deepseek": lambda row: as_score(row.get("deepseek_score")),
            "naive_human_qwen_deepseek_median": naive_three_way_median,
            "exp27i_calibrated": lambda row: as_score(row.get("exp27i_calibrated_score")),
        }
        for source, getter in source_getters.items():
            for view in ["all", "representative", "risk_enriched"]:
                predictions: list[int] = []
                gold: list[int] = []
                for sid, ref in final.items():
                    meta = private.get(sid, {})
                    if view != "all" and meta.get("view") != view:
                        continue
                    pred = getter(meta)
                    if pred is None:
                        continue
                    predictions.append(pred)
                    gold.append(int(ref["score"]))
                expected_n = sum(
                    view == "all" or private.get(sid, {}).get("view") == view for sid in final
                )
                source_rows.append(source_metric_row(source, view, predictions, gold, expected_n))
    write_table(
        out_dir / "tables" / "exp27j_source_vs_adjudicated.csv",
        source_rows,
        [
            "source",
            "view",
            "n",
            "expected_n",
            "coverage_rate",
            "mae",
            "exact",
            "within_one",
            "qwk",
            "signed_bias",
            "low_to_high_count",
            "low_to_high_rate",
            "high_to_low_count",
            "high_to_low_rate",
            "label1_recall",
            "label2_recall",
            "label5_recall",
        ],
    )

    tier_rows: list[dict[str, Any]] = []
    tier_analysis_records: list[dict[str, Any]] = []
    if final:
        groups: dict[str, list[tuple[int, int, dict[str, Any]]]] = defaultdict(list)
        source_groups: dict[str, list[tuple[int, int, dict[str, Any]]]] = defaultdict(list)
        for sid, ref in final.items():
            meta = private.get(sid, {})
            pred = as_score(meta.get("exp27i_calibrated_score"))
            if pred is None:
                continue
            tier = str(meta.get("exp27i_recommended_training_use") or "not_audited")
            groups[tier].append((pred, int(ref["score"]), ref))
            calibration_source = str(meta.get("exp27i_calibration_source") or "not_audited")
            source_groups[calibration_source].append((pred, int(ref["score"]), ref))
            tier_analysis_records.append(
                {
                    "question_key": packets[sid]["question_key"],
                    "tier": tier,
                    "error": abs(pred - int(ref["score"])) >= 2,
                }
            )

        def add_tier_row(tier: str, source: str, items: list[tuple[int, int, dict[str, Any]]]) -> None:
            predictions = [item[0] for item in items]
            gold = [item[1] for item in items]
            metrics = pair_metrics(predictions, gold)
            error_rate = statistics.fmean(abs(pred - truth) >= 2 for pred, truth in zip(predictions, gold))
            tier_rows.append(
                {
                    "recommended_training_use": tier,
                    "calibration_source": source,
                    "n": len(items),
                    "mae_to_adjudicated": fmt(metrics["mae"]),
                    "exact": fmt(metrics["exact"]),
                    "within_one": fmt(metrics["within_one"]),
                    "conflict_error_rate_abs_ge_2": fmt(error_rate),
                    "mean_score_range_width": fmt(
                        statistics.fmean(item[2]["score_range"][1] - item[2]["score_range"][0] for item in items)
                    ),
                    "high_final_confidence_rate": fmt(
                        statistics.fmean(item[2].get("confidence") == "high" for item in items)
                    ),
                }
            )

        for tier, items in sorted(groups.items()):
            add_tier_row(tier, "", items)
        for source, items in sorted(source_groups.items()):
            add_tier_row("", source, items)
    write_table(
        out_dir / "tables" / "exp27j_calibration_tier_validation.csv",
        tier_rows,
        [
            "recommended_training_use",
            "calibration_source",
            "n",
            "mae_to_adjudicated",
            "exact",
            "within_one",
            "conflict_error_rate_abs_ge_2",
            "mean_score_range_width",
            "high_final_confidence_rate",
        ],
    )
    def tier_error_difference(rows: list[dict[str, Any]]) -> float | None:
        high = [bool(row["error"]) for row in rows if row["tier"] == "high_weight"]
        review = [bool(row["error"]) for row in rows if row["tier"] == "review_only"]
        if not high or not review:
            return None
        return statistics.fmean(review) - statistics.fmean(high)

    tier_diff = tier_error_difference(tier_analysis_records)
    tier_diff_low, tier_diff_high = cluster_bootstrap_ci(
        tier_analysis_records,
        tier_error_difference,
        seed=args.seed + 271,
        resamples=args.bootstrap_resamples,
    )
    write_table(
        out_dir / "tables" / "exp27j_calibration_tier_contrast.csv",
        [
            {
                "contrast": "review_only_minus_high_weight_abs_ge_2_error_rate",
                "estimate": fmt(tier_diff),
                "ci95_low": fmt(tier_diff_low),
                "ci95_high": fmt(tier_diff_high),
                "bootstrap_unit": "question_key",
                "bootstrap_resamples": args.bootstrap_resamples,
            }
        ] if tier_analysis_records else [],
        ["contrast", "estimate", "ci95_low", "ci95_high", "bootstrap_unit", "bootstrap_resamples"],
    )

    prevalence_rows: list[dict[str, Any]] = []
    representative = []
    for sid, ref in final.items():
        meta = private.get(sid, {})
        if meta.get("view") != "representative":
            continue
        representative.append({**meta, "question_key": packets[sid]["question_key"], "final": ref})
    if representative:
        indicators: dict[
            str,
            tuple[Callable[[dict[str, Any]], bool], Callable[[dict[str, Any]], bool]],
        ] = {
            "original_label_conflict_abs_ge_2": (
                lambda row: abs(int(row["original_label_5"]) - int(row["final"]["score"])) >= 2,
                lambda row: True,
            ),
            "teacher_human_conflict_abs_ge_2": (
                lambda row: max(
                    [
                        abs(int(value) - int(row["original_label_5"]))
                        for value in [row.get("qwen_score"), row.get("deepseek_score")]
                        if value is not None
                    ]
                )
                >= 2,
                lambda row: row.get("qwen_score") is not None or row.get("deepseek_score") is not None,
            ),
            "evidence_gap_or_hidden_failure": (
                lambda row: row["final"].get("failure_bucket") == "hidden_or_missing_failure",
                lambda row: True,
            ),
            "exp27i_review_only": (
                lambda row: row.get("exp27i_recommended_training_use") == "review_only",
                lambda row: row.get("exp27i_recommended_training_use") is not None,
            ),
        }
        strata = {
            "all": lambda row: True,
            "low_1_2": lambda row: int(row["original_label_5"]) <= 2,
            "mid_3": lambda row: int(row["original_label_5"]) == 3,
            "high_4_5": lambda row: int(row["original_label_5"]) >= 4,
        }
        for stratum, stratum_filter in strata.items():
            subset = [row for row in representative if stratum_filter(row)]
            for name, (predicate, applicable) in indicators.items():
                eligible = [row for row in subset if applicable(row)]
                estimate = weighted_rate(eligible, predicate)
                low, high = cluster_bootstrap_ci(
                    eligible,
                    lambda sampled, fn=predicate: weighted_rate(sampled, fn),
                    seed=args.seed,
                    resamples=args.bootstrap_resamples,
                )
                prevalence_rows.append(
                    {
                        "population_view": "representative_design_weighted",
                        "score_stratum": stratum,
                        "metric": name,
                        "n": len(subset),
                        "eligible_n": len(eligible),
                        "coverage_rate": fmt(safe_div(len(eligible), len(subset))),
                        "population_estimable": str(len(eligible) == len(subset)).lower(),
                        "estimate": fmt(estimate),
                        "ci95_low": fmt(low),
                        "ci95_high": fmt(high),
                        "bootstrap_unit": "question_key",
                        "bootstrap_resamples": args.bootstrap_resamples,
                    }
                )
    write_table(
        out_dir / "tables" / "exp27j_representative_weighted_prevalence.csv",
        prevalence_rows,
        [
            "population_view",
            "score_stratum",
            "metric",
            "n",
            "eligible_n",
            "coverage_rate",
            "population_estimable",
            "estimate",
            "ci95_low",
            "ci95_high",
            "bootstrap_unit",
            "bootstrap_resamples",
        ],
    )

    alt_rows = []
    for sid, ref in final.items():
        meta = private.get(sid, {})
        alt_rows.append(
            {
                "sample_id_hash": sid,
                "view": meta.get("view"),
                "design_weight": meta.get("design_weight"),
                "adjudicated_score": ref.get("score"),
                "original_score": meta.get("original_label_5"),
                "qwen_score": meta.get("qwen_score"),
                "deepseek_score": meta.get("deepseek_score"),
                "exp27i_calibrated_score": meta.get("exp27i_calibrated_score"),
            }
        )
    write_table(
        out_dir / "tables" / "exp27j_alt_test_input.csv",
        alt_rows,
        [
            "sample_id_hash",
            "view",
            "design_weight",
            "adjudicated_score",
            "original_score",
            "qwen_score",
            "deepseek_score",
            "exp27i_calibrated_score",
        ],
    )

    required_adjudication = sum(
        needs_adjudication(reviewer_a[sid], reviewer_b[sid]) for sid in common
    )
    completed_required = sum(
        needs_adjudication(reviewer_a[sid], reviewer_b[sid]) and sid in adjudicated for sid in common
    )
    dual_completion = min(len(reviewer_a), len(reviewer_b)) / 180 if packets else 0.0
    high_row = next((row for row in tier_rows if row["recommended_training_use"] == "high_weight"), None)
    low_row = next((row for row in tier_rows if row["recommended_training_use"] == "low_weight"), None)
    review_row = next((row for row in tier_rows if row["recommended_training_use"] == "review_only"), None)
    high_within = float(high_row["within_one"]) if high_row and high_row["within_one"] else None
    high_mae = float(high_row["mae_to_adjudicated"]) if high_row and high_row["mae_to_adjudicated"] else None
    tier_trust_ordering = bool(
        high_row
        and low_row
        and review_row
        and float(high_row["mae_to_adjudicated"])
        < float(low_row["mae_to_adjudicated"])
        < float(review_row["mae_to_adjudicated"])
        and float(high_row["within_one"])
        > float(low_row["within_one"])
        > float(review_row["within_one"])
    )
    review_gt_high = tier_diff_low is not None and tier_diff_low > 0
    implementation_path = out_dir / "decision" / "exp27j_exp27i_implementation_audit.json"
    implementation = json.loads(implementation_path.read_text(encoding="utf-8")) if implementation_path.exists() else {}
    criteria = {
        "dual_review_completion_ge_0p95": dual_completion >= 0.95,
        "reviewer_qwk_ge_0p70_or_all_required_adjudicated": bool(
            (reviewer_qwk is not None and reviewer_qwk >= 0.70)
            or (required_adjudication > 0 and completed_required == required_adjudication)
        ),
        "high_weight_within_one_ge_0p90": high_within is not None and high_within >= 0.90,
        "high_weight_mae_le_0p50": high_mae is not None and high_mae <= 0.50,
        "review_only_error_rate_gt_high_weight": review_gt_high,
        "tier_trust_ordering_high_gt_low_gt_review": tier_trust_ordering,
        "representative_and_risk_reported_separately": True,
        "exp27i_implementation_issue_documented": bool(implementation),
        "final_reference_complete": len(final) == 180,
    }
    proceed = all(criteria.values())
    if proceed:
        recommended_next_step = "build_exp27i_v2_provenance_preserving_calibration"
    elif len(final) == 180:
        recommended_next_step = "revise_exp27i_calibration_tiers_using_exp27j_then_external_review"
    else:
        recommended_next_step = "complete_independent_audit_before_training"
    decision = {
        "reviewer_a_rows": len(reviewer_a),
        "reviewer_b_rows": len(reviewer_b),
        "dual_review_completion_rate": dual_completion,
        "reviewer_qwk": reviewer_qwk,
        "reviewer_ordinal_alpha": reviewer_alpha,
        "required_adjudication_rows": required_adjudication,
        "completed_required_adjudication_rows": completed_required,
        "final_reference_rows": len(final),
        "criteria": criteria,
        "blocking_criteria": sorted(name for name, passed in criteria.items() if not passed),
        "current_exp27i_292_formal_training_ready": False,
        "proceed_to_qwen3_reranker_downstream_experiment": proceed,
        "recommended_next_step": recommended_next_step,
        "alt_test_status": "prepared_not_run",
        "review_provenance": provenance,
    }
    write_json(out_dir / "decision" / "exp27j_independent_audit_decision.json", decision)

    report = [
        "# Exp27J Independent Audit Analysis",
        "",
        "## Completion",
        "",
        f"- reviewer A: {len(reviewer_a)}/180",
        f"- reviewer B: {len(reviewer_b)}/180",
        f"- required adjudications: {required_adjudication}",
        f"- completed required adjudications: {completed_required}",
        f"- final reference rows: {len(final)}/180",
        "",
        "## Reviewer Reliability",
        "",
        f"- QWK: {fmt(reviewer_qwk)}",
        f"- ordinal alpha (quadratic disagreement): {fmt(reviewer_alpha)}",
        f"- reference status: {provenance.get('reference_status', 'unspecified')}",
        "",
        "## Exp27I Implementation Audit",
        "",
        f"- independent top80 adjudication input found: {implementation.get('independent_top80_adjudication_input_found')}",
        f"- current top80 is rule-based: {implementation.get('current_codex_top80_is_rule_based')}",
        f"- current 292 formal-training-ready: {implementation.get('exp27i_current_292_formal_training_ready')}",
        "",
        "## Decision",
        "",
        f"- proceed to Qwen3-Reranker downstream experiment: {proceed}",
        f"- blocking criteria: {decision['blocking_criteria']}",
        f"- recommendation: {recommended_next_step}",
        "- representative prevalence uses design weights and question-key cluster bootstrap.",
        "- Teacher/Exp27I metrics on the representative view use only rows with existing Exp27I coverage; coverage is reported and missing rows are never treated as non-conflicts.",
        "- risk-enriched metrics are stress-test results and are not population prevalence estimates.",
        "- Alternative Annotator Test input is prepared; no unverified implementation was run.",
    ]
    report.extend(["", "## Source Scores Against Silver Reference", ""])
    for row in source_rows:
        if row.get("view") != "all":
            continue
        report.append(
            f"- {row['source']}: n={row['n']}/{row['expected_n']}, coverage={row['coverage_rate']}, "
            f"MAE={row['mae']}, QWK={row['qwk']}, within-one={row['within_one']}, "
            f"low-to-high={row['low_to_high_rate']}"
        )
    report.extend(["", "## Exp27I Tier Validation", ""])
    for row in tier_rows:
        if not row.get("recommended_training_use"):
            continue
        report.append(
            f"- {row['recommended_training_use']}: n={row['n']}, MAE={row['mae_to_adjudicated']}, "
            f"within-one={row['within_one']}, abs>=2 error={row['conflict_error_rate_abs_ge_2']}"
        )
    report.append(
        f"- review-only minus high-weight abs>=2 error: estimate={fmt(tier_diff)}, "
        f"95% CI=[{fmt(tier_diff_low)}, {fmt(tier_diff_high)}]"
    )
    report.extend(["", "## Representative View", ""])
    for row in prevalence_rows:
        if row.get("score_stratum") != "all":
            continue
        qualifier = "population estimate" if row.get("population_estimable") == "true" else "observed covered subset only"
        report.append(
            f"- {row['metric']}: estimate={row['estimate']}, 95% CI=[{row['ci95_low']}, {row['ci95_high']}], "
            f"coverage={row['eligible_n']}/{row['n']} ({qualifier})"
        )
    report.extend(
        [
            "",
            "## Limitations",
            "",
            "- The reference is a dual-Codex blind-review, model-adjudicated silver reference; no human domain expert participated.",
            "- Original-label conflict rates mean disagreement with this silver reference, not proven human annotation error.",
            "- The 180 rows contain 84 question-key clusters because train has only 118 unique question keys; uncertainty uses question-key cluster bootstrap.",
            "- Only 47/120 representative rows have existing Qwen/DeepSeek/Exp27I coverage. Covered-subset teacher statistics are not full-population prevalence estimates.",
            "- Current Exp27I top-80 wording overstates the implementation: the file is a generated rule-based output, not an external adjudication input.",
        ]
    )
    if not reviewer_a or not reviewer_b:
        report.extend(
            [
                "",
                "## Pending",
                "",
                "Filled dual reviews are missing. No semantic labels or performance metrics were fabricated.",
            ]
        )
    write_text(out_dir / "reports" / "exp27j_independent_audit_analysis_report.md", "\n".join(report))
    if not args.allow_missing_reviews and (len(reviewer_a) < 180 or len(reviewer_b) < 180):
        raise SystemExit("Exp27J dual reviews are incomplete")
    if not args.allow_missing_reviews and adjudication_templates:
        raise SystemExit(f"Exp27J requires {len(adjudication_templates)} additional final adjudications")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Exp27J independent blind reviews.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--reviewer-a", type=Path)
    parser.add_argument("--reviewer-b", type=Path)
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument("--review-provenance", type=Path)
    parser.add_argument("--allow-missing-reviews", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = parser.parse_args()
    args.reviewer_a = args.reviewer_a or args.out_dir / "annotation" / "exp27j_reviewer_a_filled.jsonl"
    args.reviewer_b = args.reviewer_b or args.out_dir / "annotation" / "exp27j_reviewer_b_filled.jsonl"
    args.adjudication = args.adjudication or args.out_dir / "annotation" / "exp27j_final_adjudication_filled.jsonl"
    args.review_provenance = (
        args.review_provenance
        or args.out_dir / "annotation" / "exp27j_review_provenance.json"
    )
    print(json.dumps(analyze(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
