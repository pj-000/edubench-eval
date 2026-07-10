"""Analyze full representative teacher coverage against the Exp27J silver reference."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence.build_exp27i_codex_calibrated_dataset import (  # noqa: E402
    choose_calibration,
    load_stage,
    provider_view,
)
from thesis_exp.src.edujudge.utils.io import write_csv, write_json, write_text  # noqa: E402


DEFAULT_EXP27I_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27i_target_aware_teacher_audit_361_seed42"
)
DEFAULT_EXP27J_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27j_independent_audit_seed42"
)
DEFAULT_OUT_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27k_representative_teacher_validation_seed42"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_int(value: Any) -> int | None:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return None
    return score if 1 <= score <= 5 else None


def fmt(value: float | None, digits: int = 6) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def round_score(value: float) -> int:
    return max(1, min(5, int(math.floor(value + 0.5))))


def weighted_mean(values: list[float], weights: list[float]) -> float | None:
    total = sum(weights)
    return sum(value * weight for value, weight in zip(values, weights)) / total if total else None


def weighted_qwk(gold: list[int], pred: list[int], weights: list[float]) -> float | None:
    if not gold or len(gold) != len(pred) or len(gold) != len(weights):
        return None
    total = sum(weights)
    if not total:
        return None
    observed = [[0.0] * 5 for _ in range(5)]
    hist_gold = [0.0] * 5
    hist_pred = [0.0] * 5
    for left, right, weight in zip(gold, pred, weights):
        observed[left - 1][right - 1] += weight
        hist_gold[left - 1] += weight
        hist_pred[right - 1] += weight
    obs = 0.0
    exp = 0.0
    for i in range(5):
        for j in range(5):
            penalty = ((i - j) / 4.0) ** 2
            obs += penalty * observed[i][j] / total
            exp += penalty * (hist_gold[i] * hist_pred[j]) / (total * total)
    return 1.0 - obs / exp if exp else (1.0 if obs == 0 else None)


def score_metrics(rows: list[dict[str, Any]], predictor: Callable[[dict[str, Any]], int | None]) -> dict[str, Any]:
    aligned = [(row, predictor(row)) for row in rows]
    aligned = [(row, pred) for row, pred in aligned if pred is not None]
    if not aligned:
        return {"n": 0}
    gold = [int(row["final_score"]) for row, _pred in aligned]
    pred = [int(value) for _row, value in aligned]
    weights = [float(row.get("analysis_weight") or 1.0) for row, _pred in aligned]
    errors = [right - left for left, right in zip(gold, pred)]
    low_den = sum(weight for value, weight in zip(gold, weights) if value <= 2)
    high_den = sum(weight for value, weight in zip(gold, weights) if value >= 4)
    return {
        "n": len(aligned),
        "coverage": len(aligned) / len(rows) if rows else 0.0,
        "mae": weighted_mean([abs(value) for value in errors], weights),
        "exact": weighted_mean([float(value == 0) for value in errors], weights),
        "within_one": weighted_mean([float(abs(value) <= 1) for value in errors], weights),
        "severe_error_rate": weighted_mean([float(abs(value) >= 2) for value in errors], weights),
        "qwk": weighted_qwk(gold, pred, weights),
        "signed_bias": weighted_mean([float(value) for value in errors], weights),
        "low_to_high_rate": (
            sum(weight for left, right, weight in zip(gold, pred, weights) if left <= 2 and right >= 4) / low_den
            if low_den
            else None
        ),
        "high_to_low_rate": (
            sum(weight for left, right, weight in zip(gold, pred, weights) if left >= 4 and right <= 2) / high_den
            if high_den
            else None
        ),
    }


def weighted_average_precision(labels: list[int], scores: list[float], weights: list[float]) -> float | None:
    positive_weight = sum(weight for label, weight in zip(labels, weights) if label == 1)
    if not positive_weight:
        return None
    groups: dict[float, list[tuple[int, float]]] = defaultdict(list)
    for label, score, weight in zip(labels, scores, weights):
        groups[score].append((label, weight))
    seen = 0.0
    true_positive = 0.0
    area = 0.0
    previous_recall = 0.0
    for score in sorted(groups, reverse=True):
        group = groups[score]
        seen += sum(weight for _label, weight in group)
        true_positive += sum(weight for label, weight in group if label == 1)
        recall = true_positive / positive_weight
        precision = true_positive / seen if seen else 0.0
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return area


def brier(labels: list[int], scores: list[float], weights: list[float]) -> float | None:
    return weighted_mean([(score - label) ** 2 for label, score in zip(labels, scores)], weights)


def ece(labels: list[int], scores: list[float], weights: list[float], bins: int = 5) -> float | None:
    total = sum(weights)
    if not total:
        return None
    value = 0.0
    for idx in range(bins):
        low = idx / bins
        high = (idx + 1) / bins
        selected = [
            pos
            for pos, score in enumerate(scores)
            if low <= score < high or (idx == bins - 1 and score == 1.0)
        ]
        if not selected:
            continue
        bin_weights = [weights[pos] for pos in selected]
        mass = sum(bin_weights)
        confidence = weighted_mean([scores[pos] for pos in selected], bin_weights) or 0.0
        accuracy = weighted_mean([float(labels[pos]) for pos in selected], bin_weights) or 0.0
        value += mass / total * abs(confidence - accuracy)
    return value


def review_at_fraction(
    rows: list[dict[str, Any]], scores: list[float], fraction: float = 0.20
) -> tuple[float | None, float | None, int]:
    count = max(1, math.ceil(len(rows) * fraction)) if rows else 0
    ranked = sorted(zip(rows, scores), key=lambda item: item[1], reverse=True)[:count]
    if not ranked:
        return None, None, 0
    flagged_positive = sum(abs(int(row["original_score"]) - int(row["final_score"])) >= 2 for row, _ in ranked)
    all_positive = sum(abs(int(row["original_score"]) - int(row["final_score"])) >= 2 for row in rows)
    return flagged_positive / len(ranked), flagged_positive / all_positive if all_positive else None, len(ranked)


def agreement_pattern(human: int, qwen: int, deepseek: int) -> str:
    if human == qwen == deepseek:
        return "triple_exact"
    if qwen == deepseek and abs(qwen - human) <= 1:
        return "teacher_consensus_human_adjacent"
    if qwen == deepseek:
        return "teacher_consensus_human_far"
    if max(human, qwen, deepseek) - min(human, qwen, deepseek) <= 1:
        return "three_way_adjacent_band"
    if human in {qwen, deepseek}:
        return "human_one_teacher_exact_other_far"
    if abs(qwen - deepseek) >= 2:
        return "teacher_gap_ge2"
    return "other_disagreement"


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def old_teacher_view(row: dict[str, Any], provider: str) -> dict[str, Any]:
    return {
        "provider": provider,
        "score": as_int(row.get(f"{provider}_score")),
        "reason": row.get(f"{provider}_reason") or "",
        "major_failures": row.get(f"{provider}_major_failures") or [],
        "score_cap": None,
        "target_confusion_risk": "possible" if row.get("target_issue_flag") else "none",
        "audit_target_confusion_detected": bool(row.get("target_issue_flag")),
        "audit_reason": row.get(f"{provider}_audit_reason") or "",
    }


def structural_evidence_checks(blind: dict[str, Any], evaluator_output: str) -> dict[str, bool | None]:
    if not blind:
        return {
            "span_valid": None,
            "score_failure_consistent": None,
            "score_cap_consistent": None,
            "low_score_rubric_clause_present": None,
        }
    score = as_int(blind.get("teacher_score"))
    span = blind.get("evidence_span")
    failures = blind.get("major_failures") if isinstance(blind.get("major_failures"), list) else []
    score_cap = as_int(blind.get("score_cap"))
    span_valid = None if span in {None, ""} else str(span) in evaluator_output
    score_failure_consistent = None
    if score is not None:
        score_failure_consistent = not (score <= 2 and (not failures or "no_major_failure" in failures))
    return {
        "span_valid": span_valid,
        "score_failure_consistent": score_failure_consistent,
        "score_cap_consistent": None if score is None else score_cap is None or score <= score_cap,
        "low_score_rubric_clause_present": (
            None if score is None or score > 2 else bool(str(blind.get("rubric_clause") or "").strip())
        ),
    }


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest = read_csv(args.exp27j_dir / "tables" / "exp27j_adjudicated_manifest.csv")
    old_rows = {
        str(row["sample_id"]): row
        for row in read_jsonl(args.exp27i_dir / "data" / "exp27i_teacher_audited_361_calibrated_train.jsonl")
    }
    new_qwen = provider_view(args.out_dir, "qwen")
    new_deepseek = provider_view(args.out_dir, "deepseek")
    old_blind = {provider: load_stage(args.exp27i_dir, provider, "blind") for provider in ["qwen", "deepseek"]}
    new_blind = {provider: load_stage(args.out_dir, provider, "blind") for provider in ["qwen", "deepseek"]}
    new_packets = {
        str(row["sample_id"]): row
        for row in read_jsonl(args.out_dir / "packets" / "exp27d_v4_repilot_blind_packets.jsonl")
    }

    combined: list[dict[str, Any]] = []
    for source in manifest:
        sid = source["sample_id"]
        old = old_rows.get(sid)
        q = old_teacher_view(old, "qwen") if old else new_qwen.get(sid, {})
        d = old_teacher_view(old, "deepseek") if old else new_deepseek.get(sid, {})
        human = int(source["original_score"])
        q_score = as_int(q.get("score"))
        d_score = as_int(d.get("score"))
        if q_score is None or d_score is None:
            raise ValueError(f"Missing complete teacher scores for {sid}: qwen={q_score} deepseek={d_score}")
        if old:
            v1_score = int(old["calibrated_score"])
            v1_tier = str(old["recommended_training_use"])
            target_issue = bool(old.get("target_issue_flag"))
            provenance = "exp27i_existing"
            evaluator_output = str(old.get("answer") or "")
        else:
            calibration = choose_calibration(sid, human, q, d, top80=False)
            v1_score = int(calibration["calibrated_score"])
            v1_tier = str(calibration["recommended_training_use"])
            target_issue = bool(calibration["target_issue_flag"])
            provenance = "exp27k_new"
            evaluator_output = str(new_packets[sid].get("teacher_input", {}).get("answer") or "")
        q_blind = (old_blind["qwen"] if old else new_blind["qwen"]).get(sid, {})
        d_blind = (old_blind["deepseek"] if old else new_blind["deepseek"]).get(sid, {})
        q_evidence = structural_evidence_checks(q_blind, evaluator_output)
        d_evidence = structural_evidence_checks(d_blind, evaluator_output)
        median_score = int(statistics.median([human, q_score, d_score]))
        teacher_mean_score = round_score((q_score + d_score) / 2)
        max_gap = max(abs(q_score - human), abs(d_score - human), abs(q_score - d_score))
        combined.append(
            {
                "sample_id": sid,
                "question_key_hash": source["question_key_hash"],
                "view": source["view"],
                "analysis_weight": float(source.get("design_weight") or 1.0) if source["view"] == "representative" else 1.0,
                "original_score": human,
                "final_score": int(source["final_score"]),
                "qwen_score": q_score,
                "deepseek_score": d_score,
                "dual_teacher_mean_score": teacher_mean_score,
                "human_teacher_median_score": median_score,
                "exp27i_v1_score": v1_score,
                "exp27i_v1_tier": v1_tier,
                "target_issue_flag": target_issue,
                "teacher_gap": abs(q_score - d_score),
                "max_three_way_gap": max_gap,
                "agreement_pattern": agreement_pattern(human, q_score, d_score),
                "teacher_provenance": provenance,
                **{f"qwen_{key}": value for key, value in q_evidence.items()},
                **{f"deepseek_{key}": value for key, value in d_evidence.items()},
            }
        )
    return combined


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_rows(args)
    if len(rows) != 180:
        raise ValueError(f"Expected 180 Exp27J rows, got {len(rows)}")
    representative = [row for row in rows if row["view"] == "representative"]
    risk = [row for row in rows if row["view"] == "risk_enriched"]
    if len(representative) != 120 or len(risk) != 60:
        raise ValueError(f"Expected representative=120 risk=60, got {len(representative)} and {len(risk)}")

    methods: dict[str, Callable[[dict[str, Any]], int | None]] = {
        "original_human": lambda row: int(row["original_score"]),
        "qwen_blind": lambda row: int(row["qwen_score"]),
        "deepseek_blind": lambda row: int(row["deepseek_score"]),
        "dual_teacher_rounded_mean": lambda row: int(row["dual_teacher_mean_score"]),
        "human_qwen_deepseek_median": lambda row: int(row["human_teacher_median_score"]),
        "exp27i_v1_rule_fusion": lambda row: int(row["exp27i_v1_score"]),
    }
    all_unweighted = [{**row, "analysis_weight": 1.0} for row in rows]
    views = {"representative": representative, "risk_enriched": risk, "all_unweighted": all_unweighted}
    metric_rows: list[dict[str, Any]] = []
    for view_name, view_rows in views.items():
        for method, predictor in methods.items():
            values = score_metrics(view_rows, predictor)
            metric_rows.append(
                {
                    "view": view_name,
                    "method": method,
                    **{key: fmt(value) if isinstance(value, float) else value for key, value in values.items()},
                }
            )
    write_csv(args.out_dir / "tables" / "exp27k_protocol_score_metrics.csv", metric_rows)

    risk_signals: dict[str, Callable[[dict[str, Any]], float]] = {
        "qwen_human_gap": lambda row: min(1.0, abs(row["qwen_score"] - row["original_score"]) / 4),
        "deepseek_human_gap": lambda row: min(1.0, abs(row["deepseek_score"] - row["original_score"]) / 4),
        "teacher_teacher_gap": lambda row: min(1.0, row["teacher_gap"] / 4),
        "max_three_way_gap": lambda row: min(1.0, row["max_three_way_gap"] / 4),
        "exp27i_v1_tier_proxy": lambda row: {
            "high_weight": 0.05,
            "low_weight": 0.35,
            "review_only": 0.80,
        }.get(row["exp27i_v1_tier"], 0.90),
    }
    risk_rows: list[dict[str, Any]] = []
    for view_name, view_rows in views.items():
        labels = [int(abs(row["original_score"] - row["final_score"]) >= 2) for row in view_rows]
        weights = [float(row.get("analysis_weight") or 1.0) for row in view_rows]
        baseline_rate = weighted_mean([float(value) for value in labels], weights)
        for signal_name, signal_fn in risk_signals.items():
            scores = [signal_fn(row) for row in view_rows]
            precision, recall, flagged = review_at_fraction(view_rows, scores)
            risk_rows.append(
                {
                    "view": view_name,
                    "signal": signal_name,
                    "n": len(view_rows),
                    "positive_rate": fmt(baseline_rate),
                    "auprc": fmt(weighted_average_precision(labels, scores, weights)),
                    "brier_proxy": fmt(brier(labels, scores, weights)),
                    "ece_proxy": fmt(ece(labels, scores, weights)),
                    "review_fraction": "0.20",
                    "flagged_rows": flagged,
                    "review_precision": fmt(precision),
                    "review_recall": fmt(recall),
                    "note": "risk scores are heuristic proxies, not fitted probabilities",
                }
            )
    write_csv(args.out_dir / "tables" / "exp27k_risk_signal_metrics.csv", risk_rows)

    pattern_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in representative:
        grouped[row["agreement_pattern"]].append(row)
    for pattern, group in sorted(grouped.items()):
        errors = [abs(row["human_teacher_median_score"] - row["final_score"]) for row in group]
        severe = sum(value >= 2 for value in errors)
        low, high = wilson_interval(severe, len(group))
        within = sum(value <= 1 for value in errors) / len(group)
        suggested = "review_only"
        if len(group) >= 10 and within >= 0.90 and high is not None and high <= 0.15:
            suggested = "high_weight_candidate"
        elif len(group) >= 5 and within >= 0.75 and high is not None and high <= 0.35:
            suggested = "low_weight_candidate"
        pattern_rows.append(
            {
                "agreement_pattern": pattern,
                "n": len(group),
                "candidate_target": "human_qwen_deepseek_median",
                "mae_to_silver": fmt(statistics.fmean(errors)),
                "within_one": fmt(within),
                "severe_error_rate": fmt(severe / len(group)),
                "severe_error_wilson_low": fmt(low),
                "severe_error_wilson_high": fmt(high),
                "suggested_use_before_external_review": suggested,
            }
        )
    write_csv(args.out_dir / "tables" / "exp27k_pattern_reliability.csv", pattern_rows)

    evidence_rows: list[dict[str, Any]] = []
    for view_name, view_rows in views.items():
        for provider in ["qwen", "deepseek"]:
            def rate(field: str) -> tuple[int, float | None]:
                values = [row.get(f"{provider}_{field}") for row in view_rows]
                observed = [bool(value) for value in values if value is not None]
                return len(observed), statistics.fmean(observed) if observed else None

            span_n, span_rate = rate("span_valid")
            failure_n, failure_rate = rate("score_failure_consistent")
            cap_n, cap_rate = rate("score_cap_consistent")
            rubric_n, rubric_rate = rate("low_score_rubric_clause_present")
            evidence_rows.append(
                {
                    "view": view_name,
                    "provider": provider,
                    "n": len(view_rows),
                    "verbatim_span_observed_n": span_n,
                    "verbatim_span_valid_rate": fmt(span_rate),
                    "score_failure_consistency_n": failure_n,
                    "score_failure_consistency_rate": fmt(failure_rate),
                    "score_cap_consistency_n": cap_n,
                    "score_cap_consistency_rate": fmt(cap_rate),
                    "low_score_rubric_clause_n": rubric_n,
                    "low_score_rubric_clause_present_rate": fmt(rubric_rate),
                    "note": "structural checks only; semantic correctness still requires independent review",
                }
            )
    write_csv(args.out_dir / "tables" / "exp27k_evidence_consistency.csv", evidence_rows)

    case_rows = [
        {
            key: row[key]
            for key in [
                "sample_id",
                "question_key_hash",
                "view",
                "original_score",
                "final_score",
                "qwen_score",
                "deepseek_score",
                "dual_teacher_mean_score",
                "human_teacher_median_score",
                "exp27i_v1_score",
                "exp27i_v1_tier",
                "target_issue_flag",
                "teacher_gap",
                "max_three_way_gap",
                "agreement_pattern",
                "teacher_provenance",
                "qwen_span_valid",
                "qwen_score_failure_consistent",
                "qwen_score_cap_consistent",
                "qwen_low_score_rubric_clause_present",
                "deepseek_span_valid",
                "deepseek_score_failure_consistent",
                "deepseek_score_cap_consistent",
                "deepseek_low_score_rubric_clause_present",
            ]
        }
        for row in rows
    ]
    write_csv(args.out_dir / "tables" / "exp27k_case_level_comparison.csv", case_rows)

    rep_metrics = {(row["method"]): row for row in metric_rows if row["view"] == "representative"}
    rep_risk = {(row["signal"]): row for row in risk_rows if row["view"] == "representative"}
    v1_ap = float(rep_risk["exp27i_v1_tier_proxy"]["auprc"])
    simple_names = ["qwen_human_gap", "deepseek_human_gap", "teacher_teacher_gap", "max_three_way_gap"]
    best_simple_name = max(simple_names, key=lambda name: float(rep_risk[name]["auprc"]))
    best_simple_ap = float(rep_risk[best_simple_name]["auprc"])
    v1_mae = float(rep_metrics["exp27i_v1_rule_fusion"]["mae"])
    median_mae = float(rep_metrics["human_qwen_deepseek_median"]["mae"])
    criteria = {
        "representative_teacher_coverage_120_of_120": all(
            row["qwen_score"] is not None and row["deepseek_score"] is not None for row in representative
        ),
        "exp27i_v1_auprc_beats_best_simple_by_0p05": v1_ap >= best_simple_ap + 0.05,
        "exp27i_v1_score_mae_better_than_naive_median": v1_mae < median_mae,
        "exp27i_v1_is_empirically_calibrated": False,
        "external_human_expert_reference_included": False,
    }
    blocking = [name for name, passed in criteria.items() if not passed]
    decision = {
        "experiment": "exp27k_representative_teacher_validation",
        "criteria": criteria,
        "blocking_criteria": blocking,
        "representative_rows": len(representative),
        "risk_enriched_rows": len(risk),
        "best_simple_risk_signal": best_simple_name,
        "best_simple_risk_signal_auprc": best_simple_ap,
        "exp27i_v1_tier_proxy_auprc": v1_ap,
        "exp27i_v1_representative_mae": v1_mae,
        "naive_median_representative_mae": median_mae,
        "proceed_to_formal_qwen3_reranker_training": False,
        "recommended_next_step": "cross_fitted_tier_recalibration_then_external_expert_review",
        "test_label_read": False,
        "dev_label_used": False,
        "dev_test_files_opened": False,
        "silver_reference_used_in_teacher_prompt": False,
    }
    write_json(args.out_dir / "decision" / "exp27k_protocol_validation_decision.json", decision)

    report = [
        "# Exp27K Representative Teacher Protocol Validation",
        "",
        "Exp27K completes teacher coverage for the 120-row representative Exp27J probability sample and compares",
        "single-teacher, simple fusion, and Exp27I-v1 rule fusion against the model-adjudicated Exp27J silver reference.",
        "",
        "## Coverage",
        "",
        f"- representative rows with Qwen and DeepSeek scores: {len(representative)}/120",
        f"- risk-enriched rows retained as a separate stress view: {len(risk)}/60",
        "- no Exp27J silver field was used in teacher prompts.",
        "",
        "## Representative Score Comparison",
        "",
    ]
    for name in methods:
        row = rep_metrics[name]
        report.append(
            f"- {name}: MAE={row['mae']}, QWK={row['qwk']}, within-one={row['within_one']}, "
            f"severe-error={row['severe_error_rate']}"
        )
    report.extend(
        [
            "",
            "## Risk-Signal Comparison",
            "",
            f"- best simple signal: {best_simple_name}, AUPRC={best_simple_ap:.6f}",
            f"- Exp27I-v1 tier proxy AUPRC: {v1_ap:.6f}",
            "- Brier/ECE values are diagnostic proxies because the v1 scores were not statistically fitted probabilities.",
            "- evidence tables check structural grounding and score-field consistency only; they do not certify semantic correctness.",
            "",
            "## Decision",
            "",
            "- formal Qwen3-Reranker training remains blocked.",
            "- next: fit revised confidence tiers with question-key-aware cross-fitting, then obtain external expert review",
            "  for high-impact ambiguous cases before constructing the formal 3326-row in-place training variants.",
            "- Exp27J remains a model-adjudicated silver reference, not human-expert gold.",
            "- representative metrics use inverse-probability design weights; risk-enriched and all-unweighted views do not.",
        ]
    )
    write_text(args.out_dir / "reports" / "exp27k_protocol_validation_report.md", "\n".join(report))
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Exp27K representative teacher protocol validation.")
    parser.add_argument("--exp27i-dir", type=Path, default=DEFAULT_EXP27I_DIR)
    parser.add_argument("--exp27j-dir", type=Path, default=DEFAULT_EXP27J_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    print(json.dumps(analyze(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
